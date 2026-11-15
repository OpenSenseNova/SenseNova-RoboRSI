# Copyright 2026 The RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The single-case coordinator records native and operational outcomes."""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from threading import Event

import pytest

from roborsi.evaluation.deadline import CaseDeadline
from roborsi.evaluation.media import CaseMediaChannel
from roborsi.evaluation.messages import (
    Action,
    NativeResult,
    Observation,
    Provenance,
    StartResult,
    Task,
)
from roborsi.evaluation.runner import CaseRunner
from roborsi.evaluation.services import AdapterClient, AgentClient
from roborsi.evaluation.store import ResultStore
from roborsi.utils.rpc import RpcError, RpcTransportError
from tests.unit_tests.roborsi.evaluation.test_services import _service


def _task() -> Task:
    return Task(
        benchmark="demo",
        task_id="place",
        seed=7,
        instruction="Place the block.",
        max_steps=2,
        native={"suite": "demo"},
        capabilities={"action_schema": {"type": "object"}},
    )


def test_runner_drives_two_processes_and_publishes_native_score(tmp_path: Path) -> None:
    store = ResultStore(tmp_path)
    with ExitStack() as stack:
        agent = AgentClient(stack.enter_context(_service("agent")))
        adapter = AdapterClient(stack.enter_context(_service("adapter")))
        store.claim("run1", "case1", _task(), Provenance(agent_version="demo-agent"))
        result = CaseRunner(store).run(
            "run1",
            "case1",
            "case-1",
            _task(),
            Provenance(agent_version="demo-agent"),
            agent,
            adapter,
            timeout_s=3.0,
            preclaimed=True,
        )
    assert result.status == "success" and result.score == 0.75
    assert store.load("run1", "case1") == result
    assert store.list_cases("run1") == {"case1": "published"}
    assert store.trace_state("run1", "case1").complete
    events = store.trace_events("run1", "case1")
    messages = [
        event["payload"]["message"]
        for event in events
        if event["type"] == "roborsi_message"
    ]
    assert [message["kind"] for message in messages] == [
        "start_result",
        "action",
        "step_result",
        "native_result",
    ]
    assert messages[1]["payload"]["data"] == {
        "native": {"__ndarray__": "literal-field"}
    }
    assert messages[2]["payload"]["observation"]["data"] == {"state": [1.0]}
    assert messages[3]["payload"]["details"] == {"native_metric": 0.75}


def test_runner_delivers_final_observation_to_agent_over_http(tmp_path: Path) -> None:
    report = tmp_path / "feedback.json"
    with ExitStack() as stack:
        agent = AgentClient(
            stack.enter_context(
                _service(
                    "agent",
                    factory="make_feedback_agent",
                    env_overrides={"ROBORSI_TEST_FEEDBACK": str(report)},
                )
            )
        )
        adapter = AdapterClient(stack.enter_context(_service("adapter")))
        store = ResultStore(tmp_path / "results")
        result = CaseRunner(store).run(
            "run",
            "case",
            "case-1",
            _task(),
            Provenance(),
            agent,
            adapter,
            timeout_s=3.0,
        )
    assert result.status == "success"
    feedback = json.loads(report.read_text())
    assert feedback["observation"]["episode_id"] == "case-1"
    assert feedback["observation"]["step_index"] == 1
    assert feedback["observation"]["data"] == {"state": [1.0]}
    assert feedback["terminated"] is True
    assert feedback["termination_reason"] == "native_success"


class _FakeAgent:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.closed = False
        self.reset_called = False

    def reset(self, task: Task, episode_id: str, *, timeout_s: float) -> None:
        self.reset_called = True

    def act(self, observation: Observation, *, timeout_s: float):
        raise self.error or RuntimeError("agent failed")

    def close(self) -> None:
        self.closed = True


class _FakeAdapter:
    def __init__(self, *, wrong_seed: bool = False) -> None:
        self.wrong_seed = wrong_seed
        self.closed = False

    def start(self, task: Task, episode_id: str, *, timeout_s: float) -> StartResult:
        actual = (
            task.model_copy(update={"seed": task.seed + 1}) if self.wrong_seed else task
        )
        return StartResult(
            task=actual,
            observation=Observation(
                benchmark=task.benchmark, episode_id=episode_id, step_index=0
            ),
        )

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    ("agent_error", "expected_code"),
    [
        (RpcError("act", "model unavailable"), "agent_error"),
        (TimeoutError("model timed out"), "timeout"),
    ],
)
def test_runner_classifies_agent_failures_and_closes_services(
    tmp_path: Path, agent_error: Exception, expected_code: str
) -> None:
    agent, adapter = _FakeAgent(agent_error), _FakeAdapter()
    store = ResultStore(tmp_path)
    result = CaseRunner(store).run(
        "run1", "case1", "case-1", _task(), Provenance(), agent, adapter
    )
    assert result.status == "agent_error"
    assert result.error is not None and result.error.code == expected_code
    assert result.score is None
    assert agent.closed and adapter.closed
    assert store.load("run1", "case1") == result


def test_trace_write_failure_stops_actions_and_preserves_error_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    armed = [False]
    original_open = Path.open

    def fail_one_append(path, mode="r", *args, **kwargs):
        if armed[0] and path.name == "events.jsonl" and mode == "ab":
            armed[0] = False
            raise OSError("trace disk is full")
        return original_open(path, mode, *args, **kwargs)

    class ActingAgent(_FakeAgent):
        def act(self, observation, *, timeout_s):
            armed[0] = True
            return Action(
                benchmark=observation.benchmark,
                episode_id=observation.episode_id,
                step_index=observation.step_index,
                data={},
            )

    class UnsteppedAdapter(_FakeAdapter):
        def step(self, action, *, timeout_s):
            pytest.fail("action executed after recording failed")

    monkeypatch.setattr(Path, "open", fail_one_append)
    store = ResultStore(tmp_path)
    agent, adapter = ActingAgent(), UnsteppedAdapter()
    result = CaseRunner(store).run(
        "run1", "case1", "case-1", _task(), Provenance(), agent, adapter
    )
    assert result.status == "infrastructure_error" and result.score is None
    assert result.error.details["stage"] == "trace.record"
    assert agent.closed and adapter.closed
    assert store.load("run1", "case1") == result
    assert store.trace_state("run1", "case1").status == "incomplete"
    assert not store.trace_state("run1", "case1").complete
    assert store.trace_events("run1", "case1")[-1]["type"] == "run_end"


def test_runner_records_lost_agent_transport_as_retryable_infrastructure(
    tmp_path: Path,
) -> None:
    agent = _FakeAgent(RpcTransportError("act", "connection lost"))
    adapter = _FakeAdapter()
    store = ResultStore(tmp_path)
    result = CaseRunner(store).run(
        "run1", "case1", "case-1", _task(), Provenance(), agent, adapter
    )
    assert result.status == "infrastructure_error"
    assert result.error is not None
    assert result.error.code == "infrastructure_error"
    assert result.error.retryable
    assert agent.closed and adapter.closed
    assert store.load("run1", "case1") == result


def test_runner_rejects_actual_task_mismatch_before_agent_reset(tmp_path: Path) -> None:
    agent, adapter = _FakeAgent(), _FakeAdapter(wrong_seed=True)
    store = ResultStore(tmp_path)
    result = CaseRunner(store).run(
        "run1", "case1", "case-1", _task(), Provenance(), agent, adapter
    )
    assert result.status == "infrastructure_error"
    assert result.error is not None and result.error.code == "adapter_error"
    assert not agent.reset_called
    assert agent.closed and adapter.closed


@pytest.mark.parametrize(
    ("stage", "remote_code", "expected_status", "expected_code"),
    [
        (
            "adapter",
            "unsupported_version",
            "infrastructure_error",
            "unsupported_version",
        ),
        ("adapter", "invalid_request", "infrastructure_error", "invalid_request"),
        ("adapter", "adapter_error", "infrastructure_error", "adapter_error"),
        ("agent", "agent_error", "agent_error", "agent_error"),
        ("agent", "adapter_error", "agent_error", "agent_error"),
    ],
)
def test_runner_preserves_only_valid_service_error_codes(
    tmp_path: Path,
    stage: str,
    remote_code: str,
    expected_status: str,
    expected_code: str,
) -> None:
    class FailingAdapter(_FakeAdapter):
        def start(self, task: Task, episode_id: str, *, timeout_s: float):
            raise RpcError("start", "service failed", code=remote_code)

    agent = _FakeAgent(RpcError("act", "service failed", code=remote_code))
    adapter = FailingAdapter() if stage == "adapter" else _FakeAdapter()
    result = CaseRunner(ResultStore(tmp_path)).run(
        "run1", "case1", "case-1", _task(), Provenance(), agent, adapter
    )
    assert result.status == expected_status
    assert result.error is not None and result.error.code == expected_code
    assert result.score is None and agent.closed and adapter.closed


def test_runner_closes_clients_when_case_is_already_claimed(tmp_path: Path) -> None:
    store = ResultStore(tmp_path)
    store.claim("run1", "case1", _task(), Provenance())
    agent, adapter = _FakeAgent(), _FakeAdapter()
    with pytest.raises(FileExistsError):
        CaseRunner(store).run(
            "run1", "case1", "case-1", _task(), Provenance(), agent, adapter
        )
    assert agent.closed and adapter.closed
    assert not agent.reset_called


def test_runner_records_missing_native_evidence_as_infrastructure_error(
    tmp_path: Path,
) -> None:
    class ScoredAdapter(_FakeAdapter):
        def step(self, action, *, timeout_s: float):
            from roborsi.evaluation.messages import StepResult

            return StepResult(
                observation=Observation(
                    benchmark=action.benchmark,
                    episode_id=action.episode_id,
                    step_index=action.step_index + 1,
                ),
                terminated=True,
                truncated=False,
                termination_reason="native_success",
            )

        def finish(self, *, timeout_s: float) -> NativeResult:
            return NativeResult(
                task=_task(),
                episode_id="case-1",
                success=True,
                score=1.0,
                source="native/result.json",
                source_sha256="a" * 64,
            )

    class ActingAgent(_FakeAgent):
        def act(self, observation: Observation, *, timeout_s: float):
            from roborsi.evaluation.messages import Action

            return Action(
                benchmark=observation.benchmark,
                episode_id=observation.episode_id,
                step_index=observation.step_index,
                data={},
            )

    store = ResultStore(tmp_path)
    result = CaseRunner(store).run(
        "run1", "case1", "case-1", _task(), Provenance(), ActingAgent(), ScoredAdapter()
    )
    assert result.status == "infrastructure_error" and result.score is None
    assert result.error is not None and "evidence invalid" in result.error.message
    assert store.load("run1", "case1") == result


@pytest.mark.parametrize("cancelled", [False, True])
def test_runner_discards_action_returned_after_case_deadline_or_cancel(
    tmp_path: Path, cancelled: bool
) -> None:
    now = [10.0]
    cancellation = Event()

    class ActingAgent(_FakeAgent):
        def act(self, observation: Observation, *, timeout_s: float) -> Action:
            if cancelled:
                cancellation.set()
            else:
                now[0] += 6.0
            return Action(
                benchmark=observation.benchmark,
                episode_id=observation.episode_id,
                step_index=observation.step_index,
                data={},
            )

    class UnsteppedAdapter(_FakeAdapter):
        def step(self, action: Action, *, timeout_s: float) -> None:
            pytest.fail("expired or cancelled action reached Adapter")

    agent, adapter = ActingAgent(), UnsteppedAdapter()
    deadline = CaseDeadline(5.0, cancel_event=cancellation, clock=lambda: now[0])
    store = ResultStore(tmp_path)
    result = CaseRunner(store).run(
        "run1",
        "case1",
        "case-1",
        _task(),
        Provenance(),
        agent,
        adapter,
        deadline=deadline,
    )

    assert result.status == ("cancelled" if cancelled else "infrastructure_error")
    assert result.error is not None
    assert result.error.code == ("cancelled" if cancelled else "timeout")
    assert result.error.details["stage"] == "agent.act"
    assert result.score is None and agent.closed and adapter.closed
    assert store.load("run1", "case1") == result


def test_runner_does_not_publish_score_after_cleanup_exceeds_deadline(
    tmp_path: Path,
) -> None:
    from roborsi.evaluation.messages import StepResult

    now = [10.0]

    class ActingAgent(_FakeAgent):
        def act(self, observation: Observation, *, timeout_s: float) -> Action:
            return Action(
                benchmark=observation.benchmark,
                episode_id=observation.episode_id,
                step_index=observation.step_index,
                data={},
            )

        def close(self) -> None:
            now[0] += 6.0
            super().close()

    class ScoringAdapter(_FakeAdapter):
        def step(self, action: Action, *, timeout_s: float) -> StepResult:
            return StepResult(
                observation=Observation(
                    benchmark=action.benchmark,
                    episode_id=action.episode_id,
                    step_index=action.step_index + 1,
                ),
                terminated=True,
                truncated=False,
                termination_reason="native_success",
            )

        def finish(self, *, timeout_s: float) -> NativeResult:
            return NativeResult(
                task=_task(), episode_id="case-1", success=True, score=1.0
            )

    store = ResultStore(tmp_path)
    result = CaseRunner(store).run(
        "run1",
        "case1",
        "case-1",
        _task(),
        Provenance(),
        ActingAgent(),
        ScoringAdapter(),
        deadline=CaseDeadline(5.0, clock=lambda: now[0]),
    )

    assert result.status == "infrastructure_error" and result.score is None
    assert result.error is not None and result.error.code == "timeout"
    assert store.load("run1", "case1") == result


def test_runner_rejects_corrupt_initial_media_before_agent_reset(
    tmp_path: Path,
) -> None:
    store = ResultStore(tmp_path)
    task, provenance = _task(), Provenance()
    store.claim("run1", "case1", task, provenance)
    channel = CaseMediaChannel(tmp_path / "run1" / "case1" / "media")
    reference = channel.write("case-1", 0, b"camera", "image/png")
    (tmp_path / "run1" / "case1" / reference.ref).write_bytes(b"damage")

    class CorruptMediaAdapter(_FakeAdapter):
        def start(
            self, task: Task, episode_id: str, *, timeout_s: float
        ) -> StartResult:
            return StartResult(
                task=task,
                observation=Observation(
                    benchmark=task.benchmark,
                    episode_id=episode_id,
                    step_index=0,
                    media=(reference,),
                ),
            )

    agent, adapter = _FakeAgent(), CorruptMediaAdapter()
    result = CaseRunner(store).run(
        "run1",
        "case1",
        "case-1",
        task,
        provenance,
        agent,
        adapter,
        media_channel=channel,
        preclaimed=True,
    )

    assert result.status == "infrastructure_error"
    assert result.error is not None and result.error.code == "adapter_error"
    assert not agent.reset_called and agent.closed and adapter.closed
    assert store.load("run1", "case1") == result
