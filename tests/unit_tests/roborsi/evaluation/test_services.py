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

"""Real subprocesses exchange RoboRSI messages through existing HTTP RPC."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Iterator

import pytest

from roborsi.evaluation.benchmark_client import BenchmarkTransport
from roborsi.evaluation.messages import (
    Action,
    Observation,
    StartResult,
    StepResult,
    Task,
    to_envelope,
)
from roborsi.evaluation.services import (
    AdapterClient,
    AdapterFacade,
    AgentClient,
    AgentFacade,
    RoboRSIRequestError,
)
from roborsi.utils.daemon import pick_free_port
from roborsi.utils.rpc import (
    RpcError,
    RpcTransportError,
    make_rpc_client,
    wait_for_ready,
)


def test_terminal_feedback_validates_identity_step_and_replays_once() -> None:
    from tests.unit_tests.roborsi.evaluation.service_child import DemoAgent

    class FeedbackAgent(DemoAgent):
        def __init__(self):
            super().__init__()
            self.feedback = []

        def observe_end(self, step):
            self.feedback.append(step)

    agent = FeedbackAgent()
    facade = AgentFacade(agent)
    task = Task(
        benchmark="demo", task_id="place", seed=7, instruction="Place", max_steps=2
    )

    def wire(message, request_id, episode_id="ep"):
        return json.dumps(
            to_envelope(
                message, benchmark="demo", episode_id=episode_id, request_id=request_id
            ).as_dict()
        )

    facade.reset(wire(task, "reset"))
    obs = Observation(benchmark="demo", episode_id="ep", step_index=0)
    facade.act(wire(obs, "act"))
    end = StepResult(
        observation=obs.model_copy(update={"step_index": 1}),
        terminated=True,
        truncated=False,
        termination_reason="native_success",
    )
    with pytest.raises(RoboRSIRequestError):
        other = end.model_copy(
            update={
                "observation": end.observation.model_copy(
                    update={"episode_id": "other"}
                )
            }
        )
        facade.observe_end(wire(other, "wrong-episode", "other"))
    with pytest.raises(RoboRSIRequestError, match="out of order"):
        facade.observe_end(
            wire(end.model_copy(update={"observation": obs}), "wrong-step")
        )
    with pytest.raises(RoboRSIRequestError, match="terminal"):
        facade.observe_end(
            wire(
                StepResult(
                    observation=end.observation, terminated=False, truncated=False
                ),
                "not-ended",
            )
        )
    request = wire(end, "final")
    assert facade.observe_end(request) == facade.observe_end(request)
    assert agent.feedback == [end]
    with pytest.raises(RoboRSIRequestError, match="ended"):
        facade.act(wire(end.observation, "late-action"))
    with pytest.raises(RoboRSIRequestError, match="reused"):
        facade.observe_end(wire(end.model_copy(update={"reward": 1.0}), "final"))
    facade.close_episode()
    assert facade.observe_end(request)["ok"]


@contextmanager
def _service(
    kind: str,
    *,
    factory: str | None = None,
    env_overrides: dict[str, str] | None = None,
) -> Iterator[object]:
    port = pick_free_port()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        f"{Path(__file__).resolve().parents[4]}{os.pathsep}"
        f"{environment.get('PYTHONPATH', '')}"
    )
    environment.update(env_overrides or {})
    factory = factory or ("make_demo_agent" if kind == "agent" else "make_demo_adapter")
    command = [
        sys.executable,
        "-m",
        "roborsi.evaluation.service_host",
        "--kind",
        kind,
        "--factory",
        f"tests.unit_tests.roborsi.evaluation.service_child:{factory}",
        "--port",
        str(port),
        "--parent-watch",
    ]
    with tempfile.TemporaryDirectory(prefix="roborsi-service-test-") as directory:
        if kind == "agent":
            command.extend(["--case-dir", directory])
        child = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=environment,
        )
        yield from _connected_service(child, port, benchmark=kind == "benchmark")


def _connected_service(
    child: subprocess.Popen, port: int, *, benchmark=False
) -> Iterator[object]:
    rpc = (BenchmarkTransport if benchmark else make_rpc_client)(
        f"http://127.0.0.1:{port}"
    )
    try:
        wait_for_ready(rpc, timeout_s=10.0, poll_interval_s=0.05)
        yield rpc
    finally:
        if child.poll() is None:
            try:
                if benchmark:
                    child.stdin.close()
                    child.stdin = None
                else:
                    rpc.call("shutdown", timeout_s=2.0)
            except RpcError:
                child.terminate()
        try:
            _, stderr = child.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            child.kill()
            _, stderr = child.communicate(timeout=5.0)
        rpc.close()
        assert child.returncode == 0, stderr.decode(errors="replace")


def test_agent_and_adapter_exchange_native_action_across_processes() -> None:
    task = Task(
        benchmark="demo",
        task_id="place",
        seed=7,
        instruction="Place the block.",
        max_steps=2,
        native={"suite": "demo"},
        capabilities={"action_schema": {"type": "object"}},
    )
    with ExitStack() as stack:
        agent = AgentClient(stack.enter_context(_service("agent")))
        adapter = AdapterClient(stack.enter_context(_service("adapter")))
        try:
            start = adapter.start(task, "case-1", timeout_s=3.0)
            assert start.task == task
            agent.reset(task, "case-1", timeout_s=3.0)
            action = agent.act(start.observation, timeout_s=3.0)
            assert action.data == {"native": {"__ndarray__": "literal-field"}}
            step = adapter.step(action, timeout_s=3.0)
            assert step.terminated and step.observation.step_index == 1
            score = adapter.finish(timeout_s=3.0)
            assert score.success and score.score == 0.75
            assert score.task.seed == 7
        finally:
            agent.close()
            adapter.close()


def test_adapter_receives_the_requested_episode_id_across_processes() -> None:
    task = Task(
        benchmark="demo",
        task_id="place",
        seed=7,
        instruction="Place",
        max_steps=2,
        native={"suite": "demo_suite"},
    )
    with _service("adapter") as rpc:
        adapter = AdapterClient(rpc)
        try:
            started = adapter.start(task, "another-episode", timeout_s=3.0)
            assert started.observation.episode_id == "another-episode"
            assert started.task.native == {"suite": "demo_suite"}
            action = Action(
                benchmark="demo",
                episode_id="another-episode",
                step_index=0,
                data={"native": {"__ndarray__": "literal-field"}},
            )
            assert adapter.step(action, timeout_s=3.0).terminated
            assert adapter.finish(timeout_s=3.0).episode_id == "another-episode"
        finally:
            adapter.close()


def test_adapter_rejects_wrong_episode_and_step_without_advancing() -> None:
    task = Task(
        benchmark="demo", task_id="place", seed=7, instruction="Place", max_steps=2
    )
    with _service("adapter") as rpc:
        adapter = AdapterClient(rpc)
        adapter.start(task, "case-1", timeout_s=3.0)
        with pytest.raises(RpcError, match="identity mismatch"):
            adapter.step(
                Action(benchmark="demo", episode_id="other", step_index=0, data={}),
                timeout_s=3.0,
            )
        with pytest.raises(RpcError, match="out of order"):
            adapter.step(
                Action(benchmark="demo", episode_id="case-1", step_index=1, data={}),
                timeout_s=3.0,
            )
        adapter.close()


def test_service_rejects_unsupported_version_and_bad_request_before_start() -> None:
    task = Task(
        benchmark="demo", task_id="place", seed=7, instruction="Place", max_steps=2
    )
    wire = to_envelope(task, benchmark="demo", episode_id="case-1").as_dict()
    with _service("adapter") as rpc:
        assert rpc.call("healthz", timeout_s=3.0)["roborsi_protocol_versions"] == [1]
        with pytest.raises(RpcError) as unsupported:
            rpc.call(
                "start",
                args=(json.dumps({**wire, "schema_version": 2}),),
                timeout_s=3.0,
            )
        assert unsupported.value.code == "unsupported_version"
        with pytest.raises(RpcError) as malformed:
            rpc.call("start", args=("{",), timeout_s=3.0)
        assert malformed.value.code == "invalid_request"
        assert AdapterClient(rpc).start(task, "case-1", timeout_s=3.0).task == task


def test_native_agent_failure_has_a_distinct_code() -> None:
    task = Task(
        benchmark="demo", task_id="place", seed=7, instruction="Place", max_steps=2
    )
    observation = Observation(benchmark="demo", episode_id="case-1", step_index=0)
    with _service("agent", factory="make_failing_agent") as rpc:
        agent = AgentClient(rpc)
        agent.reset(task, "case-1", timeout_s=3.0)
        with pytest.raises(RpcError, match="demo agent action failed") as failure:
            agent.act(observation, timeout_s=3.0)
        assert failure.value.code == "agent_error"


def test_native_adapter_failure_has_a_distinct_code() -> None:
    task = Task(
        benchmark="demo", task_id="place", seed=7, instruction="Place", max_steps=2
    )
    with _service("adapter", factory="make_failing_adapter") as rpc:
        with pytest.raises(RpcError, match="demo adapter start failed") as failure:
            AdapterClient(rpc).start(task, "case-1", timeout_s=3.0)
        assert failure.value.code == "adapter_error"


@pytest.mark.parametrize("advertisement", [{}, {"heart_protocol_versions": [1]}])
def test_client_rejects_service_without_roborsi_version_advertisement(
    advertisement,
) -> None:
    class OldService:
        def call(self, method, args=(), *, timeout_s=None):
            assert method == "healthz"
            return {"status": "ok", **advertisement}

    task = Task(
        benchmark="demo", task_id="place", seed=7, instruction="Place", max_steps=2
    )
    with pytest.raises(RpcError) as failure:
        AdapterClient(OldService()).start(task, "case-1", timeout_s=3.0)
    assert failure.value.code == "unsupported_version"


def test_adapter_rejects_actual_seed_mismatch_and_closes_native_resources() -> None:
    class WrongSeed:
        closes = 0

        def start(self, task: Task, episode_id: str) -> StartResult:
            return StartResult(
                task=task.model_copy(update={"seed": task.seed + 1}),
                observation=Observation(
                    benchmark=task.benchmark, episode_id=episode_id, step_index=0
                ),
            )

        def close(self) -> None:
            self.closes += 1

    native = WrongSeed()
    facade = AdapterFacade(native)
    task = Task(
        benchmark="demo", task_id="place", seed=7, instruction="Place", max_steps=2
    )
    wire = to_envelope(task, benchmark="demo", episode_id="case-1").as_dict()
    with pytest.raises(ValueError, match="different task or seed"):
        facade.start(json.dumps(wire))
    facade.close()
    assert native.closes == 1


def test_adapter_rejects_wrong_actual_episode_and_closes_native_resources() -> None:
    class WrongEpisode:
        closes = 0

        def start(self, task: Task, episode_id: str) -> StartResult:
            return StartResult(
                task=task,
                observation=Observation(
                    benchmark=task.benchmark, episode_id="another-episode", step_index=0
                ),
            )

        def close(self) -> None:
            self.closes += 1

    native = WrongEpisode()
    facade = AdapterFacade(native)
    task = Task(
        benchmark="demo", task_id="place", seed=7, instruction="Place", max_steps=2
    )
    wire = to_envelope(task, benchmark="demo", episode_id="case-1").as_dict()
    with pytest.raises(ValueError, match="identity"):
        facade.start(json.dumps(wire))
    assert native.closes == 1


def test_retried_requests_replay_without_advancing_the_native_episode() -> None:
    task = Task(
        benchmark="demo", task_id="place", seed=7, instruction="Place", max_steps=2
    )
    with _service("adapter") as rpc:
        start_wire = json.dumps(
            to_envelope(
                task, benchmark="demo", episode_id="case-1", request_id="start-1"
            ).as_dict()
        )
        first = rpc.call("start", args=(start_wire,), timeout_s=3.0)
        assert rpc.call("start", args=(start_wire,), timeout_s=3.0) == first

        action = Action(
            benchmark="demo",
            episode_id="case-1",
            step_index=0,
            data={"native": {"__ndarray__": "literal-field"}},
        )
        step_wire = json.dumps(
            to_envelope(
                action, benchmark="demo", episode_id="case-1", request_id="step-1"
            ).as_dict()
        )
        step = rpc.call("step", args=(step_wire,), timeout_s=3.0)
        assert rpc.call("step", args=(step_wire,), timeout_s=3.0) == step
        with pytest.raises(RpcError, match="out of order|ended"):
            rpc.call(
                "step",
                args=(step_wire.replace('"step-1"', '"step-2"'),),
                timeout_s=3.0,
            )
        with pytest.raises(RpcError, match="reused"):
            rpc.call(
                "start", args=(start_wire.replace("place", "other"),), timeout_s=3.0
            )

        score = rpc.call("finish", args=("finish-1",), timeout_s=3.0)
        assert rpc.call("finish", args=("finish-1",), timeout_s=3.0) == score


def test_client_rejects_response_with_another_request_id() -> None:
    task = Task(
        benchmark="demo", task_id="place", seed=7, instruction="Place", max_steps=2
    )

    class WrongReply:
        def call(self, method, args=(), *, timeout_s=None):
            if method == "healthz":
                return {"status": "ok", "roborsi_protocol_versions": [1]}
            assert method == "start"
            return json.dumps(
                to_envelope(
                    StartResult(
                        task=task,
                        observation=Observation(
                            benchmark="demo", episode_id="case-1", step_index=0
                        ),
                    ),
                    benchmark="demo",
                    episode_id="case-1",
                    request_id="late-response",
                ).as_dict()
            )

    with pytest.raises(ValueError, match="request_id mismatch"):
        AdapterClient(WrongReply()).start(task, "case-1", timeout_s=3.0)


def test_clients_replay_lost_responses_without_repeating_native_actions() -> None:
    task = Task(
        benchmark="demo", task_id="place", seed=7, instruction="Place", max_steps=2
    )

    class LostFirstReply:
        def __init__(self, rpc) -> None:
            self.rpc = rpc
            self.lost: set[str] = set()
            self.requests: dict[str, list[tuple]] = {}

        def call(self, method, args=(), *, timeout_s=None):
            if method in {"reset", "act", "start", "step", "finish"}:
                self.requests.setdefault(method, []).append(args)
                reply = self.rpc.call(method, args=args, timeout_s=timeout_s)
                if method not in self.lost:
                    self.lost.add(method)
                    raise RpcTransportError(method, "response lost after execution")
                return reply
            return self.rpc.call(method, args=args, timeout_s=timeout_s)

        def close(self):
            self.rpc.close()

    with ExitStack() as stack:
        agent_rpc = LostFirstReply(stack.enter_context(_service("agent")))
        adapter_rpc = LostFirstReply(stack.enter_context(_service("adapter")))
        agent = AgentClient(agent_rpc)
        adapter = AdapterClient(adapter_rpc)
        try:
            started = adapter.start(task, "case-1", timeout_s=3.0)
            agent.reset(task, "case-1", timeout_s=3.0)
            action = agent.act(started.observation, timeout_s=3.0)
            assert adapter.step(action, timeout_s=3.0).terminated
            assert adapter.finish(timeout_s=3.0).success
        finally:
            agent.close()
            adapter.close()
    for rpc in (agent_rpc, adapter_rpc):
        assert all(requests[0] == requests[1] for requests in rpc.requests.values())
        assert all(len(requests) == 2 for requests in rpc.requests.values())


def test_client_does_not_retry_a_remote_backend_failure() -> None:
    task = Task(
        benchmark="demo", task_id="place", seed=7, instruction="Place", max_steps=2
    )

    class BackendFailure:
        calls = 0

        def call(self, method, args=(), *, timeout_s=None):
            if method == "healthz":
                return {"status": "ok", "roborsi_protocol_versions": [1]}
            self.calls += 1
            raise RpcError(method, "native failure", code="adapter_error")

    rpc = BackendFailure()
    with pytest.raises(RpcError, match="native failure"):
        AdapterClient(rpc).start(task, "case-1", timeout_s=3.0)
    assert rpc.calls == 1


def test_lost_native_failure_replays_the_original_error_code() -> None:
    task = Task(
        benchmark="demo", task_id="place", seed=7, instruction="Place", max_steps=2
    )

    class LostErrorReply:
        def __init__(self, rpc) -> None:
            self.rpc = rpc
            self.requests: list[tuple] = []

        def call(self, method, args=(), *, timeout_s=None):
            if method == "start":
                self.requests.append(args)
            try:
                return self.rpc.call(method, args=args, timeout_s=timeout_s)
            except RpcError as exc:
                if method == "start" and len(self.requests) == 1:
                    assert exc.code == "adapter_error"
                    raise RpcTransportError(method, "error response lost") from exc
                raise

    with _service("adapter", factory="make_failing_adapter") as rpc:
        flaky = LostErrorReply(rpc)
        with pytest.raises(RpcError, match="demo adapter start failed") as failure:
            AdapterClient(flaky).start(task, "case-1", timeout_s=3.0)
        assert failure.value.code == "adapter_error"
        assert len(flaky.requests) == 2
        assert flaky.requests[0] == flaky.requests[1]


def test_adapter_close_failure_retries_cleanup_without_reopening_episode() -> None:
    class RetryClose:
        closes = 0

        def start(self, task, episode_id):
            return StartResult(
                task=task,
                observation=Observation(
                    benchmark=task.benchmark, episode_id=episode_id, step_index=0
                ),
            )

        def close(self):
            self.closes += 1
            if self.closes < 3:
                raise TimeoutError("native cleanup incomplete")

    native = RetryClose()
    facade = AdapterFacade(native)
    task = Task(
        benchmark="demo", task_id="place", seed=7, instruction="Place", max_steps=2
    )

    def wire(message, request_id):
        return json.dumps(
            to_envelope(
                message, benchmark="demo", episode_id="ep", request_id=request_id
            ).as_dict()
        )

    facade.start(wire(task, "start"))
    for attempt in (1, 2):
        with pytest.raises(TimeoutError, match="native cleanup incomplete"):
            facade.close_episode()
        assert native.closes == attempt
        with pytest.raises(RoboRSIRequestError, match="one episode"):
            facade.start(wire(task, f"restart-{attempt}"))
        with pytest.raises(RoboRSIRequestError, match="identity"):
            facade.step(
                wire(
                    Action(benchmark="demo", episode_id="ep", step_index=0, data={}),
                    "step",
                )
            )
    assert facade.close_episode() == {"ok": True}
    facade.close()
    assert native.closes == 3
