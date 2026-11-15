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

"""Real LIBERO action counts stay unchanged by retries and rejected requests."""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest

from roborsi.benchmarks.registry import BenchmarkRegistry
from roborsi.evaluation.agent_process import PythonAgentLauncher
from roborsi.evaluation.benchmark_client import BenchmarkClient
from roborsi.evaluation.docker_runtime import DockerExecutor
from roborsi.evaluation.messages import (
    Action,
    NativeResult,
    from_envelope,
    to_envelope,
)
from roborsi.evaluation.orchestration import CaseOrchestrator
from roborsi.evaluation.protocol import InteractionEnvelope
from roborsi.evaluation.store import ResultStore
from roborsi.utils.config import get_project_env, require_project_env
from roborsi.utils.rpc import RpcError
from tests.unit_tests.roborsi.evaluation.http_fault_proxy import lost_responses

pytestmark = [
    pytest.mark.skipif(
        get_project_env("ROBORSI_RUN_LIBERO_BENCHMARK") != "1",
        reason="requires the LIBERO HTTP image and assets",
    ),
    pytest.mark.timeout(240),
]


def _assert_removed(container):
    assert not subprocess.check_output(
        ["docker", "ps", "-aq", "--filter", f"id={container.container_id}"],
        text=True,
        timeout=10,
    ).strip()


def test_native_steps_ignore_replays_conflicts_and_invalid_versions(tmp_path):
    lock = BenchmarkRegistry([require_project_env("ROBORSI_LIBERO_REGISTRY")]).lock(
        "libero"
    )
    task = lock.task_message("0", 0)
    assert task.max_steps == 3, "prepare the three-step integration registry"
    episode_id = "wire-order"
    base = f"/v1/episodes/{episode_id}"
    with DockerExecutor().start_adapter(lock, tmp_path) as container:
        client = container.client
        request = client.transport.request
        start_body = to_envelope(
            task, benchmark="libero", episode_id=episode_id, request_id="start"
        ).as_dict()
        with pytest.raises(RpcError):
            request("POST", "/v1/episodes", body={**start_body, "schema_version": 2})
        assert not request("GET", "/v1/health")[1]["busy"]
        with pytest.raises(RpcError, match="episode not found"):
            client.status(episode_id)

        first = request("POST", "/v1/episodes", body=start_body, timeout_s=120)
        assert request("POST", "/v1/episodes", body=start_body) == first
        assert first[1]["request_id"] == "start"

        def action(index, request_id, **updates):
            wire = to_envelope(
                Action(
                    benchmark="libero",
                    episode_id=episode_id,
                    step_index=index,
                    data={"vector": [0, 0, 0, 0, 0, 0, -1]},
                ),
                benchmark="libero",
                episode_id=episode_id,
                request_id=request_id,
            ).as_dict()
            return {**wire, **updates}

        def rejected(body):
            before = client.status(episode_id)
            with pytest.raises(RpcError):
                request("POST", base + "/actions", body=body)
            assert client.status(episode_id) == before

        rejected(action(0, "wrong-version", schema_version=2))
        rejected(action(0, "wrong-episode", episode_id="another-episode"))
        rejected(action(1, "future-step"))
        with pytest.raises(RpcError, match="end before finish"):
            request("POST", base + "/finish", body={"request_id": "too-early"})
        assert client.result(episode_id)[0] == 202

        for index in range(3):
            body = action(index, f"step-{index}")

            def submit():
                return request("POST", base + "/actions", body=body, timeout_s=30)

            # Concurrent callers and a later replay must observe one native step.
            with ThreadPoolExecutor(max_workers=3) as pool:
                replies = list(pool.map(lambda _: submit(), range(3)))
            assert all(reply == replies[0] for reply in replies)
            assert submit() == replies[0]
            assert client.status(episode_id)["step_index"] == index + 1
            rejected(action(index, f"old-step-new-id-{index}"))
            conflicting = action(index, f"step-{index}")
            conflicting["payload"]["data"]["vector"][0] = 0.1
            rejected(conflicting)

        finish = request("POST", base + "/finish", body={"request_id": "finish"})
        assert (
            request("POST", base + "/finish", body={"request_id": "finish"}) == finish
        )
        assert client.result(episode_id) == finish
        native = from_envelope(InteractionEnvelope.from_dict(finish[1]))
        assert isinstance(native, NativeResult)
        assert native.task == task and not native.success and native.score == 0
        artifacts = client.artifacts(episode_id=episode_id)
        for info in artifacts:
            client.download(info, tmp_path / "download")
        assert len([a for a in artifacts if a.kind == "image"]) == 8
        assert len([a for a in artifacts if a.kind == "log"]) == 5
        assert (
            json.loads((tmp_path / "download/native/score.json").read_text())["steps"]
            == 3
        )
        logs = [
            json.loads((tmp_path / "download" / a.ref).read_text())
            for a in artifacts
            if a.kind == "log"
        ]
        assert [entry["method"] for entry in logs] == [
            "start",
            "step",
            "step",
            "step",
            "finish",
        ]
        request("POST", base + "/close", body={})
        client.delete(episode_id)
        with pytest.raises(RpcError, match="episode not found"):
            request("POST", base + "/actions", body=action(2, "step-2"))
    _assert_removed(container)


def test_runner_recovers_truncated_native_replies_without_extra_actions(
    tmp_path, monkeypatch
):
    registry = BenchmarkRegistry([require_project_env("ROBORSI_LIBERO_REGISTRY")])
    assert registry.lock("libero").task_message("0", 0).max_steps == 3
    store = ResultStore(tmp_path / "results")
    launcher = PythonAgentLauncher("examples.roborsi.hold_agent:make_agent")
    executor = DockerExecutor()
    start_adapter, start_agent = executor.start_adapter, launcher.start
    containers, agents, requests = [], [], []

    @contextmanager
    def track_adapter(*args, **kwargs):
        with start_adapter(*args, **kwargs) as container:
            containers.append(container)
            with lost_responses(
                container.client.transport.endpoint,
                token=container.client.transport.token,
            ) as (
                endpoint,
                calls,
            ):
                container.client = BenchmarkClient(endpoint)
                yield container
                requests.extend(calls)

    def track_agent(*args, **kwargs):
        process = start_agent(*args, **kwargs)
        agents.append(process)
        return process

    monkeypatch.setattr(executor, "start_adapter", track_adapter)
    monkeypatch.setattr(launcher, "start", track_agent)
    result = CaseOrchestrator(registry, store, launcher, executor).run(
        "replay",
        "case",
        "recover-replies",
        "libero",
        "0",
        0,
        "hold-agent",
        timeout_s=120,
        case_timeout_s=180,
    )
    assert result.status == "task_failure" and result.score == 0, result.as_dict()
    assert store.load("replay", "case") == result
    assert store.trace_state("replay", "case").complete
    directory = store.root / "replay/case"
    assert json.loads((directory / "native/score.json").read_text())["steps"] == 3
    assert len([a for a in result.benchmark_artifacts if a.kind == "image"]) == 8
    assert len([a for a in result.benchmark_artifacts if a.kind == "log"]) == 5
    operations = [
        r
        for r in requests
        if r["method"] == "POST"
        and r["path"].endswith(("/v1/episodes", "/actions", "/finish"))
    ]
    assert len(operations) == 10
    for first, retry in zip(operations[::2], operations[1::2], strict=True):
        assert first["dropped"] and not retry["dropped"]
        assert first["body"] == retry["body"]
    (tmp_path / "http-attempts.json").write_text(json.dumps(operations, indent=2))
    assert len(agents) == len(containers) == 1
    assert agents[0].daemon.poll() is not None
    _assert_removed(containers[0])
