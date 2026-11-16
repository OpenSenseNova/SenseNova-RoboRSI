# Copyright 2026 The SenseNova-RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Real Agent and LIBERO containers deny cross-case files and forged scores."""

import json
import subprocess
import threading
import time

import pytest

from roborsi.benchmarks.registry import BenchmarkRegistry
from roborsi.evaluation.benchmark_client import BenchmarkClient
from roborsi.evaluation.docker_agent import DockerAgentLauncher
from roborsi.evaluation.docker_runtime import DockerExecutor
from roborsi.evaluation.orchestration import CaseOrchestrator
from roborsi.evaluation.store import ResultStore
from roborsi.utils.config import get_project_env, require_project_env
from roborsi.utils.rpc import RpcError

pytestmark = [
    pytest.mark.skipif(
        get_project_env("ROBORSI_RUN_LIBERO_BENCHMARK") != "1",
        reason="requires LIBERO and the isolation Agent image",
    ),
    pytest.mark.timeout(240),
]


@pytest.mark.parametrize("mode", ["files", "forge_score", "cancel"])
def test_agent_cannot_read_other_cases_modify_scores_or_bypass_runner(
    tmp_path, monkeypatch, mode
):
    registry = BenchmarkRegistry([require_project_env("ROBORSI_LIBERO_REGISTRY")])
    assert registry.lock("libero").task_message("0", 0).max_steps == 3
    monkeypatch.setenv("TEST_MODEL_TOKEN", "integration-test-value")
    monkeypatch.setenv("TEST_RUNNER_SECRET", "do-not-inherit")
    store = ResultStore(tmp_path / "results")
    other = tmp_path / "results/other/case/private.txt"
    other.parent.mkdir(parents=True)
    other.write_text("other case original")
    launcher = DockerAgentLauncher(
        require_project_env("ROBORSI_ISOLATION_AGENT_IMAGE"),
        "roborsi_isolation_agent:make_agent",
        credential_env=["TEST_MODEL_TOKEN"],
    )
    executor = DockerExecutor()
    start_agent, start_adapter = launcher.start, executor.start_adapter
    agents, benchmarks, probe_paths = [], [], []
    stop, cancellation = threading.Event(), threading.Event()

    def track_adapter(*args, **kwargs):
        service = start_adapter(*args, **kwargs)
        benchmarks.append(service)
        # The host can reach the port, but it still needs this container's token.
        with pytest.raises(RpcError, match="authentication required"):
            BenchmarkClient(service.client.transport.endpoint).tasks(timeout_s=3)
        return service

    def track_agent(directory, **kwargs):
        benchmark = benchmarks[-1]
        inspected = json.loads(
            subprocess.check_output(
                ["docker", "inspect", benchmark.container_id], text=True, timeout=10
            )
        )[0]
        network = next(iter(inspected["NetworkSettings"]["Networks"].values()))
        score = directory.parent / "native/protected.json"
        score.parent.mkdir()
        score.write_text('{"score":0}')
        (directory / "probe.json").write_text(
            json.dumps(
                {
                    "mode": mode,
                    "other_case": str(other),
                    "runner_score": str(score),
                    "benchmark_endpoints": [
                        benchmark.client.transport.endpoint,
                        f"http://{network['IPAddress']}:8000",
                    ],
                }
            )
        )
        probe_paths.append(directory / "probe-result.json")
        service = start_agent(directory, **kwargs)
        agents.append(service)
        return service

    def cancel_after_probe():
        while not stop.wait(0.05):
            if probe_paths and probe_paths[0].is_file():
                cancellation.set()
                return

    monkeypatch.setattr(executor, "start_adapter", track_adapter)
    monkeypatch.setattr(launcher, "start", track_agent)
    watcher = (
        threading.Thread(target=cancel_after_probe, daemon=True)
        if mode == "cancel"
        else None
    )
    if watcher:
        watcher.start()
    started = time.monotonic()
    try:
        result = CaseOrchestrator(registry, store, launcher, executor).run(
            "isolation",
            mode,
            mode,
            "libero",
            "0",
            0,
            launcher.agent_version,
            timeout_s=120,
            case_timeout_s=180,
            cancel_event=cancellation,
        )
    finally:
        stop.set()
        if watcher:
            watcher.join(2)
    expected = {
        "files": "task_failure",
        "forge_score": "agent_error",
        "cancel": "cancelled",
    }[mode]
    assert result.status == expected, result.as_dict()
    assert result.score == (0 if mode == "files" else None)
    assert store.load("isolation", mode) == result
    assert store.trace_state("isolation", mode).complete
    probe = json.loads(probe_paths[0].read_text())
    assert len(probe["denied"]) == 11
    assert all(probe["denied"].values()), probe
    assert all(
        probe[key]
        for key in (
            "allowed_credential",
            "blocked_credential_absent",
            "benchmark_token_absent",
            "python_isolated",
            "simulator_absent",
        )
    )
    assert probe["episode_id"] == mode
    assert other.read_text() == "other case original"
    directory = store.root / "isolation" / mode
    assert (directory / "native/protected.json").read_text() == '{"score":0}'
    assert json.loads((directory / "agent/result.json").read_text())["score"] == 1
    if mode == "files":
        assert result.runtime.agent.verified_after_close
        assert result.runtime.observed.agent_version == launcher.agent_version
        assert len(list((directory / "media").rglob("*.bin"))) == 8
        assert (directory / "agent/agent-closed").is_file()
    assert len(agents) == len(benchmarks) == 1
    for service in [*agents, *benchmarks]:
        assert not subprocess.check_output(
            ["docker", "ps", "-aq", "--filter", f"id={service.container_id}"],
            text=True,
            timeout=10,
        ).strip()
    assert not subprocess.check_output(
        [
            "docker",
            "network",
            "ls",
            "--filter",
            f"name=^{agents[0].network}$",
            "--format",
            "{{.ID}}",
        ],
        text=True,
        timeout=10,
    ).strip()
    (tmp_path / "timing.json").write_text(
        json.dumps({"elapsed_s": time.monotonic() - started})
    )
