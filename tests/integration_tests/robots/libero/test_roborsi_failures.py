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

"""Inject faults after real LIBERO reset and prove results and runtime cleanup."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from threading import Event, Thread

import pytest

from roborsi.benchmarks.registry import BenchmarkRegistry
from roborsi.evaluation.agent_process import PythonAgentLauncher
from roborsi.evaluation.docker_runtime import DockerExecutor
from roborsi.evaluation.orchestration import CaseOrchestrator
from roborsi.evaluation.store import ResultStore
from roborsi.utils.config import get_project_env, require_project_env

pytestmark = [
    pytest.mark.skipif(
        get_project_env("ROBORSI_RUN_LIBERO_BENCHMARK") != "1",
        reason="requires a built LIBERO image and downloaded assets",
    ),
    pytest.mark.timeout(180),
]


def _docker(*args: str) -> str:
    return subprocess.run(
        ["docker", *args], check=True, capture_output=True, text=True, timeout=15
    ).stdout.strip()


@pytest.mark.parametrize(
    ("mode", "status", "code", "stage"),
    [
        ("agent_error", "agent_error", "agent_error", "agent.act"),
        ("agent_crash", "infrastructure_error", "infrastructure_error", "agent.act"),
        ("invalid_action", "infrastructure_error", "adapter_error", "adapter.step"),
        (
            "adapter_crash",
            "infrastructure_error",
            "infrastructure_error",
            "adapter.step",
        ),
        ("agent_cancel", "cancelled", "cancelled", "agent.act"),
        ("adapter_cancel", "cancelled", "cancelled", "adapter.step"),
        ("adapter_late_cancel", "cancelled", "cancelled", "adapter.step"),
        ("agent_timeout", "infrastructure_error", "timeout", "agent.act"),
        ("adapter_timeout", "infrastructure_error", "timeout", "adapter.step"),
    ],
)
def test_real_failures_cancel_timeout_and_cleanup(
    tmp_path: Path, monkeypatch, mode: str, status: str, code: str, stage: str
) -> None:
    registry = BenchmarkRegistry([Path(require_project_env("ROBORSI_LIBERO_REGISTRY"))])
    lock = registry.lock("libero")
    identity = lock.list_tasks()[0]
    store = ResultStore(tmp_path / "results")
    case_dir = tmp_path / "results" / "faults" / mode
    cancellation, stop, step_started = Event(), Event(), Event()
    timings: dict[str, float] = {}
    processes, containers, inspections = [], [], []
    launcher = PythonAgentLauncher(
        "tests.integration_tests.robots.libero.fault_agent:make_agent",
        env_overrides={"ROBORSI_TEST_FAULT": mode},
    )
    executor = DockerExecutor()
    start_agent, start_adapter = launcher.start, executor.start_adapter

    def track_agent(*args, **kwargs):
        process = start_agent(*args, **kwargs)
        processes.append(process)
        act = process.client.act

        def observed_act(*args, **kwargs):
            try:
                return act(*args, **kwargs)
            finally:
                if stage == "agent.act":
                    timings["rpc_returned"] = time.monotonic()

        monkeypatch.setattr(process.client, "act", observed_act)
        return process

    def track_adapter(*args, **kwargs):
        container = start_adapter(*args, **kwargs)
        containers.append(container)
        inspections.append(json.loads(_docker("inspect", container.container_id))[0])
        step = container.client.step

        def observed_step(*args, **kwargs):
            if mode.startswith("adapter_"):
                operation = "kill" if mode == "adapter_crash" else "pause"
                _docker(operation, container.container_id)
            step_started.set()
            try:
                return step(*args, **kwargs)
            finally:
                timings["rpc_returned"] = time.monotonic()

        monkeypatch.setattr(container.client, "step", observed_step)
        return container

    monkeypatch.setattr(launcher, "start", track_agent)
    monkeypatch.setattr(executor, "start_adapter", track_adapter)
    trigger_errors = []

    def cancel_after_request():
        try:
            while not stop.wait(0.01):
                ready = (
                    (case_dir / "agent" / "act.started").exists()
                    if mode == "agent_cancel"
                    else step_started.is_set()
                )
                if ready:
                    # Give the paused Adapter request time to reach its socket.
                    if mode.startswith("adapter_") and stop.wait(0.2):
                        return
                    timings["cancel_requested"] = time.monotonic()
                    cancellation.set()
                    if mode == "adapter_late_cancel" and not stop.wait(0.5):
                        _docker("unpause", containers[0].container_id)
                        timings["adapter_resumed"] = time.monotonic()
                    return
        except Exception as exc:
            trigger_errors.append(exc)

    trigger = Thread(target=cancel_after_request, daemon=True)
    if "cancel" in mode:
        trigger.start()
    started = time.monotonic()
    try:
        result = CaseOrchestrator(registry, store, launcher, executor).run(
            "faults",
            mode,
            f"episode-{mode}",
            "libero",
            identity.task_id,
            identity.seed,
            "libero-fault-agent-v1",
            timeout_s=180,
            case_timeout_s=30 if "timeout" in mode else 120,
            cancel_event=cancellation,
        )
        ended = time.monotonic()
        assert not trigger_errors
        assert result.status == status, result.as_dict()
        assert result.error.code == code and result.error.details["stage"] == stage
        assert result.score is None and result.native_score_source is None
        assert store.load("faults", mode) == result
        assert store.trace_state("faults", mode).complete
        events = store.trace_events("faults", mode)
        kinds = [
            e["payload"]["message"]["kind"]
            for e in events
            if e["type"] == "roborsi_message"
        ]
        assert kinds[0] == "start_result"  # The native environment really reset.
        assert "step_result" not in kinds and "native_result" not in kinds
        assert (case_dir / "agent" / "act.started").is_file()
        assert len(list((case_dir / "media").rglob("*.bin"))) >= 2
        assert len(processes) == len(containers) == 1
        assert processes[0].daemon.poll() is not None
        assert not _docker(
            "container",
            "ls",
            "--all",
            "--filter",
            f"id={containers[0].container_id}",
            "--format",
            "{{.ID}}",
        )
        config = inspections[0]
        assert config["HostConfig"]["NetworkMode"] != "host"
        assert not config["HostConfig"]["Privileged"]
        assert config["HostConfig"]["PidMode"] != "host"
        assert (
            config["HostConfig"]["NanoCpus"]
            == lock.as_dict()["resources"]["cpus"] * 10**9
        )
        assert (
            config["HostConfig"]["Memory"]
            == lock.as_dict()["resources"]["memory_mb"] * 1024**2
        )
        assert all(
            p["HostIp"] == "127.0.0.1"
            for ports in config["HostConfig"]["PortBindings"].values()
            for p in ports
        )
        mounts = {mount["Destination"]: mount for mount in config["Mounts"]}
        assert set(mounts) == {"/roborsi/assets/libero"}
        assert config["HostConfig"]["ReadonlyRootfs"]
        assert all(str(case_dir) not in mount["Source"] for mount in mounts.values())
        assert not mounts["/roborsi/assets/libero"]["RW"]
        measurements = {
            "case_id": mode,
            "total_s": ended - started,
            "status": result.status,
        }
        if "cancel" in mode:
            measurements["interrupt_s"] = (
                timings["rpc_returned"] - timings["cancel_requested"]
            )
            measurements["cancel_and_cleanup_s"] = ended - timings["cancel_requested"]
            assert 0 <= measurements["interrupt_s"] < 2
            assert measurements["cancel_and_cleanup_s"] < 15
        if "timeout" in mode:
            measurements["timeout_and_cleanup_s"] = ended - started - 30
            assert 0 <= measurements["timeout_and_cleanup_s"] < 15
        if mode == "adapter_late_cancel":
            assert timings["rpc_returned"] < timings["adapter_resumed"]
        (tmp_path / "measurements.json").write_text(json.dumps(measurements, indent=2))
    finally:
        stop.set()
        if trigger.is_alive():
            trigger.join(5)
        for process in processes:
            process.daemon.stop(timeout=1)
        for container in containers:
            subprocess.run(
                ["docker", "rm", "--force", container.container_id],
                capture_output=True,
                timeout=15,
            )
