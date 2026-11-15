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

"""Opt-in real Docker/LIBERO check, including independent native scoring."""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path

import pytest
from PIL import Image

from roborsi.benchmarks.registry import BenchmarkRegistry
from roborsi.evaluation.agent_artifact import AgentArtifactStore
from roborsi.evaluation.docker_runtime import DockerExecutor
from roborsi.evaluation.orchestration import CaseOrchestrator
from roborsi.evaluation.store import ResultStore
from roborsi.utils.config import get_project_env, require_project_env

pytestmark = [
    pytest.mark.skipif(
        get_project_env("ROBORSI_RUN_LIBERO_BENCHMARK") != "1",
        reason="requires a built LIBERO image and downloaded assets",
    ),
    pytest.mark.timeout(1800),
]


def test_native_score_media_and_runtime_cleanup(tmp_path: Path, monkeypatch) -> None:
    registry = BenchmarkRegistry([Path(require_project_env("ROBORSI_LIBERO_REGISTRY"))])
    lock = registry.lock("libero")
    identity = lock.list_tasks()[0]
    task = lock.task_message(identity.task_id, identity.seed)
    repo = Path(__file__).resolve().parents[4]
    artifact = AgentArtifactStore(tmp_path / "agents").create(
        repo / "examples" / "roborsi",
        factory="hold_agent:make_agent",
        components={"control": ["hold_agent.py"]},
    )
    launcher = artifact.launcher()
    docker = DockerExecutor()
    processes, containers = [], []
    launch_agent, launch_adapter = launcher.start, docker.start_adapter

    def track_agent(*args, **kwargs):
        process = launch_agent(*args, **kwargs)
        processes.append(process)
        return process

    def track_adapter(*args, **kwargs):
        container = launch_adapter(*args, **kwargs)
        containers.append(container)
        return container

    monkeypatch.setattr(launcher, "start", track_agent)
    monkeypatch.setattr(docker, "start_adapter", track_adapter)
    store = ResultStore(tmp_path / "results")
    result = CaseOrchestrator(registry, store, launcher, docker).run(
        "libero-native-check",
        "first-task",
        "libero-real-episode",
        "libero",
        identity.task_id,
        identity.seed,
        artifact.version_id,
        timeout_s=180,
        case_timeout_s=900,
    )
    assert result.status in {"success", "task_failure"}, result.as_dict()
    assert store.load("libero-native-check", "first-task") == result
    assert len(processes) == len(containers) == 1
    assert processes[0].daemon.poll() is not None
    assert (
        subprocess.run(
            ["docker", "inspect", containers[0].container_id], capture_output=True
        ).returncode
        != 0
    )
    case_dir = tmp_path / "results" / "libero-native-check" / "first-task"
    pictures = list((case_dir / "media").rglob("*.bin"))
    assert len(pictures) >= 2
    for picture in pictures[:2]:
        with Image.open(picture) as image:
            assert image.size == (256, 256)
            assert image.getbbox() is not None
    asset = lock.as_dict()["assets"][0]
    asset_path = (lock.manifest_path.parent / asset["path"]).resolve()
    # Compare against LIBERO's own check_success(), outside RoboRSI and RLinf.
    reference_name = f"roborsi-native-reference-{uuid.uuid4().hex}"
    try:
        native_process = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--name",
                reference_name,
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--cpus",
                "2",
                "--memory",
                "4096m",
                "--mount",
                f"type=bind,src={asset_path},dst=/roborsi/assets/libero,readonly",
                "--mount",
                f"type=bind,src={Path(__file__).parent},dst=/checks,readonly",
                lock.image_reference,
                "python",
                "/checks/native_reference.py",
                "--suite",
                task.native["suite"],
                "--task",
                task.task_id,
                "--seed",
                str(task.seed),
                "--max-steps",
                str(task.max_steps),
            ],
            check=True,
            timeout=900,
            capture_output=True,
            text=True,
        )
    finally:
        subprocess.run(
            ["docker", "rm", "--force", reference_name],
            capture_output=True,
            timeout=15,
        )
    reference = json.loads(
        next(
            line.removeprefix("ROBORSI_NATIVE_REFERENCE=")
            for line in native_process.stdout.splitlines()
            if line.startswith("ROBORSI_NATIVE_REFERENCE=")
        )
    )
    details = json.loads((case_dir / result.native_score_source).read_text())
    assert reference["success"] is result.success
    assert reference["score"] == result.score
    assert reference["steps"] == details["steps"]
    assert reference["last_reward"] == details["last_reward"]
    trace = store.trace_state("libero-native-check", "first-task")
    assert trace.complete and trace.status == result.status
    events = store.trace_events("libero-native-check", "first-task")
    messages = [
        event["payload"]["message"]
        for event in events
        if event["type"] == "roborsi_message"
    ]
    assert (
        sum(message["kind"] == "action" for message in messages) == reference["steps"]
    )
    assert (
        sum(message["kind"] == "step_result" for message in messages)
        == reference["steps"]
    )
    assert messages[0]["kind"] == "start_result"
    assert messages[-1]["kind"] == "native_result"
    from roborsi.evaluation.trajectory import TrajectoryReader

    reader = TrajectoryReader(case_dir, trace_manifest=result.trace_manifest)
    index = reader.index()
    assert index["summary"]["n_turns"] == reference["steps"]
    assert len(index["media"]) == len(pictures)
    assert all(item["available"] for item in index["media"])
