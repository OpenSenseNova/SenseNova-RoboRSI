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

"""Real LIBERO HTTP deployment with no Runner files mounted into the service."""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid

import pytest
from PIL import Image

from roborsi.benchmarks.registry import BenchmarkRegistry
from roborsi.evaluation.agent_process import PythonAgentLauncher
from roborsi.evaluation.benchmark_client import BenchmarkClient
from roborsi.evaluation.http_benchmark import (
    HttpBenchmarkExecutor,
    HttpBenchmarkRegistry,
)
from roborsi.evaluation.messages import NativeResult, from_envelope
from roborsi.evaluation.orchestration import CaseOrchestrator
from roborsi.evaluation.protocol import InteractionEnvelope
from roborsi.evaluation.store import ResultStore
from roborsi.utils.config import get_project_env, require_project_env
from roborsi.utils.daemon import pick_free_port
from roborsi.utils.rpc import RpcError, wait_for_ready

pytestmark = [
    pytest.mark.skipif(
        get_project_env("ROBORSI_RUN_LIBERO_BENCHMARK") != "1",
        reason="requires the LIBERO HTTP image and assets",
    ),
    pytest.mark.timeout(600),
]


def test_remote_catalog_runner_complete_test_and_downloads(tmp_path):
    lock = BenchmarkRegistry([require_project_env("ROBORSI_LIBERO_REGISTRY")]).lock(
        "libero"
    )
    asset = lock.as_dict()["assets"][0]
    assets_path = (lock.manifest_path.parent / asset["path"]).resolve()
    name, port = f"roborsi-http-{uuid.uuid4().hex}", pick_free_port()
    token = uuid.uuid4().hex
    environment = {**os.environ, "ROBORSI_BENCHMARK_TOKEN": token}
    command = [
        "docker",
        "run",
        "--detach",
        "--rm",
        "--name",
        name,
        "--read-only",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=1g",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--publish",
        f"127.0.0.1:{port}:8000",
        "--cpus",
        "2",
        "--memory",
        "4096m",
        "--env",
        "ROBORSI_BENCHMARK_TOKEN",
        "--env",
        "ROBORSI_LIBERO_MAX_STEPS=3",
        "--mount",
        f"type=bind,src={assets_path},dst=/roborsi/assets/libero,readonly",
        lock.image_reference,
    ]
    container = None
    try:
        container = subprocess.check_output(
            command, env=environment, text=True, timeout=60
        ).strip()
        endpoint = f"http://127.0.0.1:{port}"
        client = BenchmarkClient(endpoint, token=token)
        wait_for_ready(client.transport, timeout_s=120, poll_interval_s=0.2)
        with pytest.raises(RpcError, match="authentication required"):
            BenchmarkClient(endpoint).tasks(timeout_s=3)
        inspection = json.loads(
            subprocess.check_output(
                ["docker", "inspect", container], text=True, timeout=10
            )
        )[0]
        assert {m["Destination"] for m in inspection["Mounts"]} == {
            "/roborsi/assets/libero"
        }
        assert all(not m["RW"] for m in inspection["Mounts"])
        assert not any(str(tmp_path) in m["Source"] for m in inspection["Mounts"])

        # This registry sees only HTTP. It has no manifest or asset directory.
        registry = HttpBenchmarkRegistry(
            {"libero": endpoint}, token=token, timeout_s=120
        )
        task = registry.lock("libero").task_message("0", 0)
        assert task.max_steps == 3
        store = ResultStore(tmp_path / "runner-results")
        result = CaseOrchestrator(
            registry,
            store,
            PythonAgentLauncher("examples.roborsi.hold_agent:make_agent"),
            HttpBenchmarkExecutor(),
        ).run(
            "remote",
            "task",
            "remote-task",
            "libero",
            "0",
            0,
            "hold-agent",
            timeout_s=120,
            case_timeout_s=300,
        )
        assert result.status == "task_failure" and result.score == 0, result.as_dict()
        assert result.runtime.observed.docker_image_digest is None
        assert {a.kind for a in result.benchmark_artifacts} == {
            "image",
            "native_result",
            "log",
        }
        assert len([a for a in result.benchmark_artifacts if a.kind == "image"]) == 8
        assert store.load("remote", "task") == result
        with pytest.raises(RpcError, match="episode not found"):
            client.status("remote-task")

        client.submit_test(
            "complete-task",
            task,
            [{"vector": [0, 0, 0, 0, 0, 0, -1]}],
            request_id="full-test",
            repeat_last=True,
            timeout_s=120,
        )
        deadline = time.monotonic() + 120
        while True:
            state = client.status("complete-task")
            if state["state"] in {"completed", "failed", "cancelled"}:
                break
            assert time.monotonic() < deadline
            time.sleep(0.05)
        assert state["state"] == "completed", state
        status, wire = client.result("complete-task")
        native = from_envelope(InteractionEnvelope.from_dict(wire))
        assert status == 200 and isinstance(native, NativeResult)
        assert not native.success and native.score == 0
        artifacts = client.artifacts(episode_id="complete-task")
        for artifact in artifacts:
            client.download(artifact, tmp_path / "full-test")
        assert len([a for a in artifacts if a.kind == "image"]) == 8
        with Image.open(next((tmp_path / "full-test/media").rglob("*.bin"))) as image:
            assert image.size == (256, 256)
        client.delete("complete-task")
        assert (
            json.loads((tmp_path / "full-test/native/score.json").read_text())["steps"]
            == 3
        )
    finally:
        subprocess.run(
            ["docker", "rm", "--force", container or name],
            capture_output=True,
            timeout=15,
        )
    assert store.load("remote", "task") == result
    assert not subprocess.check_output(
        ["docker", "ps", "-aq", "--filter", f"name=^{name}$"], text=True
    ).strip()
