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

"""Real LIBERO faults leave queryable states even when result files cannot save."""

from __future__ import annotations

import errno
import json
import subprocess
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from roborsi.benchmarks.registry import BenchmarkRegistry
from roborsi.evaluation import store as store_module
from roborsi.evaluation.agent_process import PythonAgentLauncher
from roborsi.evaluation.benchmark_client import BenchmarkClient
from roborsi.evaluation.docker_runtime import DockerExecutor
from roborsi.evaluation.orchestration import CaseOrchestrator
from roborsi.evaluation.store import ResultStore
from roborsi.utils.config import get_project_env, require_project_env

pytestmark = [
    pytest.mark.skipif(
        get_project_env("ROBORSI_RUN_LIBERO_BENCHMARK") != "1",
        reason="requires the LIBERO HTTP image and assets",
    ),
    pytest.mark.timeout(240),
]


@contextmanager
def _response_fault(endpoint, mode, token):
    """Forward real HTTP bytes, changing only the selected native result field."""
    faults = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def forward(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", "0"))) or None
            request = urllib.request.Request(
                endpoint + self.path,
                data=body,
                method=self.command,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                },
            )
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            try:
                response = opener.open(request, timeout=120)
            except urllib.error.HTTPError as exc:
                response = exc
            with response:
                content = response.read()
                status = response.status
                headers = dict(response.headers)
            if (
                self.command == "POST"
                and self.path.endswith("/finish")
                and status == 200
                and mode != "result_disk_full"
            ):
                reply = json.loads(content)
                if mode == "nan_result":
                    reply["payload"]["score"] = float("nan")
                else:
                    reply["episode_id"] = "another-episode"
                content = json.dumps(reply).encode()
                faults.append(mode)
            self.send_response(status)
            for key, value in headers.items():
                if key.lower() not in {
                    "content-length",
                    "connection",
                    "transfer-encoding",
                    "server",
                    "date",
                }:
                    self.send_header(key, value)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        do_GET = do_POST = do_DELETE = forward

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", faults
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


@pytest.mark.parametrize("mode", ["nan_result", "wrong_episode", "result_disk_full"])
def test_real_native_response_and_result_storage_failures(tmp_path, monkeypatch, mode):
    registry = BenchmarkRegistry([require_project_env("ROBORSI_LIBERO_REGISTRY")])
    task = registry.lock("libero").task_message("0", 0)
    assert task.max_steps == 3, "prepare the three-step integration registry"
    status_root = tmp_path / "status"
    store = ResultStore(tmp_path / "results", status_root=status_root)
    launcher = PythonAgentLauncher("examples.roborsi.hold_agent:make_agent")
    executor = DockerExecutor()
    start_adapter, start_agent = executor.start_adapter, launcher.start
    containers, agents, injected = [], [], []

    @contextmanager
    def track_adapter(*args, **kwargs):
        container = start_adapter(*args, **kwargs)
        containers.append(container)
        try:
            with _response_fault(
                container.client.transport.endpoint,
                mode,
                container.client.transport.token,
            ) as (
                endpoint,
                faults,
            ):
                container.client = BenchmarkClient(endpoint)
                yield container
                injected.extend(faults)
        finally:
            container.close()

    def track_agent(*args, **kwargs):
        process = start_agent(*args, **kwargs)
        agents.append(process)
        return process

    monkeypatch.setattr(executor, "start_adapter", track_adapter)
    monkeypatch.setattr(launcher, "start", track_agent)
    original_write = store_module.write_json_atomic

    def write_result(path, value):
        if mode == "result_disk_full" and Path(path).name.startswith(".result-"):
            raise OSError(errno.ENOSPC, "injected result-volume exhaustion")
        return original_write(path, value)

    monkeypatch.setattr(store_module, "write_json_atomic", write_result)
    result = CaseOrchestrator(registry, store, launcher, executor).run(
        "faults",
        mode,
        mode,
        "libero",
        "0",
        0,
        "hold-agent",
        timeout_s=120,
        case_timeout_s=180,
    )
    assert result.status == "infrastructure_error" and result.score is None, (
        result.as_dict()
    )
    assert result.error.details["stage"] == (
        "store.publish" if mode == "result_disk_full" else "adapter.finish"
    )
    if mode != "result_disk_full":
        assert injected
    reader = ResultStore(store.root, status_root=status_root)
    assert reader.load("faults", mode) == result
    state = reader.status("faults", mode)
    assert state.state == "failed"
    assert state.result_published is (mode != "result_disk_full")
    directory = store.root / "faults" / mode
    assert len(list((directory / "media").rglob("*.bin"))) == 8
    assert all(process.daemon.poll() is not None for process in agents)
    assert len(agents) == len(containers) == 1
    assert not subprocess.check_output(
        ["docker", "ps", "-aq", "--filter", f"id={containers[0].container_id}"],
        text=True,
        timeout=10,
    ).strip()
    json.dumps(state.as_dict(), allow_nan=False)
    if mode == "result_disk_full":
        assert not (directory / "result.json").exists()
        assert (status_root / "faults" / f"{mode}.json").is_file()
    else:
        assert reader.trace_state("faults", mode).complete
