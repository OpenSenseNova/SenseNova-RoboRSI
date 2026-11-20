# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Isolate native SIMPLE from the HTTP service and external Agent environment."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

from roborsi.evaluation.http_artifacts import ArtifactPublisher
from roborsi.evaluation.messages import (
    Action,
    NativeResult,
    StartResult,
    StepResult,
    Task,
)
from roborsi.utils.daemon import ProcessDaemon, pick_free_port
from roborsi.utils.rpc import make_rpc_client, wait_for_ready
from robots.simple.catalog import Catalog
from robots.simple.runtime_setup import prepare_kit_cleanup


class SimpleAdapter:
    """Own exactly one lazy native worker and publish its checked output files."""

    def __init__(
        self,
        directory: Path,
        artifacts: ArtifactPublisher,
        *,
        factory: str = "robots.simple.native:make_runtime",
        timeout_s: float = 900,
        catalog: Catalog | None = None,
    ) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("SIMPLE timeout must be finite and positive")
        self.directory, self.artifacts = directory, artifacts
        self.catalog = catalog
        (directory / "native").mkdir(parents=True, exist_ok=True)
        self.timeout, self.closed = timeout_s, False
        self._cleanup_complete = False
        port = pick_free_port()
        root = Path(__file__).resolve().parents[2]
        native_env = {
            name: value
            for name, value in os.environ.items()
            if name.startswith(
                (
                    "ROBORSI_SIMPLE_",
                    "SIMPLE_ISAAC_",
                    "OMNI_",
                    "CUDA_",
                    "NVIDIA_",
                    "VK_",
                    "XDG_",
                    "__GL_",
                )
            )
            or name
            in {
                "PATH",
                "HOME",
                "LANG",
                "LC_ALL",
                "LD_LIBRARY_PATH",
                "TMPDIR",
                "MUJOCO_GL",
                "MUJOCO_EGL_DEVICE_ID",
                "PYNPUT_BACKEND",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "PYTHONPATH",
            }
        }
        native_env["PYTHONPATH"] = (
            str(root) + os.pathsep + native_env.get("PYTHONPATH", "")
        )
        self.daemon = ProcessDaemon(
            "simple-native",
            [
                os.environ.get("ROBORSI_SIMPLE_PYTHON", sys.executable),
                "-m",
                "robots.simple.worker",
                "--port",
                str(port),
                "--directory",
                str(directory),
                "--factory",
                factory,
            ],
            prepend_rlinf_path=False,
            cwd=str(directory),
            log_path=str(directory / "native/worker.log"),
            env_overrides=native_env,
            inherit_environment=False,
        )
        self.rpc = make_rpc_client(f"http://127.0.0.1:{port}")
        try:
            self.daemon.start()
            wait_for_ready(self.rpc, timeout_s=30, daemon=self.daemon)
        except BaseException:
            self.close()
            raise

    def _call(self, operation: str, payload: dict) -> dict:
        if self.closed:
            raise RuntimeError("SIMPLE adapter is closed")
        return self.rpc.call("request", (operation, payload), timeout_s=self.timeout)

    def _publish(self, result: StartResult | StepResult) -> None:
        for ref in result.observation.media:
            self.artifacts.publish(
                ref.ref,
                "image",
                ref.mime_type,
                step_index=result.observation.step_index,
                sha256=ref.sha256,
                size_bytes=ref.size_bytes,
            )

    def start(self, task: Task, episode_id: str) -> StartResult:
        if self.catalog is not None:
            self.catalog.reset(task)
        result = StartResult.from_dict(
            self._call("start", {"task": task.as_dict(), "episode_id": episode_id})
        )
        self._publish(result)
        return result

    def step(self, action: Action) -> StepResult:
        result = StepResult.from_dict(self._call("step", action.as_dict()))
        self._publish(result)
        return result

    def finish(self) -> NativeResult:
        result = NativeResult.from_dict(self._call("finish", {}))
        self.artifacts.publish(
            result.source,
            "native_result",
            "application/json",
            sha256=result.source_sha256,
        )
        self.artifacts.publish("native/actions.jsonl", "log", "application/x-ndjson")
        reset = self.directory / "native/reset_verification.json"
        if reset.is_file():
            self.artifacts.publish(
                "native/reset_verification.json", "file", "application/json"
            )
        return result

    def close(self) -> None:
        """Reject new work immediately and retry any incomplete resource cleanup."""
        if self._cleanup_complete:
            return
        self.closed = True
        try:
            if self.daemon.poll() is None:
                try:
                    self.rpc.call("request", ("close", {}), timeout_s=5)
                    self.rpc.call("shutdown", timeout_s=1)
                except Exception:
                    pass
        finally:
            try:
                self.daemon.stop(timeout=5)
                if self.daemon.poll() is not None:
                    prepare_kit_cleanup(self.directory)
            finally:
                self.rpc.close()
        self._cleanup_complete = True
