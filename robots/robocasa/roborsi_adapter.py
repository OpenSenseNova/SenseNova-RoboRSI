# Copyright 2026 The RoboRSI Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Map each RoboRSI action to exactly one RoboCasa toolkit call."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from roborsi.evaluation.media import CaseMediaChannel
from roborsi.evaluation.messages import (
    Action,
    EpisodeStop,
    NativeResult,
    Observation,
    StartResult,
    StepResult,
    Task,
)
from roborsi.evaluation.result import write_json_atomic
from robots.robocasa.benchmark_service import (
    BENCHMARK,
    RLDX_ENVIRONMENT,
    make_task,
    runtime_info,
)

# The toolkit returns camera frames as PNG bytes under these keys, in the order
# the legacy tool result attached them.
_IMAGE_KEYS = (
    ("_image_bytes", "image"),
    ("_image_cam_bytes", "agentview"),
    ("_image_nav_bytes", "navview"),
    ("_image_wrist_bytes", "wrist"),
)
# Native records shared as case artifacts; per-step images and world maps stay
# in the case directory. env_server keeps logging and the toolkit keeps writing
# its state file until close, so each is published as a copy taken at scoring.
_PUBLISHED = ("trajectory.jsonl", "env_server.log", "state/states.json")


def _start_runtime(args: argparse.Namespace, output_dir: Path, events):
    """Spawn this episode's env_server and attach the shared RLDX-1 service."""
    from robots.robocasa.runtime import _init_runtime

    return _init_runtime(args, output_dir, events, None)


def _make_toolkit(runtime_kwargs: dict[str, Any], events, state_dir: Path):
    from robots.robocasa.toolkit import RoboCasaToolkit

    return RoboCasaToolkit(
        runtime_kwargs=runtime_kwargs,
        dashboard_events=events,
        enable_vla=True,
        mode="evaluation",
        state_output_dir=state_dir,
    )


def _stop_runtime(daemons, events) -> None:
    from roborsi.robots.runtime import stop_owned_daemons

    stop_owned_daemons({f"d{i}": d for i, d in enumerate(daemons)}, events)


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str, allow_nan=False))


class RoboCasa365Adapter:
    """Run one Target50 cell; the native verdict is the final state.success."""

    def __init__(
        self,
        directory: Path,
        *,
        vla_endpoint: str | None,
        cuda_device: int | None = None,
        artifacts=None,
        start_runtime: Callable = _start_runtime,
        make_toolkit: Callable = _make_toolkit,
        stop_runtime: Callable = _stop_runtime,
    ) -> None:
        from roborsi.dashboard.events import NullDashboardEventSink

        self.directory = Path(directory)
        self.vla_endpoint = vla_endpoint
        self.cuda_device = cuda_device
        self.artifacts = artifacts
        self._start_runtime = start_runtime
        self._make_toolkit = make_toolkit
        self._stop_runtime = stop_runtime
        self._events = NullDashboardEventSink()
        self.media = CaseMediaChannel(self.directory / "media")
        self.native_dir = self.directory / "native"
        self.native_dir.mkdir(parents=True, exist_ok=True)
        self.daemons: list = []
        self.runtime_kwargs: dict[str, Any] = {}
        self.toolkit = None
        self.task: Task | None = None
        self.episode: str | None = None
        self.steps = 0
        self.ended = False
        self.closed = False
        self.reason: str | None = None
        self.success = False
        self.finish_call: dict | None = None
        self.observation: Observation | None = None
        self.result: NativeResult | None = None

    def runtime_info(self):
        """Expose identical source metadata before and after cleanup."""
        return runtime_info()

    def _record(self, row: dict) -> None:
        with (self.native_dir / "trajectory.jsonl").open("a") as stream:
            stream.write(json.dumps(_jsonable(row), allow_nan=False) + "\n")

    def _observe(self, result: dict) -> Observation:
        data = dict(result)
        media = []
        cameras = []
        for key, camera in _IMAGE_KEYS:
            frame = data.pop(key, None)
            if frame:
                media.append(
                    self.media.write(self.episode, self.steps, frame, "image/png")
                )
                cameras.append(camera)
        data = _jsonable(data)
        data["camera_names"] = cameras
        data["remaining_steps"] = self.task.max_steps - self.steps
        self.observation = Observation(
            benchmark=BENCHMARK,
            episode_id=self.episode,
            step_index=self.steps,
            data=data,
            media=tuple(media),
        )
        return self.observation

    def start(self, task: Task, episode_id: str) -> StartResult:
        """Validate the catalog entry, then build the kitchen and its toolkit."""
        if self.task is not None or self.closed:
            raise ValueError("RoboCasa365 Adapter owns exactly one episode")
        if task != make_task(task.task_id, task.seed):
            raise ValueError("RoboCasa365 task differs from the Target50 catalog")
        if not self.vla_endpoint:
            raise ValueError(
                "Target50 needs the RLDX-1 service; set ROBORSI_ROBOCASA_VLA_ENDPOINT"
            )
        self.task, self.episode = task, episode_id
        os.environ.update(RLDX_ENVIRONMENT)
        args = argparse.Namespace(
            task_name=task.task_id,
            split=task.native["environment_split"],
            seed=task.seed,
            env_endpoint=None,
            vla_endpoint=self.vla_endpoint,
            vla_model_path=None,
            cuda_device=self.cuda_device,
            hi_res=0,
            enable_vla=True,
        )
        try:
            self.daemons, self.runtime_kwargs = self._start_runtime(
                args, self.native_dir, self._events
            )
            self.toolkit = self._make_toolkit(
                self.runtime_kwargs, self._events, self.native_dir / "state"
            )
            initial = self.toolkit.execute_tool("view_env_state", {}).result
            if initial.get("error"):
                raise RuntimeError(f"initial RoboCasa state: {initial['error']}")
            self._record({"step": 0, "phase": "reset", "seed": task.seed})
            return StartResult(
                task=task,
                observation=self._observe(initial),
                capabilities=task.capabilities,
            )
        except BaseException:
            self.close()
            raise

    def _identity(self, message) -> None:
        if self.toolkit is None or self.ended or self.closed:
            raise ValueError("RoboCasa365 episode is not active")
        if (message.benchmark, message.episode_id, message.step_index) != (
            BENCHMARK,
            self.episode,
            self.steps,
        ):
            raise ValueError("RoboCasa365 episode or step mismatch")

    def step(self, action: Action) -> StepResult:
        """Run one RoboCasa tool and return the state it leaves behind."""
        self._identity(action)
        arguments = dict(action.data)
        name = arguments.pop("primitive")
        outcome = self.toolkit.execute_tool(name, arguments)
        self.steps += 1
        result = outcome.result if isinstance(outcome.result, dict) else {}
        terminated = bool(outcome.is_finish)
        if terminated:
            self.finish_call = {"status": arguments.get("status"), **result}
            self.reason = "agent_finish"
        truncated = not terminated and self.steps >= self.task.max_steps
        if truncated:
            self.reason = "episode_limit"
        self.ended = terminated or truncated
        if self.ended:
            self.success = bool(self.toolkit.solved())
        self._record(
            {
                "step": self.steps,
                "phase": "action",
                "primitive": name,
                "arguments": arguments,
                "result": {k: v for k, v in result.items() if not k.endswith("_bytes")},
            }
        )
        if terminated:
            # finish reads no new state; show the agent the state it stopped on.
            result = {
                **self.toolkit.execute_tool("view_env_state", {}).result,
                "finish": {k: v for k, v in result.items() if k != "_finish"},
            }
        return StepResult(
            observation=self._observe(result),
            terminated=terminated,
            truncated=truncated,
            termination_reason=self.reason,
        )

    def stop(self, request: EpisodeStop) -> StepResult:
        """End the cell on a planner timeout or give-up; the state still scores."""
        self._identity(request)
        self.ended = True
        self.reason = "policy_stop"
        self.success = bool(self.toolkit.solved())
        write_json_atomic(
            self.native_dir / "policy_stop.json",
            {"reason": request.reason, "step": self.steps},
        )
        return StepResult(
            observation=self.observation,
            terminated=False,
            truncated=True,
            termination_reason=self.reason,
        )

    def finish(self) -> NativeResult:
        """Score the cell from the simulator's own success flag."""
        if not self.ended:
            raise ValueError("RoboCasa365 episode must end before scoring")
        if self.result is not None:
            return self.result
        success = self.success
        details = {
            "task": self.task.task_id,
            "seed": self.task.seed,
            "split": self.task.native["split"],
            "environment_split": self.task.native["environment_split"],
            "protocol": self.task.native["protocol"],
            "success_source": "state.success",
            "actions": self.steps,
            "rldx": dict(RLDX_ENVIRONMENT),
            "termination_reason": self.reason,
            "finish": _jsonable(self.finish_call),
        }
        path = self.native_dir / "score.json"
        write_json_atomic(path, details)
        if self.artifacts is not None:
            records = self.native_dir / "records"
            records.mkdir(exist_ok=True)
            for name in _PUBLISHED:
                source = self.native_dir / name
                if source.is_file():
                    shutil.copyfile(source, records / source.name)
                    self.artifacts.publish(
                        "native/records/" + source.name,
                        "file",
                        "application/octet-stream",
                    )
        self.result = NativeResult(
            task=self.task,
            episode_id=self.episode,
            success=success,
            score=float(success),
            details=details,
            source="native/score.json",
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        return self.result

    def close(self) -> None:
        """Release the toolkit, this episode's RPC sessions and its env_server.

        The RLDX-1 service is shared by every episode and keeps running; only
        this episode's session on it, with its memory and RTC state, ends here.
        """
        self.closed = True
        toolkit, self.toolkit = self.toolkit, None
        daemons, self.daemons = self.daemons, []
        clients, self.runtime_kwargs = self.runtime_kwargs, {}
        try:
            if toolkit is not None:
                toolkit.close()
        finally:
            try:
                for name in ("vla_client", "env_client"):
                    rpc = getattr(clients.get(name), "_client", None)
                    if rpc is not None:
                        rpc.close()
            finally:
                if daemons:
                    self._stop_runtime(daemons, self._events)
