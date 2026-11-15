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

"""RoboRSI Adapter for RoboRSI's existing single-environment LIBERO runtime."""

from __future__ import annotations

import hashlib
import io
import math
import platform
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, Protocol

from roborsi.benchmarks.runtime_source import benchmark_source_digest
from roborsi.evaluation.media import CaseMediaChannel
from roborsi.evaluation.messages import (
    Action,
    BenchmarkRuntimeInfo,
    NativeResult,
    Observation,
    StartResult,
    StepResult,
    Task,
)
from roborsi.evaluation.result import write_json_atomic
from roborsi.utils.config import get_project_env, require_project_env

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "vector": {
            "type": "array",
            "description": (
                "One Panda OSC_POSE action at 20 Hz: three normalized translation "
                "components, three normalized axis-angle rotation components, "
                "then gripper control. The native arm controller clips each of "
                "the first six values to [-1, 1] before scaling; finite values "
                "outside that range retain native clipping behavior."
            ),
            "prefixItems": [
                {
                    "type": "number",
                    "description": (
                        f"{axis} translation in the robot base frame; [-1, 1] "
                        "scales to [-0.05, 0.05] metres."
                    ),
                }
                for axis in ("x", "y", "z")
            ]
            + [
                {
                    "type": "number",
                    "description": (
                        f"{axis} component of the rotation vector in the robot "
                        "base frame; [-1, 1] scales to [-0.5, 0.5] radians."
                    ),
                }
                for axis in ("x", "y", "z")
            ]
            + [
                {
                    "type": "number",
                    "description": (
                        "Gripper: negative opens, positive closes, zero keeps "
                        "the current command. Use -1, 0 or 1."
                    ),
                }
            ],
            "items": {"type": "number"},
            "minItems": 7,
            "maxItems": 7,
        }
    },
    "required": ["vector"],
    "additionalProperties": False,
}
_MAX_INLINE_BYTES = 64 * 1024
EVALUATION_RULES = {
    "success_metric": "native_success",
    "score_metric": "native_success",
    "aggregation": "task_weighted_success_rate",
    "error_denominator": "expected",
}


def runtime_info() -> BenchmarkRuntimeInfo | None:
    """Read versions from the installed LIBERO runtime and packaged source."""
    commit = get_project_env("ROBORSI_LIBERO_COMMIT")
    if commit is None:
        return None
    from importlib.metadata import version

    root = Path(__file__).resolve().parents[2]
    source_digest = benchmark_source_digest(root, "libero")
    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        cpu = next(
            (
                line.partition(":")[2].strip()
                for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu,
        )
    return BenchmarkRuntimeInfo(
        benchmark="libero",
        benchmark_version=f"rpent-libero-{version('rpent-libero')}",
        benchmark_commit=commit,
        adapter_version=f"roborsi-source-sha256:{source_digest}",
        evaluation=dict(EVALUATION_RULES),
        platform={
            "system": platform.system(),
            "kernel": platform.release(),
            "machine": platform.machine(),
            "cpu": cpu,
        },
    )


class NativeLiberoEnv(Protocol):
    """The narrow surface reused from :class:`LiberoEnvFacade`."""

    def get_env_meta(self) -> dict[str, Any]: ...

    def get_task_language(self) -> str | None: ...

    def reset(self) -> tuple[dict[str, Any], Any]: ...

    def step(self, action: Any) -> tuple[dict[str, Any], Any, Any, Any, Any]: ...

    def close(self) -> None: ...


def _make_native(
    suite: str,
    task_index: int,
    seed: int,
    max_steps: int,
    *,
    video_dir: Path,
) -> NativeLiberoEnv:
    """Create the existing RLinf-backed environment only inside the Adapter."""
    from robots.libero import env_server

    if get_project_env("LIBERO_ROBOT_BASE", "Panda") != "Panda":
        raise ValueError("RoboRSI LIBERO requires the Panda OSC_POSE action contract")
    raw = env_server.make_env(
        task_index,
        seed,
        suite_name=suite,
        max_episode_steps=max_steps,
        video_dir=video_dir,
    )
    try:
        return env_server.LiberoEnvFacade(
            raw,
            meta={
                "suite": raw.task_suite.name,
                "task": int(raw.task_ids[0]),
                "seed": int(raw.seed),
                "max_episode_steps": int(raw.cfg.max_episode_steps),
                "reset_state_id": int(raw.reset_state_ids[0]),
            },
        )
    except Exception:
        raw.env.close()
        raise


def _json_value(value: Any) -> Any:
    """Convert small native state values without flattening their shape."""
    import numpy as np

    if isinstance(value, np.ndarray):
        if value.nbytes > _MAX_INLINE_BYTES:
            raise ValueError("large LIBERO observation array needs a media channel")
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class LiberoRoboRSIAdapter:
    """Map one LIBERO policy episode to the RoboRSI Adapter contract.

    The native environment owns termination and thus the success score. The
    Adapter owns its environment and case-scoped native/media files.
    """

    def runtime_info(self) -> BenchmarkRuntimeInfo | None:
        return runtime_info()

    def __init__(
        self,
        media_dir: str | Path,
        native_dir: str | Path,
        *,
        env_factory: Callable[[str, int, int, int], NativeLiberoEnv] | None = None,
    ) -> None:
        self._media = CaseMediaChannel(media_dir)
        self._native_dir = Path(native_dir)
        if self._native_dir.is_symlink():
            raise ValueError("native directory must not be a symlink")
        self._native_dir.mkdir(parents=True, exist_ok=True)
        self._env_factory = (
            partial(_make_native, video_dir=self._native_dir / "primitive_videos")
            if env_factory is None
            else env_factory
        )
        self._env: NativeLiberoEnv | None = None
        self._task: Task | None = None
        self._episode_id: str | None = None
        self._steps = 0
        self._ended = False
        self._success = False
        self._last_reward: float | None = None
        self._reason: str | None = None

    def _observation(self, values: dict[str, Any]) -> Observation:
        import numpy as np
        from PIL import Image

        assert self._task is not None and self._episode_id is not None
        data: dict[str, Any] = {}
        media = []
        image_fields: dict[str, int] = {}
        for key, value in values.items():
            array = np.asarray(value) if isinstance(value, (np.ndarray, list)) else None
            if array is not None and array.ndim == 3 and array.dtype == np.uint8:
                if array.shape[-1] not in (3, 4):
                    raise ValueError(f"LIBERO image {key} has unsupported channels")
                image = Image.fromarray(array)
                output = io.BytesIO()
                image.save(output, format="PNG")
                image_fields[key] = len(media)
                media.append(
                    self._media.write(
                        self._episode_id, self._steps, output.getvalue(), "image/png"
                    )
                )
            else:
                data[key] = _json_value(value)
        if image_fields:
            data["media_fields"] = image_fields
        return Observation(
            benchmark=self._task.benchmark,
            episode_id=self._episode_id,
            step_index=self._steps,
            data=data,
            media=tuple(media),
        )

    def start(self, task: Task, episode_id: str) -> StartResult:
        """Start the requested suite/index/seed without importing RLinf in Runner."""
        if self._env is not None:
            raise ValueError("LIBERO Adapter already started")
        if task.benchmark != "libero":
            raise ValueError("LIBERO Adapter requires benchmark=libero")
        if not task.task_id.isdecimal() or str(int(task.task_id)) != task.task_id:
            raise ValueError("LIBERO task_id must be a canonical numeric index")
        if (
            set(task.native) != {"suite"}
            or not isinstance(task.native["suite"], str)
            or not task.native["suite"]
        ):
            raise ValueError("LIBERO Task.native needs only a non-empty suite")
        if task.capabilities.get("action_schema") != ACTION_SCHEMA:
            raise ValueError("LIBERO Task.capabilities needs the native action schema")
        suite = task.native["suite"]
        self._env = self._env_factory(
            suite, int(task.task_id), task.seed, task.max_steps
        )
        self._task = task
        self._episode_id = episode_id
        expected = {
            "suite": suite,
            "task": int(task.task_id),
            "seed": task.seed,
            "max_episode_steps": task.max_steps,
        }
        actual = self._env.get_env_meta()
        if any(actual.get(name) != value for name, value in expected.items()):
            raise ValueError("LIBERO environment identity differs from requested task")
        language = self._env.get_task_language()
        if language != task.instruction:
            raise ValueError("LIBERO native task instruction differs from manifest")
        values, _info = self._env.reset()
        return StartResult(
            task=task,
            observation=self._observation(values),
            capabilities={
                "action_schema": ACTION_SCHEMA,
                "reset_state_id": actual.get("reset_state_id"),
            },
        )

    def step(self, action: Action) -> StepResult:
        """Apply one native 7D action and surface native termination."""
        if self._env is None or self._task is None or self._episode_id is None:
            raise ValueError("LIBERO Adapter has not started")
        if self._ended:
            raise ValueError("LIBERO episode already ended")
        if (
            action.benchmark != self._task.benchmark
            or action.episode_id != self._episode_id
            or action.step_index != self._steps
        ):
            raise ValueError("LIBERO action identity or step differs")
        if set(action.data) != {"vector"}:
            raise ValueError("LIBERO action requires only vector")
        vector = action.data["vector"]
        if (
            not isinstance(vector, list)
            or len(vector) != 7
            or any(
                type(number) not in (int, float) or not math.isfinite(number)
                for number in vector
            )
        ):
            raise ValueError("LIBERO action vector needs seven finite numbers")
        values, reward, terminated, truncated, _info = self._env.step(vector)
        self._steps += 1
        self._success = bool(terminated)
        self._last_reward = float(reward)
        self._ended = (
            self._success or bool(truncated) or self._steps >= self._task.max_steps
        )
        self._reason = (
            "native_success"
            if self._success
            else "native_truncation"
            if bool(truncated)
            else "episode_limit"
            if self._ended
            else None
        )
        return StepResult(
            observation=self._observation(values),
            terminated=self._success,
            truncated=self._ended and not self._success,
            reward=self._last_reward,
            termination_reason=self._reason,
        )

    def finish(self) -> NativeResult:
        """Persist native termination as the only success signal."""
        if not self._ended or self._task is None or self._episode_id is None:
            raise ValueError("LIBERO episode must end before scoring")
        details = {
            "native_terminated": self._success,
            "termination_reason": self._reason,
            "steps": self._steps,
            "last_reward": self._last_reward,
        }
        path = self._native_dir / "score.json"
        write_json_atomic(path, details)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return NativeResult(
            task=self._task,
            episode_id=self._episode_id,
            success=self._success,
            score=float(self._success),
            details=details,
            source="native/score.json",
            source_sha256=digest,
        )

    def close(self) -> None:
        if self._env is not None:
            env, self._env = self._env, None
            env.close()


def make_libero_roborsi_adapter() -> LiberoRoboRSIAdapter:
    """Factory for ``roborsi.evaluation.service_host`` in a LIBERO image."""
    return LiberoRoboRSIAdapter(
        require_project_env("ROBORSI_MEDIA_DIR"),
        require_project_env("ROBORSI_NATIVE_DIR"),
    )
