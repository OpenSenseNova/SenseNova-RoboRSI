# Copyright 2026 The SenseNova-RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""RoboRSI Adapter for LIBERO-PRO, Long++ and Ultra-Long."""

from __future__ import annotations

import hashlib
import io
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from roborsi.evaluation.http_artifacts import ArtifactPublisher
from roborsi.evaluation.media import CaseMediaChannel
from roborsi.evaluation.messages import (
    Action,
    NativeResult,
    Observation,
    StartResult,
    StepResult,
    Task,
)
from roborsi.evaluation.result import write_json_atomic

BENCHMARKS = frozenset({"libero_pro", "libero_long_plus_plus", "libero_ultra_long"})
_MAX_INLINE_BYTES = 64 * 1024
VECTOR_SCHEMA = {
    "type": "array",
    "items": {"type": "number"},
    "minItems": 7,
    "maxItems": 7,
}
XYZ_SCHEMA = {
    "type": "array",
    "items": {"type": "number"},
    "minItems": 3,
    "maxItems": 3,
}
ACTION_SCHEMA = {
    "oneOf": [
        {
            "type": "object",
            "description": (
                "One raw LIBERO OSC action [dx, dy, dz, droll, dpitch, dyaw, "
                "gripper]. Prefer a named direct-control primitive when it fits."
            ),
            "properties": {"vector": VECTOR_SCHEMA},
            "required": ["vector"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "description": (
                "A raw LIBERO OSC action chunk. Prefer a named direct-control "
                "primitive for reaching, grasping, releasing, or wrist rotation."
            ),
            "properties": {
                "vectors": {
                    "type": "array",
                    "items": VECTOR_SCHEMA,
                    "minItems": 1,
                }
            },
            "required": ["vectors"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "description": (
                "Legacy move_to: servo the end effector to an absolute world xyz. "
                "It preserves orientation unless target_yaw is supplied and does "
                "not itself articulate a grasped knob."
            ),
            "properties": {
                "primitive": {"const": "move_to"},
                "xyz": XYZ_SCHEMA,
                "max_steps": {"type": "integer", "minimum": 1, "maximum": 1000},
                "gripper": {"type": "number", "minimum": -1, "maximum": 1},
                "step_clip": {"type": "number", "exclusiveMinimum": 0},
                "tol": {"type": "number", "exclusiveMinimum": 0},
                "action_scale": {"type": "number", "exclusiveMinimum": 0},
                "target_yaw": {"type": "number"},
                "yaw_step_clip": {"type": "number", "exclusiveMinimum": 0},
            },
            "required": ["primitive", "xyz"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "description": (
                "Legacy move_pose: servo absolute world xyz and optional absolute "
                "pitch/yaw together. Use it to align an approach pose, not as a "
                "substitute for rotate_wrist after grasping an articulated control."
            ),
            "properties": {
                "primitive": {"const": "move_pose"},
                "xyz": XYZ_SCHEMA,
                "target_pitch": {"type": "number"},
                "target_yaw": {"type": "number"},
                "gripper": {"type": "number", "minimum": -1, "maximum": 1},
                "step_clip": {"type": "number", "exclusiveMinimum": 0},
                "pitch_step": {"type": "number", "exclusiveMinimum": 0},
                "yaw_step": {"type": "number", "exclusiveMinimum": 0},
                "tol": {"type": "number", "exclusiveMinimum": 0},
                "ori_tol": {"type": "number", "exclusiveMinimum": 0},
                "action_scale": {"type": "number", "exclusiveMinimum": 0},
                "max_steps": {"type": "integer", "minimum": 1, "maximum": 1000},
            },
            "required": ["primitive", "xyz"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "description": (
                "Legacy in-place orientation primitive. rotate_wrist changes yaw "
                "about the current end-effector position and is the direct primitive "
                "for twisting a grasped stove control; rotate_pitch changes pitch. "
                "Provide exactly one of absolute target or relative delta, in radians."
            ),
            "properties": {
                "primitive": {"enum": ["rotate_wrist", "rotate_pitch"]},
                "target": {"type": "number"},
                "delta": {"type": "number"},
                "gripper": {"type": "number", "minimum": -1, "maximum": 1},
                "max_steps": {"type": "integer", "minimum": 1, "maximum": 1000},
                "tol": {"type": "number", "exclusiveMinimum": 0},
                "step_clip": {"type": "number", "exclusiveMinimum": 0},
            },
            "required": ["primitive"],
            "oneOf": [
                {"required": ["target"], "not": {"required": ["delta"]}},
                {"required": ["delta"], "not": {"required": ["target"]}},
            ],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "description": "Legacy release: hold position while opening the gripper.",
            "properties": {
                "primitive": {"const": "release"},
                "max_steps": {"type": "integer", "minimum": 1, "maximum": 1000},
            },
            "required": ["primitive"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "description": (
                "Legacy set_gripper: hold the current pose while commanding the "
                "gripper; positive closes and negative opens."
            ),
            "properties": {
                "primitive": {"const": "set_gripper"},
                "gripper": {"type": "number", "minimum": -1, "maximum": 1},
                "steps": {"type": "integer", "minimum": 1, "maximum": 1000},
            },
            "required": ["primitive"],
            "additionalProperties": False,
        },
    ]
}


def _json_value(value: Any) -> Any:
    """Convert bounded native state values without flattening their shape."""
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


class NativeThreeBenchmarkEnv(Protocol):
    """Narrow native surface owned exclusively by the Benchmark service."""

    def identity(self) -> dict[str, Any]: ...

    def instruction(self) -> str: ...

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]: ...

    def step(
        self, action: list[float]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]: ...

    def close(self) -> None: ...


def _vectors(data: dict[str, Any]) -> list[list[float]]:
    if set(data) == {"vector"}:
        values = [data["vector"]]
    elif set(data) == {"vectors"}:
        values = data["vectors"]
        if not isinstance(values, list) or not values:
            raise ValueError("LIBERO action chunk must be a nonempty list")
    else:
        raise ValueError("LIBERO action needs exactly vector or vectors")
    if any(
        not isinstance(vector, list)
        or len(vector) != 7
        or any(
            type(number) not in (int, float) or not math.isfinite(number)
            for number in vector
        )
        for vector in values
    ):
        raise ValueError("LIBERO actions need seven finite numbers per vector")
    return values


def _finite(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"LIBERO {name} must be a finite number")
    return float(value)


def _positive(value: Any, name: str) -> float:
    number = _finite(value, name)
    if number <= 0:
        raise ValueError(f"LIBERO {name} must be positive")
    return number


def _count(value: Any, name: str) -> int:
    if type(value) is not int or not 1 <= value <= 1000:
        raise ValueError(f"LIBERO {name} must be an integer within [1, 1000]")
    return value


class ThreeBenchmarkAdapter:
    """Enforce official 1x scoring while optionally continuing failed cases."""

    def __init__(
        self,
        directory: str | Path,
        artifacts: ArtifactPublisher | None = None,
        *,
        env_factory: Callable[[Task], NativeThreeBenchmarkEnv] | None = None,
    ) -> None:
        self._directory = Path(directory)
        self._media = CaseMediaChannel(self._directory / "media")
        self._native_dir = self._directory / "native"
        self._native_dir.mkdir(parents=True, exist_ok=True)
        self._artifacts = artifacts
        self._env_factory = env_factory
        self._env: NativeThreeBenchmarkEnv | None = None
        self._task: Task | None = None
        self._episode_id: str | None = None
        self._decision_steps = 0
        self._env_steps = 0
        self._ended = False
        self._primary_success = False
        self._cumulative_success = False
        self._score_at_1x: float | None = None
        self._snapshot: dict[str, Any] | None = None
        self._last_reward: float | None = None
        self._reason: str | None = None
        self._writer = None
        self._latest: dict[str, Any] | None = None

    def _observation(self, values: dict[str, Any]) -> Observation:
        import numpy as np
        from PIL import Image

        assert self._task is not None and self._episode_id is not None
        data: dict[str, Any] = {"native_env_steps": self._env_steps}
        media = []
        image_fields: dict[str, int] = {}
        for key, value in values.items():
            array = np.asarray(value) if isinstance(value, (np.ndarray, list)) else None
            if array is not None and array.ndim == 3 and array.dtype == np.uint8:
                if array.shape[-1] not in (3, 4):
                    raise ValueError(f"LIBERO image {key} has unsupported channels")
                output = io.BytesIO()
                Image.fromarray(array).save(output, format="PNG")
                image_fields[key] = len(media)
                media.append(
                    self._media.write(
                        self._episode_id,
                        self._decision_steps,
                        output.getvalue(),
                        "image/png",
                    )
                )
            else:
                data[key] = _json_value(value)
        if image_fields:
            data["media_fields"] = image_fields
        return Observation(
            benchmark=self._task.benchmark,
            episode_id=self._episode_id,
            step_index=self._decision_steps,
            data=data,
            media=tuple(media),
        )

    def _video_frame(self, values: dict[str, Any]) -> None:
        import numpy as np

        frame = next(
            (
                np.asarray(values[name])
                for name in ("main_images", "agentview_image")
                if name in values
            ),
            None,
        )
        if frame is None or frame.ndim != 3 or frame.shape[-1] not in (3, 4):
            return
        if self._writer is None:
            import imageio.v2 as imageio

            self._writer = imageio.get_writer(
                self._native_dir / "episode.mp4", fps=20, macro_block_size=1
            )
        self._writer.append_data(frame[..., :3].astype("uint8", copy=False))

    def start(self, task: Task, episode_id: str) -> StartResult:
        """Start one frozen native case after validating its declared protocol."""
        if self._env is not None:
            raise ValueError("LIBERO three-benchmark Adapter already started")
        if task.benchmark not in BENCHMARKS:
            raise ValueError("unsupported LIBERO three-benchmark identity")
        required = {
            "suite",
            "task_index",
            "reset_protocol",
            "official_1x_steps",
            "max_env_steps",
            "plus9_on_failure",
            "score_protocol",
        }
        optional = {
            "aggregation_task",
            "init_state_id",
            "trial_id",
            "skill_count",
            "observation_mode",
            "observation_protocol",
        }
        if not required <= set(task.native) or set(task.native) - required - optional:
            raise ValueError("LIBERO task native protocol fields differ")
        one_x = task.native["official_1x_steps"]
        maximum = task.native["max_env_steps"]
        plus9 = task.native["plus9_on_failure"]
        if (
            type(one_x) is not int
            or one_x < 1
            or type(maximum) is not int
            or type(plus9) is not bool
            or maximum != one_x * (10 if plus9 else 1)
            or task.max_steps != maximum
        ):
            raise ValueError("LIBERO task has an inconsistent 1x/10x budget")
        if task.capabilities.get("action_schema") != ACTION_SCHEMA:
            raise ValueError("LIBERO task needs the chunk-capable action schema")
        factory = self._env_factory
        if factory is None:
            from robots.libero.three_benchmark_env import make_native_env

            factory = make_native_env
        env = factory(task)
        if env.identity() != task.native:
            env.close()
            raise ValueError("LIBERO native environment identity differs from task")
        if env.instruction() != task.instruction:
            env.close()
            raise ValueError("LIBERO native instruction differs from task")
        self._env, self._task, self._episode_id = env, task, episode_id
        values, _ = env.reset()
        self._latest = values
        self._video_frame(values)
        return StartResult(
            task=task,
            observation=self._observation(values),
            capabilities={
                "action_schema": ACTION_SCHEMA,
                "official_1x_steps": one_x,
                "max_env_steps": maximum,
            },
        )

    def _state(self) -> Any:
        import numpy as np

        if self._latest is None or "states" not in self._latest:
            raise ValueError(
                "LIBERO feedback primitive requires the robot state channel"
            )
        state = np.asarray(self._latest["states"], dtype=np.float64)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("LIBERO robot state must be a finite eight-vector")
        return state

    def _orientation(self) -> tuple[float, float]:
        """Return the legacy controller's world pitch and yaw from axis-angle."""
        import numpy as np

        vector = self._state()[3:6]
        angle = float(np.linalg.norm(vector))
        if angle < 1e-12:
            rotation = np.eye(3)
        else:
            x, y, z = vector / angle
            skew = np.asarray(((0, -z, y), (z, 0, -x), (-y, x, 0)))
            rotation = (
                np.eye(3)
                + math.sin(angle) * skew
                + (1.0 - math.cos(angle)) * (skew @ skew)
            )
        pitch = math.atan2(float(rotation[1, 2]), -float(rotation[2, 2]))
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
        return pitch, yaw

    @staticmethod
    def _wrapped_error(target: float, current: float) -> float:
        return (target - current + math.pi) % (2 * math.pi) - math.pi

    def _primitive_vectors(self, data: dict[str, Any]):
        """Compile one legacy direct-control primitive into live 7-DoF actions."""
        import numpy as np

        primitive = data.get("primitive")
        if primitive == "release":
            for _ in range(_count(data.get("max_steps", 20), "max_steps")):
                yield [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
            return
        if primitive == "set_gripper":
            gripper = _finite(data.get("gripper", -1.0), "gripper")
            if not -1.0 <= gripper <= 1.0:
                raise ValueError("LIBERO gripper must be within [-1, 1]")
            for _ in range(_count(data.get("steps", 5), "steps")):
                yield [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper]
            return
        if primitive in {"rotate_wrist", "rotate_pitch"}:
            supplied = {name for name in ("target", "delta") if name in data}
            if len(supplied) != 1:
                raise ValueError("LIBERO rotation needs exactly target or delta")
            orientation_index = 1 if primitive == "rotate_wrist" else 0
            action_index = 5 if primitive == "rotate_wrist" else 3
            current = self._orientation()[orientation_index]
            target = (
                _finite(data["target"], "target")
                if "target" in data
                else current + _finite(data["delta"], "delta")
            )
            gripper = _finite(data.get("gripper", 1.0), "gripper")
            if not -1.0 <= gripper <= 1.0:
                raise ValueError("LIBERO gripper must be within [-1, 1]")
            tolerance = _positive(data.get("tol", 0.02), "tol")
            step_clip = _positive(data.get("step_clip", 0.10), "step_clip")
            for _ in range(_count(data.get("max_steps", 40), "max_steps")):
                error = self._wrapped_error(
                    target, self._orientation()[orientation_index]
                )
                if abs(error) < tolerance:
                    break
                vector = np.zeros(7, dtype=np.float64)
                vector[action_index] = np.clip(error, -step_clip, step_clip) / 0.10
                vector[action_index] = np.clip(vector[action_index], -1.0, 1.0)
                vector[6] = gripper
                yield vector.tolist()
            return
        if primitive not in {"move_to", "move_pose"}:
            raise ValueError(f"unknown LIBERO primitive: {primitive}")
        xyz = np.asarray(data.get("xyz"), dtype=np.float64)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError("LIBERO primitive xyz must be a finite three-vector")
        gripper = _finite(data.get("gripper", -1.0), "gripper")
        if not -1.0 <= gripper <= 1.0:
            raise ValueError("LIBERO gripper must be within [-1, 1]")
        step_clip = _positive(data.get("step_clip", 0.025), "step_clip")
        tolerance = _positive(data.get("tol", 0.012), "tol")
        action_scale = _positive(data.get("action_scale", 0.05), "action_scale")
        pitch_step = _positive(data.get("pitch_step", 0.08), "pitch_step")
        yaw_step = _positive(
            data.get("yaw_step", data.get("yaw_step_clip", 0.10)), "yaw_step"
        )
        orientation_tolerance = _positive(data.get("ori_tol", 0.05), "ori_tol")
        target_pitch = (
            _finite(data["target_pitch"], "target_pitch")
            if "target_pitch" in data
            else None
        )
        target_yaw = (
            _finite(data["target_yaw"], "target_yaw") if "target_yaw" in data else None
        )
        default_steps = 80 if primitive == "move_to" else 150
        for _ in range(_count(data.get("max_steps", default_steps), "max_steps")):
            state = self._state()
            difference = xyz - state[:3]
            distance = float(np.linalg.norm(difference))
            pitch, yaw = self._orientation()
            pitch_error = (
                0.0
                if target_pitch is None
                else self._wrapped_error(target_pitch, pitch)
            )
            yaw_error = (
                0.0 if target_yaw is None else self._wrapped_error(target_yaw, yaw)
            )
            if (
                distance < tolerance
                and abs(pitch_error) < orientation_tolerance
                and abs(yaw_error) < orientation_tolerance
            ):
                break
            vector = np.zeros(7, dtype=np.float64)
            position = np.clip(difference, -step_clip, step_clip) / action_scale
            vector[:3] = np.clip(position, -1.0, 1.0)
            if primitive == "move_pose" and target_pitch is not None:
                vector[3] = np.clip(pitch_error, -pitch_step, pitch_step) / 0.10
            if target_yaw is not None:
                vector[5] = np.clip(yaw_error, -yaw_step, yaw_step) / 0.10
            vector[3:6] = np.clip(vector[3:6], -1.0, 1.0)
            vector[6] = gripper
            yield vector.tolist()

    def _native_progress(
        self, terminated: bool, info: dict[str, Any]
    ) -> tuple[bool, float]:
        assert self._task is not None
        protocol = self._task.native["score_protocol"]
        if protocol == "final_goal_success_at_1x":
            success = bool(info.get("final_goal_success", terminated))
            return success, float(success)
        if protocol != "ordered_prefix_at_1x":
            raise ValueError("unknown LIBERO score protocol")
        success = info.get("ordered_success")
        progress = info.get("prefix_progress")
        if type(success) is not bool or type(progress) not in (int, float):
            raise ValueError("ordered LIBERO step needs success and prefix progress")
        numeric = float(progress)
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise ValueError("ordered prefix progress must be within [0, 1]")
        return success, numeric

    def _public_native_info(
        self, terminated: bool, info: dict[str, Any]
    ) -> dict[str, Any]:
        """Expose only the finite, benchmark-owned score contract on RoboRSI."""
        assert self._task is not None
        if self._task.native["score_protocol"] == "final_goal_success_at_1x":
            return {
                "final_goal_success": bool(info.get("final_goal_success", terminated))
            }
        success, progress = self._native_progress(terminated, info)
        completed = info.get(
            "skills_completed",
            round(progress * self._task.native["skill_count"]),
        )
        total = info.get("total_skills", self._task.native["skill_count"])
        if type(completed) is not int or type(total) is not int:
            raise ValueError("ordered LIBERO skill counts must be integers")
        return {
            "ordered_success": success,
            "prefix_progress": progress,
            "skills_completed": completed,
            "total_skills": total,
            "final_goal_success": bool(info.get("final_goal_success", False)),
            **(
                {"observation_protocol": info["observation_protocol"]}
                if "observation_protocol" in info
                else {}
            ),
        }

    def step(self, action: Action) -> StepResult:
        """Execute a chunk, stopping at native success or a frozen budget boundary."""
        if self._env is None or self._task is None or self._episode_id is None:
            raise ValueError("LIBERO three-benchmark Adapter has not started")
        if self._ended:
            raise ValueError("LIBERO episode already ended")
        if (
            action.benchmark != self._task.benchmark
            or action.episode_id != self._episode_id
            or action.step_index != self._decision_steps
        ):
            raise ValueError("LIBERO action identity or decision step differs")
        values: dict[str, Any] | None = None
        native_info: dict[str, Any] = {}
        vectors = (
            self._primitive_vectors(action.data)
            if "primitive" in action.data
            else iter(_vectors(action.data))
        )
        for vector in vectors:
            if self._env_steps >= self._task.native["max_env_steps"]:
                break
            values, reward, terminated, native_truncated, info = self._env.step(vector)
            self._latest = values
            self._env_steps += 1
            self._last_reward = float(reward)
            if not math.isfinite(self._last_reward):
                raise ValueError("LIBERO returned a non-finite native reward")
            self._video_frame(values)
            raw_info = dict(info or {})
            success, progress = self._native_progress(bool(terminated), raw_info)
            native_info = self._public_native_info(bool(terminated), raw_info)
            one_x = self._task.native["official_1x_steps"]
            if self._env_steps <= one_x and success:
                self._primary_success = self._cumulative_success = True
                self._score_at_1x = progress
                self._snapshot = {
                    "env_steps": self._env_steps,
                    "success": True,
                    "score": progress,
                }
                self._ended, self._reason = True, "native_success_at_1x"
            elif self._env_steps == one_x:
                self._score_at_1x = progress
                self._snapshot = {
                    "env_steps": self._env_steps,
                    "success": False,
                    "score": progress,
                }
                if not self._task.native["plus9_on_failure"]:
                    self._ended, self._reason = True, "official_1x_limit"
            elif self._env_steps > one_x and success:
                self._cumulative_success = True
                self._ended, self._reason = True, "recovered_after_1x"
            if bool(native_truncated) and not self._ended:
                self._ended, self._reason = True, "native_truncation"
            if (
                self._env_steps >= self._task.native["max_env_steps"]
                and not self._ended
            ):
                self._ended, self._reason = True, "cumulative_limit"
            if self._ended:
                break
        if values is None:
            if "primitive" not in action.data or self._latest is None:
                raise ValueError("LIBERO action chunk executed no native action")
            values = self._latest
        self._decision_steps += 1
        return StepResult(
            observation=self._observation(values),
            terminated=self._primary_success,
            truncated=self._ended and not self._primary_success,
            reward=self._last_reward,
            termination_reason=self._reason,
            native={
                **native_info,
                "action_kind": action.data.get("primitive", "vector"),
                "native_env_steps": self._env_steps,
                "official_1x_snapshot": self._snapshot,
            },
        )

    def finish(self) -> NativeResult:
        """Publish the immutable official 1x score and cumulative diagnostics."""
        if not self._ended or self._task is None or self._episode_id is None:
            raise ValueError("LIBERO episode must end before scoring")
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._snapshot is None or self._score_at_1x is None:
            raise ValueError("LIBERO episode ended without an official 1x snapshot")
        details = {
            "score_protocol": self._task.native["score_protocol"],
            "official_1x_snapshot": self._snapshot,
            "official_1x_success": self._primary_success,
            "official_1x_score": self._score_at_1x,
            "cumulative_success": self._cumulative_success,
            "completed_env_steps": self._env_steps,
            "decision_steps": self._decision_steps,
            "termination_reason": self._reason,
            "last_reward": self._last_reward,
        }
        path = self._native_dir / "score.json"
        write_json_atomic(path, details)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        video = self._native_dir / "episode.mp4"
        if video.is_file() and self._artifacts is not None:
            self._artifacts.publish("native/episode.mp4", "file", "video/mp4")
        return NativeResult(
            task=self._task,
            episode_id=self._episode_id,
            success=self._primary_success,
            score=self._score_at_1x,
            details=details,
            source="native/score.json",
            source_sha256=digest,
        )

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._env is not None:
            env, self._env = self._env, None
            env.close()
