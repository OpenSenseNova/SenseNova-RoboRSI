# Copyright 2026 The SenseNova-RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Native environments for LIBERO-PRO, Long++ and Ultra-Long.

All heavyweight simulator imports are intentionally local to the factory.  This
keeps manifest generation and unit tests usable without MuJoCo installed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from roborsi.evaluation.messages import Task
from roborsi.utils.config import get_project_env
from robots.libero.lilo_assistance import ASSISTANCE_CONTRACT, task_context
from robots.libero.lilo_task import OrderedSkillTracker, build_task_contract


def _axis_angle(quaternion: Any) -> Any:
    import numpy as np

    quat = np.asarray(quaternion, dtype=np.float64)
    if quat.shape != (4,):
        raise ValueError(
            f"end-effector quaternion must have shape (4,), got {quat.shape}"
        )
    scalar = float(np.clip(quat[3], -1.0, 1.0))
    denominator = float(np.sqrt(max(0.0, 1.0 - scalar * scalar)))
    if denominator < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * (2.0 * np.arccos(scalar) / denominator)).astype(np.float32)


def _policy_observation(
    raw: dict[str, Any], language: str, *, instance_pose_feedback: dict | None = None
) -> dict[str, Any]:
    """Apply the OpenGL-to-policy image and eight-dimensional state contract."""
    import numpy as np

    main = np.asarray(raw["agentview_image"])
    wrist = np.asarray(raw["robot0_eye_in_hand_image"])
    if main.ndim != 3 or wrist.ndim != 3 or main.shape[-1] != 3 or wrist.shape[-1] != 3:
        raise ValueError("LIBERO must return two RGB camera observations")
    state = np.concatenate(
        (
            np.asarray(raw["robot0_eef_pos"], dtype=np.float32),
            _axis_angle(raw["robot0_eef_quat"]),
            np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32),
        )
    )
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("LIBERO robot state must be a finite eight-vector")
    observation = {
        "main_images": np.ascontiguousarray(main[::-1, ::-1], dtype=np.uint8),
        "wrist_images": np.ascontiguousarray(wrist[::-1, ::-1], dtype=np.uint8),
        "states": state,
        # Preserve the named state contract returned by the legacy v13 direct
        # tools.  The compact eight-vector remains available for policy clients,
        # while the Agent no longer has to infer its field layout or quaternion
        # convention from an undocumented array.
        "robot_state_feedback": {
            "contract": "libero-robot-state-feedback-v1",
            "end_effector_position": np.asarray(
                raw["robot0_eef_pos"], dtype=np.float64
            ).tolist(),
            "end_effector_quaternion_xyzw": np.asarray(
                raw["robot0_eef_quat"], dtype=np.float64
            ).tolist(),
            "gripper_qpos": np.asarray(
                raw["robot0_gripper_qpos"], dtype=np.float64
            ).tolist(),
        },
        "task_descriptions": language,
    }
    if instance_pose_feedback is not None:
        observation["instance_pose_feedback"] = instance_pose_feedback
    return observation


class _ProEnvironment:
    def __init__(self, task: Task) -> None:
        from robots.libero.env_server import LiberoEnvFacade, make_env

        native = task.native
        if native["reset_protocol"] != "fixed_init_state":
            raise ValueError("LIBERO-PRO requires a fixed initial state")
        init_state = native.get("init_state_id")
        if type(init_state) is not int:
            raise ValueError("LIBERO-PRO requires an integer init_state_id")
        raw = make_env(
            native["task_index"],
            task.seed,
            suite_name=native["suite"],
            max_episode_steps=native["max_env_steps"],
            init_state_id=init_state,
        )
        self._facade = LiberoEnvFacade(
            raw,
            meta={
                "suite": native["suite"],
                "task": native["task_index"],
                "seed": task.seed,
                "init_state_id": init_state,
                "max_episode_steps": native["max_env_steps"],
            },
        )
        self._task = task

    def identity(self) -> dict[str, Any]:
        return dict(self._task.native)

    def instruction(self) -> str:
        instruction = self._facade.get_task_language()
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("LIBERO-PRO native task has no language instruction")
        return instruction

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        values, info = self._facade.reset()
        return values, dict(info or {})

    def step(
        self, action: list[float]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        values, reward, terminated, truncated, info = self._facade.step(action)
        details = dict(info or {})
        details["final_goal_success"] = bool(terminated)
        return values, float(reward), bool(terminated), bool(truncated), details

    def close(self) -> None:
        self._facade.close()


class _OrderedEnvironment:
    def __init__(self, task: Task) -> None:
        import lilo_vla
        import lilo_vla.benchmark  # noqa: F401
        import robosuite.macros as macros
        from libero.libero.benchmark import get_benchmark
        from lilo_vla.core import skill_checker

        native = task.native
        if native.get("observation_protocol") != ASSISTANCE_CONTRACT:
            raise ValueError("ordered LIBERO requires lilo-instance-pose-feedback-v1")
        macros.IMAGE_CONVENTION = "opengl"
        suite = get_benchmark(native["suite"])()
        index = native["task_index"]
        if not 0 <= index < suite.get_num_tasks():
            raise ValueError(f"ordered LIBERO task index is out of range: {index}")
        descriptor = suite.get_task(index)
        lilo_root = Path(
            get_project_env("ROBORSI_LILO_ROOT", Path(lilo_vla.__file__).parents[1])
        )
        config = json.loads(
            (lilo_root / "configs/tasks_and_skills.json").read_text(encoding="utf-8")
        )
        bddl = (
            lilo_root
            / "configs/bddl"
            / descriptor.problem_folder
            / descriptor.bddl_file
        )
        contract = build_task_contract(config, native["suite"], descriptor, bddl)
        if contract["instruction"] != task.instruction:
            raise ValueError(
                "ordered LIBERO instruction differs from its score contract"
            )
        env, observation = suite.make_env(index, camera_depths=True)
        try:
            if native["reset_protocol"] == "fixed_init_state":
                observation = self._restore_fixed_state(
                    env, suite, index, native["init_state_id"], task.seed
                )
            elif native["reset_protocol"] != "official_bare_random_reset":
                raise ValueError("unknown ordered LIBERO reset protocol")
            tracker = OrderedSkillTracker(env, contract["skills"], skill_checker)
        except BaseException:
            env.close()
            raise
        self._task = task
        self._env = env
        self._tracker = tracker
        self._skill_mappings = config["skill_mappings"]
        self._initial = observation
        self._raw: dict[str, Any] | None = None
        self._started = False
        self._goal_success = False

    @staticmethod
    def _restore_fixed_state(
        env: Any, suite: Any, task_index: int, state_index: int, seed: int
    ) -> dict[str, Any]:
        import numpy as np

        states = suite.get_task_init_states(task_index)
        if type(state_index) is not int or not 0 <= state_index < len(states):
            raise ValueError("ordered LIBERO fixed initial state is out of range")
        state = np.asarray(states[state_index])
        env.seed(seed)
        env.reset()
        current = np.asarray(env.get_sim_state())
        if state.shape != current.shape or not np.isfinite(state).all():
            raise ValueError("saved LIBERO state is incompatible with the environment")
        return env.set_init_state(state)

    def identity(self) -> dict[str, Any]:
        return dict(self._task.native)

    def instruction(self) -> str:
        return self._task.instruction

    def _observation(self) -> dict[str, Any]:
        if self._raw is None:
            raise RuntimeError("ordered LIBERO has no live observation")
        feedback = task_context(
            self._env, self._tracker, self._skill_mappings, self._raw
        )
        return _policy_observation(
            self._raw,
            self._task.instruction,
            instance_pose_feedback=feedback,
        )

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._started:
            raise RuntimeError("ordered LIBERO environments cannot be reset in-place")
        self._started = True
        self._raw = self._initial
        self._goal_success = bool(self._env.env._check_success())
        return self._observation(), {
            "ordered_success": self._tracker.success,
            "prefix_progress": self._tracker.progress,
            "final_goal_success": self._goal_success,
            "observation_protocol": ASSISTANCE_CONTRACT,
        }

    def step(
        self, action: list[float]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        import numpy as np

        if self._raw is None:
            raise RuntimeError(
                "ordered LIBERO environment must be reset before stepping"
            )
        values = np.asarray(action, dtype=np.float32)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError("ordered LIBERO action must be a finite seven-vector")
        raw, _reward, _done, info = self._env.step(values)
        self._raw = raw
        self._goal_success = bool(self._env.env._check_success())
        ordered_success = self._tracker.observe()
        details = dict(info or {})
        details.update(
            {
                "ordered_success": ordered_success,
                "prefix_progress": self._tracker.progress,
                "skills_completed": self._tracker.completed,
                "total_skills": len(self._tracker.skills),
                "final_goal_success": self._goal_success,
                "observation_protocol": ASSISTANCE_CONTRACT,
            }
        )
        return (
            self._observation(),
            float(ordered_success),
            ordered_success,
            False,
            details,
        )

    def close(self) -> None:
        self._env.close()


def make_native_env(task: Task) -> _ProEnvironment | _OrderedEnvironment:
    """Construct exactly the runtime selected by a validated RoboRSI task."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from robots.libero.three_benchmark_service import _ensure_libero_config

    _ensure_libero_config()
    if task.benchmark == "libero_pro":
        return _ProEnvironment(task)
    if task.benchmark in {"libero_long_plus_plus", "libero_ultra_long"}:
        return _OrderedEnvironment(task)
    raise ValueError(f"unsupported three-benchmark environment: {task.benchmark}")
