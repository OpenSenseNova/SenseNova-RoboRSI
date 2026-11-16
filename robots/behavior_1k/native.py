# Copyright 2026 The SenseNova-RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Isolated BEHAVIOR-1K simulator process.

Isaac Sim owns process-global state and is not safe to move between the
Uvicorn event loop and a worker thread.  This module keeps the official
Evaluator, OmniGibson and Isaac lifecycle in one child process.  The parent
communicates through the framework RPC worker and never imports
the simulator stack.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any

from robots.behavior_1k.contracts import public_proprio

EXPECTED_CONTROLLER_ORDER = (
    "base",
    "trunk",
    "arm_left",
    "gripper_left",
    "arm_right",
    "gripper_right",
)


def _to_plain(value: Any) -> Any:
    """Convert simulator scalars and arrays into plain Python values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _to_plain(tolist())
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return str(value)


def _to_image(value: Any) -> Any:
    """Move an OmniGibson image to a compact CPU uint8 ndarray."""

    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach().cpu().numpy()
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return np.ascontiguousarray(value.astype(np.uint8, copy=False))
    except ImportError:  # pragma: no cover - behavior env always provides numpy
        pass
    return value


def _to_depth_image(value: Any) -> Any:
    """Convert metric linear depth to a stable, viewable uint8 image.

    The source sensor remains ``depth_linear``.  The public image uses a
    fixed 0.01--10 metre logarithmic scale with nearer pixels brighter, so
    frames remain comparable across observations and camera views.
    """

    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach().cpu().numpy()
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            depth = np.asarray(value, dtype=np.float32).squeeze()
            if depth.ndim != 2:
                raise ValueError(
                    f"depth image must be two-dimensional; got {depth.shape}"
                )
            valid = np.isfinite(depth) & (depth > 0)
            clipped = np.clip(depth, 0.01, 10.0)
            shift = 3.5
            low = np.log(0.01 + shift)
            high = np.log(10.0 + shift)
            normalized = (np.log(clipped + shift) - low) / (high - low)
            grayscale = np.rint((1.0 - normalized) * 255.0).astype(np.uint8)
            grayscale[~valid] = 0
            return np.ascontiguousarray(np.repeat(grayscale[..., None], 3, axis=-1))
    except ImportError:  # pragma: no cover - behavior env always provides numpy
        pass
    return value


def _plain_metric(value: Any) -> Any:
    """Make official metric output serializable without changing its meaning."""

    value = _to_plain(value)
    if isinstance(value, dict):
        return {str(key): _plain_metric(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_metric(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        # JSON has no representation for +/-inf.  Null is explicit and keeps
        # the result file valid when a normalized metric has zero distance.
        return None
    return _to_plain(value)


class Behavior1KEpisode:
    """Official evaluator state living entirely in the simulator child."""

    def __init__(self) -> None:
        self.evaluator: Any = None
        self.repo_root: Path | None = None
        self.data_root: Path | None = None
        self.task_name = ""
        self.mode = "public_test"
        self.instance_id: int | None = None
        self.max_steps = 500
        self.include_depth = True
        self.action_profile = "r1pro_eef21_v1"
        self.steps = 0
        self.terminated = False
        self.truncated = False
        self.invalid = False
        self.infrastructure_failure = False
        self.recovery_events = 0
        self.last_obs: dict[str, Any] = {}
        self.task_instruction = ""
        self.bounds_low: tuple[float, ...] = ()
        self.bounds_high: tuple[float, ...] = ()
        self._og: Any = None

    def start(
        self,
        *,
        repo_root: str,
        data_root: str,
        task_name: str,
        mode: str,
        instance_id: int,
        max_steps: int | None,
        include_depth: bool = True,
        action_profile: str = "r1pro_eef21_v1",
    ) -> dict[str, Any]:
        self.close()
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.data_root = Path(data_root).expanduser().resolve()
        self.task_name = str(task_name)
        self.mode = str(mode)
        self.instance_id = int(instance_id)
        self.max_steps = None if max_steps is None else int(max_steps)
        from robots.behavior_1k.official import (
            action_dimension,
            validate_include_depth,
        )

        action_dimension(action_profile)
        if action_profile != "r1pro_eef21_v1":
            raise ValueError("only the official 21D pipeline is supported")
        self.action_profile = action_profile
        self.include_depth = validate_include_depth(action_profile, include_depth)
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if not self.repo_root.exists():
            raise FileNotFoundError(
                f"BEHAVIOR-1K repo does not exist: {self.repo_root}"
            )
        if not self.data_root.exists():
            raise FileNotFoundError(
                f"BEHAVIOR data root does not exist: {self.data_root}"
            )

        from robots.behavior_1k.provenance import verify_native_source

        verify_native_source(self.repo_root)
        self._prepare_runtime()
        self._build_evaluator()
        if self.max_steps is None:
            # Let the official evaluator apply its task-level human statistics,
            # then share that exact limit with the adapter and model prompt.
            self.max_steps = self.evaluator.env.task_config["termination_config"][
                "max_steps"
            ]
            if type(self.max_steps) is not int or self.max_steps <= 0:
                raise RuntimeError(
                    "Official evaluator did not resolve a positive max_steps"
                )
        self.evaluator.reset()
        self.evaluator.load_task_instance(self.instance_id)
        self.evaluator.reset()

        self.last_obs = dict(self.evaluator.obs or {})
        self.steps = 0
        self.terminated = False
        self.truncated = False
        self.invalid = False
        self.infrastructure_failure = False
        self.recovery_events = 0
        self.bounds_low, self.bounds_high = self._read_action_bounds()
        self._validate_action_layout()
        self.task_instruction = self._read_task_instruction()
        return self.metadata()

    def _prepare_runtime(self) -> None:
        assert self.repo_root is not None
        assert self.data_root is not None
        os.environ["OMNIGIBSON_DATA_PATH"] = str(self.data_root)
        os.environ["OMNIGIBSON_HEADLESS"] = "True"
        os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        # OmniGibson sets this during its normal launcher path.  Set it before
        # importing Isaac Sim as this worker is launched directly by Python.
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        omnigibson_root = self.repo_root / "OmniGibson"
        if str(omnigibson_root) not in sys.path:
            sys.path.insert(0, str(omnigibson_root))

        # Import and launch Isaac only in this process.  The explicit library
        # path avoids an older CUDA compatibility shim shadowing the host
        # driver's libcuda.so and producing CUDA error 803.
        import omnigibson as og
        from omnigibson.eval.utils.eval_utils import DEFAULT_EVAL_SEED, seed_everything
        from omnigibson.macros import gm

        gm.HEADLESS = True
        seed_everything(DEFAULT_EVAL_SEED)
        self._og = og
        if og.sim is None:
            og.launch(
                physics_dt=1.0 / 120.0,
                rendering_dt=1.0 / 30.0,
                sim_step_dt=1.0 / 30.0,
                device=os.environ.get("BEHAVIOR_1K_SIM_DEVICE", "cpu"),
                viewer_width=1280,
                viewer_height=720,
            )

    def _build_evaluator(self) -> None:
        assert self.repo_root is not None
        from omegaconf import OmegaConf
        from omnigibson.macros import gm

        # The pinned evaluator writes two protected macro values at import
        # time.  Unlock only for that import so the third-party submodule can
        # remain pristine while still working with the installed OmniGibson.
        with gm.unlocked():
            from omnigibson.eval.evaluator import Evaluator

        robot_config = OmegaConf.load(
            str(self.repo_root / "OmniGibson" / "omnigibson" / "eval" / "r1pro.yaml")
        )
        from robots.behavior_1k.official import EEF21, robot_config_for_profile

        if self.action_profile == EEF21:
            robot_config = OmegaConf.create(
                robot_config_for_profile(
                    OmegaConf.to_container(robot_config, resolve=True),
                    self.action_profile,
                )
            )
        wrapper = (
            "omnigibson.eval.wrappers.RGBDFullResWrapper"
            if self.include_depth
            else "omnigibson.eval.wrappers.DefaultWrapper"
        )
        cfg = OmegaConf.create(
            {
                "env_wrapper": {"_target_": wrapper},
                "policy_name": "local",
                "model": {
                    "_target_": "omnigibson.eval.policies.LocalPolicy",
                    "action_dim": None,
                },
                "headless": True,
                "partial_scene_load": True,
                "max_steps": self.max_steps,
                "write_video": False,
                "mode": self.mode,
                "seed": 0,
                "task": {"name": self.task_name},
                "robot": robot_config,
            }
        )
        self.evaluator = Evaluator(cfg)
        if self.action_profile == EEF21:
            from omnigibson.eval.wrappers import RGBDFullResWrapper

            if not isinstance(self.evaluator.env, RGBDFullResWrapper):
                raise RuntimeError(
                    "official evaluator must use official RGBDFullResWrapper"
                )

    def _read_action_bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        space = self.evaluator.robot.action_space
        low = tuple(float(value) for value in _to_plain(space.low))
        high = tuple(float(value) for value in _to_plain(space.high))
        return low, high

    def _validate_action_layout(self) -> None:
        from robots.behavior_1k.official import EEF21, action_dimension

        expected_dim = action_dimension(self.action_profile)
        action_dim = int(self.evaluator.robot.action_dim)
        if action_dim != expected_dim:
            raise RuntimeError(
                f"official R1Pro action dimension is {action_dim}, expected {expected_dim}"
            )
        raw_order = tuple(
            str(item) for item in self.evaluator.robot._raw_controller_order
        )
        if raw_order != EXPECTED_CONTROLLER_ORDER:
            raise RuntimeError(
                "official R1Pro controller order changed: "
                f"got {raw_order!r}, expected {EXPECTED_CONTROLLER_ORDER!r}"
            )
        if self.action_profile == EEF21:
            from omnigibson.controllers import ControllerView

            for arm in ("left", "right"):
                group_key, _ = self.evaluator.robot.controllers[f"arm_{arm}"]
                if (
                    ControllerView.get_controller_type_str(group_key)
                    != "InverseKinematicsController"
                    or ControllerView.get_mode(group_key) != "absolute_pose"
                    or ControllerView.get_command_dim(group_key) != 6
                ):
                    raise RuntimeError(
                        "official requires official absolute_pose arm IK"
                    )

    def _read_task_instruction(self) -> str:
        task = self.evaluator.env.task
        conditions = (
            getattr(task, "activity_natural_language_goal_conditions", None) or []
        )
        if isinstance(conditions, str):
            return conditions
        return " ".join(str(condition) for condition in conditions)

    def _public_bounds(self) -> dict[str, list[float | None]]:
        def finite_or_none(values: tuple[float, ...]) -> list[float | None]:
            return [value if math.isfinite(value) else None for value in values]

        return {
            "low": finite_or_none(self.bounds_low),
            "high": finite_or_none(self.bounds_high),
        }

    def metadata(self) -> dict[str, Any]:
        from robots.behavior_1k.official import EEF21, action_dimension

        return {
            "status": "ready" if self.evaluator is not None else "stopped",
            "task_name": self.task_name,
            "mode": self.mode,
            "instance_id": self.instance_id,
            "task_instruction": self.task_instruction,
            "max_steps": self.max_steps,
            "include_depth": self.include_depth,
            "action_dim": action_dimension(self.action_profile),
            **({"action_profile": EEF21} if self.action_profile == EEF21 else {}),
            "action_bounds": self._public_bounds(),
            "invalid": self.invalid,
            "infrastructure_failure": self.infrastructure_failure,
            "recovery_events": self.recovery_events,
        }

    def _public_observation(self) -> dict[str, Any]:
        if self.evaluator is None:
            raise RuntimeError("episode is not initialized")
        obs = self.last_obs
        robot_name = self.evaluator.robot.name
        result: dict[str, Any] = {
            "step_index": self.steps,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "action_bounds": self._public_bounds(),
            "proprio": {},
        }
        for role, camera_name in self.evaluator.robot_camera_names.items():
            key = f"{camera_name}::rgb"
            if key in obs:
                result[f"{role}_image"] = _to_image(obs[key])
            if self.include_depth:
                depth_key = f"{camera_name}::depth_linear"
                if depth_key in obs:
                    result[f"{role}_depth_image"] = _to_depth_image(obs[depth_key])
        vector_key = f"{robot_name}::proprio"
        if vector_key in obs:
            result["proprio"] = public_proprio(_to_plain(obs[vector_key]))
        prefix = f"{robot_name}::proprio::"
        for key, value in obs.items():
            if key.startswith(prefix):
                result["proprio"][key[len(prefix) :]] = _to_plain(value)
        # The explicit construction above is the privilege boundary.  Do not
        # return evaluator.env.info, task predicates, object poses or scene data.
        return result

    def observe(self) -> dict[str, Any]:
        return self._public_observation()

    def step(self, action: list[float]) -> dict[str, Any]:
        if self.evaluator is None:
            return {
                "ok": False,
                "error": "episode_not_initialized",
                "step_index": self.steps,
            }
        if self.terminated or self.truncated:
            return {"ok": False, "error": "episode_done", "step_index": self.steps}
        if self.invalid or self.infrastructure_failure:
            return {
                "ok": False,
                "error": "episode_invalid",
                "step_index": self.steps,
                "invalid": self.invalid,
                "infrastructure_failure": self.infrastructure_failure,
            }

        try:
            import torch

            action_tensor = torch.tensor(action, dtype=torch.float32)
            raw_obs, reward, terminated, truncated, info = self.evaluator.env.step(
                action_tensor,
                n_render_iterations=1,
            )
            raw_obs = self.evaluator._sync_lights_and_get_obs(raw_obs)
            self.last_obs = self.evaluator._preprocess_obs(raw_obs)
            self.evaluator.robot_action = action_tensor
            self.steps += 1
            self.terminated = bool(terminated)
            # Enforce the episode budget at this boundary as well.  Some
            # OmniGibson task wrappers report the time limit one step late.
            self.truncated = bool(truncated) or self.steps >= self.max_steps

            if self.terminated or self.truncated:
                self.evaluator.n_trials += 1
                if info.get("done", {}).get("success", False):
                    self.evaluator.n_success_trials += 1

            # This is the official evaluator metric lifecycle.  In
            # particular, TaskMetric must see the raw environment observation
            # and the official reward/termination/info tuple.
            for metric in self.evaluator.metrics:
                metric.step(
                    self.evaluator.env,
                    action_tensor,
                    raw_obs,
                    reward,
                    self.terminated,
                    self.truncated,
                    info,
                )
            return {
                "ok": True,
                "step_index": self.steps,
                "terminated": self.terminated,
                "truncated": self.truncated,
            }
        except Exception as exc:
            recovered = False
            self.invalid = True
            self.infrastructure_failure = True
            return {
                "ok": False,
                "error": "simulator_error" if recovered else "simulator_unrecoverable",
                "message": str(exc),
                "reset": recovered,
                "invalid": True,
                "infrastructure_failure": not recovered,
                "step_index": self.steps,
            }

    def evaluate(self) -> dict[str, Any]:
        if self.evaluator is None:
            return {"status": "stopped"}
        metrics: dict[str, Any] = {}
        errors: dict[str, str] = {}
        if self.steps == 0:
            from omnigibson.metrics import TaskMetric

        for metric in self.evaluator.metrics:
            try:
                if self.steps == 0 and isinstance(metric, TaskMetric):
                    # The pinned MetricBase.aggregate skips empty reset state.
                    # Compute the official final predicates/time directly; a
                    # synthetic metric.step would incorrectly count one action.
                    values = metric._compute_episode_metrics(self.evaluator.env, {})
                else:
                    values = metric.aggregate(self.evaluator.env)
                metrics.update(_plain_metric(values))
            except Exception as exc:
                errors[type(metric).__name__] = str(exc)
        q_score = None
        if isinstance(metrics.get("q_score"), dict):
            q_score = metrics["q_score"].get("final")
        if self.infrastructure_failure:
            status = "infrastructure_failure"
        elif self.invalid:
            status = "invalid"
        elif self.terminated or self.truncated:
            status = "finished"
        else:
            status = "incomplete"
        result = {
            "status": status,
            "success": bool(self.evaluator.env.task.success),
            "steps": self.steps,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "invalid": self.invalid,
            "infrastructure_failure": self.infrastructure_failure,
            "recovery_events": self.recovery_events,
            "n_trials": self.evaluator.n_trials,
            "n_success_trials": self.evaluator.n_success_trials,
            "metrics": metrics,
            "metrics_errors": errors,
            "q_score": q_score,
        }
        return result

    def close(self) -> None:
        evaluator = self.evaluator
        self.evaluator = None
        if evaluator is None:
            return
        try:
            evaluator.__exit__(None, None, None)
        except SystemExit:
            pass
        except Exception:
            try:
                evaluator.env.close()
            except Exception:
                pass
