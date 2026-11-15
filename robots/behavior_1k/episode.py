# Copyright 2026 The SenseNova-RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Native lifecycle and public observation boundary, owned by the worker."""

from __future__ import annotations

import hashlib
import io
import math
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
from robots.behavior_1k.catalog import CHANNELS, task_from_metadata, validate_task_name
from robots.behavior_1k.contracts import ActionBounds, validate_action


class BehaviorEpisode:
    """Execute exactly one native step per public action; never recover by reset."""

    def __init__(
        self, directory: Path, runtime: Any, repo_root: Path, data_root: Path
    ) -> None:
        self.directory = directory
        self.runtime = runtime
        self.repo_root = repo_root
        self.data_root = data_root
        self.media = CaseMediaChannel(directory / "media")
        self.task = None
        self.episode_id = ""
        self.steps = 0
        self.ended = False
        self.closed = False
        self.started = False
        self.failed = False
        self.result = None
        self.last_observation = None

    def inspect(self, specification: dict) -> Task:
        """Reset the native task and export the resolved instruction and limit."""
        if self.started or self.closed:
            raise ValueError("a worker owns exactly one episode")
        if set(specification) != {"task_id"}:
            raise ValueError(
                "inspection requires only task_id; native owns the step limit"
            )
        validate_task_name(specification["task_id"])
        self.started = True
        metadata = self.runtime.start(
            repo_root=str(self.repo_root),
            data_root=str(self.data_root),
            task_name=specification["task_id"],
            mode="public_test",
            instance_id=301,
            max_steps=None,
            include_depth=True,
            action_profile="r1pro_eef21_v1",
        )
        return task_from_metadata(metadata)

    def start(self, task: Task, episode_id: str) -> StartResult:
        actual = self.inspect({"task_id": task.task_id})
        if task != actual:
            raise ValueError("requested task differs from native reset")
        self.task = actual
        self.episode_id = episode_id
        self.last_observation = self._observe()
        return StartResult(
            task=actual,
            observation=self.last_observation,
            capabilities=actual.capabilities,
        )

    def _observe(self) -> Observation:
        import numpy as np
        from PIL import Image

        raw = self.runtime.observe()
        if raw["step_index"] != self.steps:
            raise RuntimeError("native observation counter differs")
        media = []
        for name in CHANNELS:
            values = np.asarray(raw[name])
            if (
                values.dtype != np.uint8
                or values.ndim != 3
                or values.shape[2] not in (3, 4)
            ):
                raise ValueError("native camera must return uint8 RGB(A)")
            output = io.BytesIO()
            Image.fromarray(values[..., :3]).save(output, format="PNG")
            media.append(
                self.media.write(
                    self.episode_id, self.steps, output.getvalue(), "image/png"
                )
            )
        return Observation(
            benchmark="behavior_1k",
            episode_id=self.episode_id,
            step_index=self.steps,
            data={
                "proprio": raw["proprio"],
                "action_bounds": raw["action_bounds"],
                "media_fields": {name: i for i, name in enumerate(CHANNELS)},
                "depth_encoding": {
                    "kind": "uint8_inverse_log",
                    "near_m": 0.01,
                    "far_m": 10.0,
                    "shift_m": 3.5,
                },
            },
            media=tuple(media),
        )

    def _check(self, request: Action | EpisodeStop) -> None:
        if self.task is None or self.closed or self.ended or self.failed:
            raise ValueError("episode is not active")
        if (request.benchmark, request.episode_id, request.step_index) != (
            "behavior_1k",
            self.episode_id,
            self.steps,
        ):
            raise ValueError("episode identity or action sequence differs")

    def step(self, action: Action) -> StepResult:
        self._check(action)
        if set(action.data) != {"action"}:
            raise ValueError("action requires only the 21D action vector")
        bounds = self.last_observation.data["action_bounds"]
        values = validate_action(
            action.data["action"],
            ActionBounds(
                tuple(-math.inf if v is None else v for v in bounds["low"]),
                tuple(math.inf if v is None else v for v in bounds["high"]),
            ),
        )
        # Once native execution starts, any exception invalidates this episode.
        self.failed = True
        native = self.runtime.step(values)
        if not native.get("ok") or native["step_index"] != self.steps + 1:
            raise RuntimeError(f"native BEHAVIOR step failed: {native}")
        self.steps += 1
        terminated, truncated = native["terminated"], native["truncated"]
        self.ended = terminated or truncated
        self.last_observation = self._observe()
        self.failed = False
        return StepResult(
            observation=self.last_observation,
            terminated=terminated,
            truncated=truncated,
            termination_reason=("native_termination" if terminated else "episode_limit")
            if self.ended
            else None,
        )

    def stop(self, request: EpisodeStop) -> StepResult:
        """End without an extra native action; score the current native state."""
        self._check(request)
        self.ended = True
        return StepResult(
            observation=self.last_observation,
            terminated=False,
            truncated=True,
            termination_reason=request.reason,
        )

    def finish(self) -> NativeResult:
        if not self.ended or self.failed or self.closed:
            raise ValueError("only a valid ended episode can be scored")
        if self.result is not None:
            return self.result
        raw = self.runtime.evaluate()
        score = raw.get("q_score")
        if (
            raw.get("invalid")
            or raw.get("infrastructure_failure")
            or raw.get("metrics_errors")
            or raw.get("steps") != self.steps
            or type(raw.get("success")) is not bool
            or type(score) not in (int, float)
            or not math.isfinite(score)
        ):
            raise RuntimeError(f"invalid native scoring result: {raw}")
        path = self.directory / "native/score.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, raw)
        self.result = NativeResult(
            task=self.task,
            episode_id=self.episode_id,
            success=raw["success"],
            score=score,
            details=raw,
            source="native/score.json",
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        return self.result

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.runtime.close()
