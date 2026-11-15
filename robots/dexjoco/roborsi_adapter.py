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

"""Map each RoboRSI action to exactly one DexJoCo control step."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

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
from robots.dexjoco.benchmark_service import make_task, runtime_info
from robots.dexjoco.contracts import (
    ALIGNMENT,
    LIMITS,
    public_observation,
    validate_vector,
)
from robots.dexjoco.native import ProcessNative, task_config


class DexJocoAdapter:
    """Keep privileged state/scoring private and retain native early failures."""

    def __init__(
        self,
        directory: Path,
        upstream: Path,
        *,
        multiplier: int = 10,
        native_factory=ProcessNative,
        artifacts=None,
    ) -> None:
        self.directory = Path(directory)
        self.upstream = Path(upstream)
        self.multiplier = multiplier
        self.factory = native_factory
        self.artifacts = artifacts
        self.media = CaseMediaChannel(self.directory / "media")
        self.native_dir = self.directory / "native"
        self.native_dir.mkdir(parents=True, exist_ok=True)
        self.env = None
        self.task = None
        self.steps = 0
        self.ended = False
        self.closed = False
        self.success = False
        self.first_success = None
        self.reason = None
        self.observation = None
        self.result = None

    def runtime_info(self):
        """Expose identical source metadata before and after cleanup."""
        return runtime_info()

    def _record(self, phase: str, raw: dict, **details) -> None:
        import numpy as np

        def encode(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            raise TypeError(type(value).__name__)

        row = {"step": self.steps, "phase": phase, "state": raw["state"], **details}
        with (self.native_dir / "trajectory.jsonl").open("a") as stream:
            stream.write(json.dumps(row, default=encode, allow_nan=False) + "\n")

    def _observe(self, raw: dict) -> Observation:
        from PIL import Image

        public, images = public_observation(
            raw, self.cfg["camera_mapping"], self.dual, self.task.instruction
        )
        media = []
        public["camera_names"] = list(images)
        public["control_step"] = self.steps
        # Before 1x, neither the policy file nor its observation announces 10x.
        limit = LIMITS[self.task.task_id]
        if self.steps >= limit and self.multiplier == 10 and not self.ended:
            limit *= 10
        public["remaining_steps"] = max(0, limit - self.steps)
        for rgb in images.values():
            stream = io.BytesIO()
            Image.fromarray(rgb).save(stream, format="PNG", compress_level=6)
            media.append(
                self.media.write(
                    self.episode, self.steps, stream.getvalue(), "image/png"
                )
            )
        self.observation = Observation(
            benchmark="dexjoco",
            episode_id=self.episode,
            step_index=self.steps,
            data=public,
            media=tuple(media),
        )
        return self.observation

    def start(self, task: Task, episode_id: str) -> StartResult:
        """Validate the complete catalog entry before starting an environment."""
        if self.task is not None or self.closed:
            raise ValueError("DexJoCo Adapter owns exactly one episode")
        expected = make_task(self.upstream, task.task_id, task.seed, self.multiplier)
        if task != expected:
            raise ValueError("DexJoCo task differs from the pinned catalog")
        self.task, self.episode = task, episode_id
        self.cfg = task_config(self.upstream, task.task_id)
        self.dual = self.cfg["robot_type"] == "dual_arm"
        try:
            self.env = self.factory(
                self.upstream, task.task_id, task.seed, self.multiplier
            )
            raw, info = self.env.reset()
            write_json_atomic(
                self.native_dir / "horizon.json", {"patch": self.env.horizon}
            )
            self._record("reset", raw, seed=task.seed, info=info)
            return StartResult(
                task=task,
                observation=self._observe(raw),
                capabilities=task.capabilities,
            )
        except BaseException:
            self.close()
            raise

    def _identity(self, message) -> None:
        if self.env is None or self.ended or self.closed:
            raise ValueError("DexJoCo episode is not active")
        if (message.benchmark, message.episode_id, message.step_index) != (
            "dexjoco",
            self.episode,
            self.steps,
        ):
            raise ValueError("DexJoCo episode or step mismatch")

    def step(self, action: Action) -> StepResult:
        """Execute one row, preserving native success and failure predicates."""
        self._identity(action)
        row = validate_vector(action.data, self.dual)
        if self.task.task_id == "click_mouse" and self.steps < 30 and row != ALIGNMENT:
            raise ValueError(
                "Click Mouse requires the original 30-step startup alignment"
            )
        raw, reward, terminated, truncated, info = self.env.step(row)
        self.steps += 1
        if info.get("succeed", False) and self.first_success is None:
            self.first_success = self.steps
        self.ended = bool(terminated or truncated)
        self.success = self.ended and bool(info.get("succeed", False))
        if self.steps >= self.task.max_steps and not self.ended:
            raise RuntimeError(
                "Native simulator failed to terminate at its pinned limit"
            )
        self.reason = (
            "native_success"
            if self.success
            else "episode_limit"
            if self.ended and self.steps == self.task.max_steps
            else "native_failure"
            if self.ended
            else None
        )
        self._record(
            "startup_alignment"
            if self.task.task_id == "click_mouse" and self.steps <= 30
            else "policy",
            raw,
            action=row,
            reward=reward,
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info,
        )
        if self.steps == LIMITS[self.task.task_id]:
            if not self.ended and self.multiplier == 10:
                (self.native_dir / "checkpoint_simulator_1x.npz").write_bytes(
                    self.env.checkpoint()
                )
            write_json_atomic(
                self.native_dir / "checkpoint_1x.json",
                {
                    "step": self.steps,
                    "success_by_1x": self.first_success is not None,
                    "native_terminal": self.ended,
                },
            )
        # Rewards and task-specific terminal details never enter model observations.
        return StepResult(
            observation=self._observe(raw),
            terminated=bool(terminated),
            truncated=bool(truncated),
            termination_reason=self.reason,
        )

    def stop(self, request: EpisodeStop) -> StepResult:
        """Score an invalid/ended policy as a failure without granting more steps."""
        self._identity(request)
        self.ended = True
        self.reason = "policy_failure"
        write_json_atomic(
            self.native_dir / "policy_failure.json",
            {"reason": request.reason, "step": self.steps},
        )
        return StepResult(
            observation=self.observation,
            terminated=False,
            truncated=True,
            termination_reason=self.reason,
        )

    def finish(self) -> NativeResult:
        """Write auditable 1x and cumulative 10x outcomes, never inferred success."""
        if not self.ended:
            raise ValueError("DexJoCo episode must end before scoring")
        if self.result is not None:
            return self.result
        details = {
            "task": self.task.task_id,
            "seed": self.task.seed,
            "regime": "rand_full",
            "protocol": self.task.native["protocol"],
            "profile": "rand-full-v2",
            "simulator_steps": self.steps,
            "first_success_step": self.first_success,
            "success_by_1x": self.first_success is not None
            and self.first_success <= LIMITS[self.task.task_id],
            "success_by_10x": self.success if self.multiplier == 10 else None,
            "standard_step_limit": LIMITS[self.task.task_id],
            "extended_step_limit": self.task.max_steps,
            "termination_reason": self.reason,
            "outcome": "success"
            if self.success
            else "policy_failure"
            if self.reason == "policy_failure"
            else "failure",
        }
        path = self.native_dir / "score.json"
        write_json_atomic(path, details)
        if self.artifacts is not None:
            for source in sorted(self.native_dir.iterdir()):
                if source.name != "score.json" and source.is_file():
                    self.artifacts.publish(
                        "native/" + source.name, "file", "application/octet-stream"
                    )
        self.result = NativeResult(
            task=self.task,
            episode_id=self.episode,
            success=self.success,
            score=float(self.success),
            details=details,
            source="native/score.json",
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        return self.result

    def close(self) -> None:
        """Release worker resources after success, cancellation or startup failure."""
        self.closed = True
        if self.env is not None:
            self.env.close()
            self.env = None
