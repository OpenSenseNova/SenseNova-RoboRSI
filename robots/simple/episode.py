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

"""Own one native episode, its physical budget and public observation boundary."""

from __future__ import annotations

import hashlib
import io
import json
import traceback
from pathlib import Path
from typing import Any

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
from roborsi.utils.logging import get_logger
from robots.simple.contracts import validate_actions, validate_task

logger = get_logger("simple.episode")


class SimpleEpisode:
    """Execute native action rows while HTTP indices count submitted chunks."""

    def __init__(
        self, directory: Path, runtime: Any, artifacts: ArtifactPublisher | None = None
    ) -> None:
        self.directory, self.runtime, self.artifacts = directory, runtime, artifacts
        self.media = CaseMediaChannel(directory / "media")
        (directory / "native").mkdir(parents=True, exist_ok=True)
        self.task: Task | None = None
        self.episode_id = ""
        self.native_steps = self.decisions = 0
        self.first_success: int | None = None
        self.ended = self.closed = False
        self.last_values = None

    def _observation(self, values: dict) -> Observation:
        import numpy as np
        from PIL import Image

        state, rgb = (
            np.asarray(values["state"], dtype=np.float32),
            np.asarray(values["rgb"]),
        )
        if state.shape != (32,) or not np.isfinite(state).all():
            raise ValueError("expected 32 finite public state values")
        if rgb.shape != (360, 640, 3) or rgb.dtype != np.uint8:
            raise ValueError("expected 640x360 RGB uint8 head-left image")
        output = io.BytesIO()
        Image.fromarray(rgb).save(output, format="PNG")
        ref = self.media.write(
            self.episode_id, self.decisions, output.getvalue(), "image/png"
        )
        if self.artifacts:
            self.artifacts.publish(
                ref.ref,
                "image",
                ref.mime_type,
                step_index=self.decisions,
                sha256=ref.sha256,
                size_bytes=ref.size_bytes,
            )
        return Observation(
            benchmark="simple",
            episode_id=self.episode_id,
            step_index=self.decisions,
            data={
                "state": state.tolist(),
                "native_env_steps": self.native_steps,
                "remaining_native_steps": self.task.max_steps - self.native_steps,
                "media_fields": {"head_stereo_left": 0},
            },
            media=(ref,),
        )

    def start(self, task: Task, episode_id: str) -> StartResult:
        """Validate the task before native creation; publish only RGB and state."""
        if self.task is not None or self.closed:
            raise RuntimeError("SIMPLE episode already used")
        validate_task(task)
        self.task, self.episode_id = task, episode_id
        try:
            self.last_values = self.runtime.start(task)
            return StartResult(
                task=task,
                observation=self._observation(self.last_values),
                capabilities=task.capabilities,
            )
        except BaseException:
            # Isaac may exit the interpreter inside close(), before RPC can reply.
            (self.directory / "native/start_failure.txt").write_text(
                traceback.format_exc()
            )
            logger.exception(
                "SIMPLE native start failed; preserving evidence before close"
            )
            self.close()
            raise

    def step(self, action: Action) -> StepResult:
        """Validate the whole chunk, execute rows and stop at native termination."""
        if self.task is None or self.closed or self.ended:
            raise RuntimeError("SIMPLE episode is not active")
        if (action.benchmark, action.episode_id, action.step_index) != (
            "simple",
            self.episode_id,
            self.decisions,
        ):
            raise ValueError("SIMPLE action identity or sequence differs")
        rows = validate_actions(action.data, self.task.max_steps - self.native_steps)
        self.runtime.submit(rows)
        for row in rows:
            values, native_success, native_done = self.runtime.step()
            if type(native_success) is not bool or type(native_done) is not bool:
                raise ValueError("native termination must use explicit booleans")
            self.native_steps += 1
            self.last_values = values
            if native_success and self.first_success is None:
                self.first_success = self.native_steps
            self.ended = (
                native_done
                or native_success
                or self.native_steps >= self.task.max_steps
            )
            with (self.directory / "native/actions.jsonl").open("a") as log:
                log.write(
                    json.dumps(
                        {
                            "native_step": self.native_steps,
                            "action": row,
                            "success": native_success,
                            "terminated": native_done,
                        }
                    )
                    + "\n"
                )
            if self.ended:
                break
        self.decisions += 1
        observation = self._observation(self.last_values)
        success = self.first_success is not None
        at_limit = self.native_steps >= self.task.max_steps and not success
        return StepResult(
            observation=observation,
            terminated=self.ended and not at_limit,
            truncated=at_limit,
            termination_reason=(
                "native_success"
                if success
                else "native_step_limit"
                if at_limit
                else "native_failure"
            )
            if self.ended
            else None,
        )

    def finish(self) -> NativeResult:
        """Persist native evidence without treating model completion as success."""
        if self.task is None or not self.ended or self.closed:
            raise RuntimeError("finish requires a terminated native episode")
        success = self.first_success is not None
        details = {
            "native_env_steps": self.native_steps,
            "decision_steps": self.decisions,
            "first_success_step": self.first_success,
            "official_1x_steps": self.task.native["official_steps"],
            "success_within_1x": success
            and self.first_success <= self.task.native["official_steps"],
            "success_within_execution_budget": success,
            "official_score": False,
            "score_protocol": self.task.native["score_protocol"],
        }
        path = self.directory / "native/result.json"
        path.write_text(json.dumps({"success": success, **details}, indent=2) + "\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if self.artifacts:
            self.artifacts.publish(
                "native/result.json", "native_result", "application/json", sha256=digest
            )
            self.artifacts.publish(
                "native/actions.jsonl", "log", "application/x-ndjson"
            )
        return NativeResult(
            task=self.task,
            episode_id=self.episode_id,
            success=success,
            score=float(success),
            details=details,
            source="native/result.json",
            source_sha256=digest,
        )

    def close(self) -> None:
        """Close the one owned runtime exactly once, including failed starts."""
        if not self.closed:
            self.closed = True
            self.runtime.close()
