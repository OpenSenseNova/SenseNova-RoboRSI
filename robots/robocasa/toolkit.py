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

"""Dispatch RoboCasa benchmark primitives and record native state."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

from roborsi.dashboard.events import DashboardEventSink
from roborsi.session import EnvState
from roborsi.tools.toolkit import Toolkit
from roborsi.utils.logging import get_logger, get_output_dir
from robots.robocasa import tools as robocasa_tools

logger = get_logger("robocasa_toolkit")


class RoboCasaToolkit(Toolkit):
    """Toolkit for the RoboCasa robot."""

    VLA_TOOLS = frozenset({"rldx_skill", "rldx_arm"})

    def __init__(
        self,
        *,
        runtime_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
        enable_vla: bool = True,
        mode: str = "evaluation",
        state_output_dir: Path | str | None = None,
    ) -> None:
        """Create a RoboCasa toolkit, wiring the primitives and tools."""
        if mode != "evaluation":
            raise ValueError(f"unsupported RoboCasa toolkit mode: {mode!r}")
        state = EnvState(Path(state_output_dir or get_output_dir()))
        super().__init__(
            dashboard_events=dashboard_events,
            state=state,
        )
        self._enable_vla = enable_vla and runtime_kwargs.get("vla_client") is not None
        if not self._enable_vla:
            runtime_kwargs = {**runtime_kwargs, "vla_client": None}
        self.init_primitives(
            runtime_kwargs={
                **runtime_kwargs,
                "allow_reset": False,
            }
        )
        self._register_robocasa_tools()

    # ---- registration: one explicit add_tool per RoboCasa tool ----
    def _register_robocasa_tools(self) -> None:
        # Stateless perception tools: bind a state= kwarg via partial.
        state_handlers = {
            "view_env_state": partial(robocasa_tools.view_env_state, state=self._state),
            "back_project_batch": partial(
                robocasa_tools.back_project_batch, state=self._state
            ),
            "query_world_map": partial(
                robocasa_tools.query_world_map, state=self._state
            ),
        }
        for spec in robocasa_tools.TOOLS_SPEC:
            name = spec["name"]
            if name in self.VLA_TOOLS and not self._enable_vla:
                continue
            if name in state_handlers:
                handler = state_handlers[name]
            elif name == "finish":
                handler = robocasa_tools.finish
            else:
                handler = getattr(self._primitives, name, None)
                if handler is None:
                    continue  # spec without a backing primitive method
            self.add_tool(name, spec, handler)

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> dict[str, Any]:
        frame_start = self._action_frame_cursor
        self._action_frame_cursor = self._primitives.recorded_frame_count()
        record = robocasa_tools.dump_state(
            self._primitives,
            self._state,
            log={"command": command, "result": result, "elapsed_s": elapsed_s},
        )
        if self._dashboard_events.enabled or getattr(self, "capture_video", False):
            try:
                frames = self._primitives.frame_slice(frame_start)
                if frames:
                    candidate = f"action_{command['action']}.mp4"
                    saved = self._state.save(
                        candidate,
                        frames,
                        step=record.step_idx,
                        fps=20,
                    )
                    if saved is not None:
                        clip_path = self._state.artifact_path(
                            saved, step=record.step_idx
                        )
                        episode_path = self._state.artifact_path(
                            "episode.mp4", step=None
                        )
                        self._state.update_step_extras(
                            record.step_idx,
                            {
                                "videos": [
                                    {
                                        "video_ref": clip_path.relative_to(
                                            episode_path.parent
                                        ).as_posix(),
                                        "frame_start": 0,
                                        "frame_end": len(frames),
                                        "fps": 20,
                                    },
                                    {
                                        "video_ref": "episode.mp4",
                                        "frame_start": frame_start,
                                        "frame_end": self._action_frame_cursor,
                                        "fps": 20,
                                    },
                                ],
                            },
                        )
            except Exception as e:
                logger.warning(
                    "failed to save action clip for step %s: %s",
                    record.step_idx,
                    e,
                )
        out = robocasa_tools.view_env_state(record.step_idx, state=self._state)
        out["agent_elapsed_s"] = elapsed_s
        if result.get("interrupted"):
            out.update(result)
        return out

    def init_primitives(
        self,
        *,
        runtime_kwargs: dict[str, Any],
    ) -> None:
        """Wipe stale run artifacts, build the primitives, dump step 0."""
        self._state.reset()

        from robots.robocasa.primitives import RoboCasaPrimitives

        primitives = RoboCasaPrimitives(
            check_cancelled=self.raise_if_cancelled,
            **runtime_kwargs,
        )
        # The constructor resets the environment. This guarded call must not
        # restart an evaluation episode.
        primitives.reset()
        primitives.start_recording()
        self._action_frame_cursor = primitives.recorded_frame_count()
        record = robocasa_tools.dump_state(primitives, self._state, log=None)
        try:
            self._state.save(
                "success_criteria.md",
                primitives.dump_success_criteria(),
                step=None,
            )
        except Exception as e:
            logger.warning("failed to save success_criteria.md: %s", e)
        self._primitives = primitives
        self._publish_step(record)

    def close(self) -> None:
        """Flush the agent-side video buffer through ``EnvState``."""
        try:
            frames = self._primitives.stop_recording()
            if frames:
                self._state.save("episode.mp4", frames, step=None, fps=20)
        except Exception as e:
            logger.warning("failed to save episode video: %s", e)

    def solved(self) -> bool:
        """Return the success value from the final recorded environment state."""
        record = self._state.latest_record()
        return bool(record is not None and record.extras.get("success", False))
