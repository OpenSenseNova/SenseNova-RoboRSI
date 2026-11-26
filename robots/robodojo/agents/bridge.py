# Copyright 2026 The RoboRSI Contributors.
# Licensed under the Apache License, Version 2.0.

"""Adapt Runner messages to the submitted policy's original action/ACK loop."""

from __future__ import annotations

import copy
import io
import threading
from pathlib import Path
from typing import Any

from roborsi.evaluation.media import CaseMediaChannel
from roborsi.evaluation.messages import (
    Action,
    EpisodeStop,
    Observation,
    StepResult,
    Task,
)


def _model(config: dict):
    # Linux-only process/sandbox dependencies are loaded only on actual reset.
    from .submission.model import Model

    return Model(config)


def _json_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_values(item) for item in value]
    return value.tolist() if hasattr(value, "tolist") else value


class SubmissionAgent:
    """Own one policy and deliver each actual observation exactly once."""

    def __init__(
        self,
        directory: Path,
        *,
        parameters: dict,
        media: CaseMediaChannel,
        model_factory=_model,
    ):
        self.directory = directory.resolve()
        self.parameters = copy.deepcopy(parameters)
        self.media = media
        self._model_factory = model_factory
        self.model = None
        self.task = None
        self._episode = None
        self._next_step = 0
        self._closed = threading.Event()
        self._terminal = False

    def reset(self, task: Task) -> None:
        """Bind the exported layout, native evaluation seed and physical budget."""
        if task.benchmark != "robodojo" or task.native.get("config") != "arx_x5":
            raise ValueError("The submitted RoboDojo Agent requires arx_x5")
        if self.model is not None:
            self.close()
        config = {
            **self.parameters,
            "bench_name": "RoboDojo",
            "task_name": task.task_id,
            "env_cfg_type": "arx_x5",
            "action_type": "joint",
            "eval_batch": False,
            "seed": task.native["eval_seed"],
            "native_step_limit": task.max_steps,
            "trajectory_dir": str(self.directory / "submission-traces"),
        }
        self.task = task
        self._episode = None
        self._next_step = 0
        self._terminal = False
        self._closed.clear()
        self.model = self._model_factory(config)
        try:
            self.model.reset()
        except BaseException:
            self.close()
            raise

    def _decode(self, observation: Observation) -> dict:
        import numpy as np
        from PIL import Image

        if self.model is None or self._closed.is_set() or self._terminal:
            raise ValueError("Reset an active RoboDojo episode before feedback")
        if observation.benchmark != "robodojo":
            raise ValueError("RoboDojo observation benchmark differs")
        if self._episode is not None and observation.episode_id != self._episode:
            raise ValueError("RoboDojo observation belongs to another episode")
        if observation.step_index != self._next_step:
            raise ValueError("RoboDojo observation sequence differs")
        self.media.verify_observation(observation)
        fields = observation.data.get("media_fields", {})
        expected = {"cam_head", "cam_left_wrist", "cam_right_wrist"}
        if (
            set(fields) != expected
            or set(fields.values()) != set(range(3))
            or any(type(index) is not int for index in fields.values())
            or len(observation.media) != 3
        ):
            raise ValueError("RoboDojo requires three distinct native RGB cameras")
        decoded = copy.deepcopy(observation.data)
        vision = {}
        for name, index in fields.items():
            reference = observation.media[index]
            if reference.mime_type != "image/png":
                raise ValueError("Submitted policy requires lossless native RGB media")
            payload = self.media.read(
                reference, observation.episode_id, observation.step_index
            )
            with Image.open(io.BytesIO(payload)) as picture:
                if picture.format != "PNG" or picture.mode != "RGB":
                    raise ValueError("Expected lossless uint8 RGB PNG")
                vision[name] = {"color": np.array(picture, dtype=np.uint8, copy=True)}
        decoded["vision"] = vision
        self._episode = observation.episode_id
        return decoded

    def act(self, observation: Observation) -> Action | EpisodeStop:
        """Wait through thinking polls; never turn an empty poll into physics."""
        decoded = self._decode(observation)
        self.model.update_obs(decoded)
        while not self._closed.is_set():
            actions = self.model.get_action()
            if not actions:
                continue
            if len(actions) != 1:
                raise ValueError(
                    "Submitted policy must yield exactly one acknowledged action"
                )
            action = _json_values(actions[0])
            if action.get("_roborsi_control") == "finish":
                return EpisodeStop(
                    benchmark="robodojo",
                    episode_id=self._episode,
                    step_index=self._next_step,
                    reason="done",
                    details={"source": "submitted_policy"},
                )
            result = Action(
                benchmark="robodojo",
                episode_id=self._episode,
                step_index=self._next_step,
                data=action,
            )
            self._next_step += 1
            return result
        raise RuntimeError("Submitted RoboDojo episode was closed")

    def observe_end(self, step: StepResult) -> None:
        """ACK terminal physics without requesting another policy action or score."""
        if not (step.terminated or step.truncated):
            raise ValueError("RoboDojo terminal feedback must end the episode")
        decoded = self._decode(step.observation)
        self.model.end_episode(decoded)
        self._terminal = True

    def close(self) -> None:
        """Release the controller and its owned VLA service, including on errors."""
        if not self._closed.is_set():
            self._closed.set()
            if self.model is not None:
                self.model.shutdown()
