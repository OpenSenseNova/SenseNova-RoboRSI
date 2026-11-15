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

"""LIBERO's RoboRSI boundary preserves native task, images, actions and scoring."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from roborsi.evaluation.media import CaseMediaChannel
from roborsi.evaluation.messages import Action, Task, to_envelope
from roborsi.evaluation.services import AdapterFacade
from robots.libero.roborsi_adapter import ACTION_SCHEMA, LiberoRoboRSIAdapter


def _task(*, max_steps: int = 2) -> Task:
    return Task(
        benchmark="libero",
        task_id="3",
        seed=7,
        instruction="Place the block.",
        max_steps=max_steps,
        native={"suite": "libero_spatial"},
        capabilities={"action_schema": ACTION_SCHEMA},
    )


def _action(step: int, vector: list[float] | None = None) -> Action:
    return Action(
        benchmark="libero",
        episode_id="episode-42",
        step_index=step,
        data={"vector": vector if vector is not None else [0.0] * 7},
    )


class FakeNative:
    def __init__(self, *, max_steps: int, success: bool = True) -> None:
        self.max_steps = max_steps
        self.success = success
        self.steps = 0
        self.closed = False

    def get_env_meta(self) -> dict:
        return {
            "suite": "libero_spatial",
            "task": 3,
            "seed": 7,
            "max_episode_steps": self.max_steps,
            "reset_state_id": 27,
        }

    def get_task_language(self) -> str:
        return "Place the block."

    def _observation(self) -> dict:
        return {
            "main_images": np.full((2, 3, 3), 11, dtype=np.uint8),
            "wrist_images": np.full((2, 3, 3), 22, dtype=np.uint8),
            "states": np.arange(8, dtype=np.float32),
            "task_descriptions": "Place the block.",
        }

    def reset(self) -> tuple[dict, dict]:
        return self._observation(), {}

    def step(self, action: list[float]) -> tuple[dict, float, bool, bool, dict]:
        assert action == [0.0] * 7
        self.steps += 1
        return (
            self._observation(),
            1.0 if self.success else 0.0,
            self.success,
            False,
            {},
        )

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("success", [True, False])
def test_native_termination_sets_score_and_preserves_two_images(
    tmp_path: Path, success: bool
) -> None:
    native = FakeNative(max_steps=1, success=success)
    adapter = LiberoRoboRSIAdapter(
        tmp_path / "media",
        tmp_path / "native",
        env_factory=lambda suite, task, seed, max_steps: native,
    )
    task = _task(max_steps=1)
    started = adapter.start(task, "episode-42")
    assert started.task == task
    assert started.capabilities["reset_state_id"] == 27
    assert started.observation.data["states"] == list(range(8))
    assert started.observation.data["media_fields"] == {
        "main_images": 0,
        "wrist_images": 1,
    }
    channel = CaseMediaChannel(tmp_path / "media")
    for index, reference in enumerate(started.observation.media):
        encoded = channel.read(reference, "episode-42", 0)
        assert encoded.startswith(b"\x89PNG")
        assert np.all(np.asarray(Image.open(io.BytesIO(encoded))) == 11 + index * 11)
    stepped = adapter.step(_action(0))
    assert stepped.terminated is success
    assert stepped.truncated is not success
    assert stepped.observation.step_index == 1
    score = adapter.finish()
    assert score.success is success and score.score == float(success)
    assert score.task == task and score.episode_id == "episode-42"
    assert score.source == "native/score.json"
    assert (
        json.loads((tmp_path / score.source).read_text())["native_terminated"]
        is success
    )
    assert (
        hashlib.sha256((tmp_path / score.source).read_bytes()).hexdigest()
        == score.source_sha256
    )
    channel.verify_observation(stepped.observation)
    adapter.close()
    assert native.closed


def test_bad_action_does_not_advance_native_environment(tmp_path: Path) -> None:
    native = FakeNative(max_steps=2)
    adapter = LiberoRoboRSIAdapter(
        tmp_path / "media",
        tmp_path / "native",
        env_factory=lambda suite, task, seed, max_steps: native,
    )
    adapter.start(_task(), "episode-42")
    with pytest.raises(ValueError, match="seven finite numbers"):
        adapter.step(_action(0, [0.0] * 6))
    assert native.steps == 0
    assert adapter.step(_action(0)).terminated
    adapter.close()


def test_wrong_native_identity_closes_environment_before_scoring(
    tmp_path: Path,
) -> None:
    class WrongSeed(FakeNative):
        def get_env_meta(self) -> dict:
            return {**super().get_env_meta(), "seed": 8}

    native = WrongSeed(max_steps=2)
    facade = AdapterFacade(
        LiberoRoboRSIAdapter(
            tmp_path / "media",
            tmp_path / "native",
            env_factory=lambda suite, task, seed, max_steps: native,
        )
    )
    wire = json.dumps(
        to_envelope(_task(), benchmark="libero", episode_id="episode-42").as_dict()
    )
    with pytest.raises(ValueError, match="identity differs"):
        facade.start(wire)
    assert native.closed
    assert not (tmp_path / "native" / "score.json").exists()


def test_default_factory_rejects_task_actually_loaded_by_rlinf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace
    from unittest.mock import Mock

    from robots.libero import env_server

    closed = []
    raw = SimpleNamespace(
        task_suite=SimpleNamespace(name="libero_spatial"),
        task_ids=np.array([4]),  # The requested task is 3.
        seed=7,
        cfg=SimpleNamespace(max_episode_steps=2),
        reset_state_ids=np.array([27]),
        env=SimpleNamespace(close=lambda: closed.append(True)),
    )
    factory = Mock(return_value=raw)
    monkeypatch.setattr(env_server, "make_env", factory)
    facade = AdapterFacade(
        LiberoRoboRSIAdapter(tmp_path / "media", tmp_path / "native")
    )
    wire = json.dumps(
        to_envelope(_task(), benchmark="libero", episode_id="episode-42").as_dict()
    )
    with pytest.raises(ValueError, match="identity differs"):
        facade.start(wire)
    assert factory.call_args.kwargs["video_dir"] == (
        tmp_path / "native" / "primitive_videos"
    )
    assert closed == [True]
    assert not (tmp_path / "native" / "score.json").exists()


@pytest.mark.parametrize("language", [None, "Another task."])
def test_unconfirmed_instruction_is_rejected_before_reset(tmp_path, language):
    class Unconfirmed(FakeNative):
        def get_task_language(self):
            return language

        def reset(self):
            pytest.fail("a mismatched task must not reset or publish observations")

    native = Unconfirmed(max_steps=2)
    facade = AdapterFacade(
        LiberoRoboRSIAdapter(
            tmp_path / "media", tmp_path / "native", env_factory=lambda *args: native
        )
    )
    wire = json.dumps(
        to_envelope(_task(), benchmark="libero", episode_id="episode-42").as_dict()
    )
    with pytest.raises(ValueError, match="instruction differs"):
        facade.start(wire)
    assert native.closed and native.steps == 0
    assert not list((tmp_path / "media").iterdir())


def test_native_clipping_inputs_are_preserved_and_episode_limit_is_enforced(tmp_path):
    from jsonschema import Draft202012Validator

    received = []

    class Capture(FakeNative):
        def step(self, vector):
            received.append(vector)
            self.steps += 1
            return self._observation(), 0, False, False, {}

    native = Capture(max_steps=1)
    adapter = LiberoRoboRSIAdapter(
        tmp_path / "media", tmp_path / "native", env_factory=lambda *args: native
    )
    vector = [-2.0, 2.0, 0.0, 4.0, -4.0, 0.0, -2.0]
    started = adapter.start(_task(max_steps=1), "episode-42")
    Draft202012Validator(started.capabilities["action_schema"]).validate(
        {"vector": vector}
    )
    assert adapter.step(_action(0, vector)).truncated
    assert received == [vector]
    with pytest.raises(ValueError, match="already ended"):
        adapter.step(_action(1))
    assert native.steps == 1 and adapter.finish().details["steps"] == 1
    adapter.close()


def test_roborsi_rejects_an_unadvertised_robot_before_creating_environment(
    tmp_path, monkeypatch
):
    from robots.libero import env_server

    monkeypatch.setenv("LIBERO_ROBOT_BASE", "AnotherRobot")
    monkeypatch.setattr(
        env_server,
        "make_env",
        lambda *args, **kwargs: pytest.fail("must reject before creation"),
    )
    adapter = LiberoRoboRSIAdapter(tmp_path / "media", tmp_path / "native")
    with pytest.raises(ValueError, match="Panda OSC_POSE"):
        adapter.start(_task(), "episode-42")
