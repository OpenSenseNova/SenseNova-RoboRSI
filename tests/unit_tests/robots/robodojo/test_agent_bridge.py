# Copyright 2026 The RoboRSI Contributors.
# Licensed under the Apache License, Version 2.0.

"""Exercise submitted policy ownership through actual Runner messages and media."""

import io

import numpy as np
import pytest
from PIL import Image

from roborsi.evaluation.media import CaseMediaChannel
from roborsi.evaluation.messages import EpisodeStop, Observation, StepResult, Task


def task():
    return Task(
        benchmark="robodojo",
        task_id="build_tower",
        seed=7,
        instruction="Build the tower",
        max_steps=9,
        native={"config": "arx_x5", "eval_seed": 2},
    )


def observation(channel, step=0, episode="episode"):
    rgb = np.array([[[255, 0, 17], [1, 91, 201]]], dtype=np.uint8)
    stream = io.BytesIO()
    Image.fromarray(rgb).save(stream, format="PNG")
    media = tuple(
        channel.write(episode, step, stream.getvalue(), "image/png") for _ in range(3)
    )
    fields = {
        name: i
        for i, name in enumerate(("cam_head", "cam_left_wrist", "cam_right_wrist"))
    }
    return Observation(
        benchmark="robodojo",
        episode_id=episode,
        step_index=step,
        data={"instruction": "Build the tower", "state": {}, "media_fields": fields},
        media=media,
    ), rgb


class Policy:
    def __init__(self, cfg):
        self.cfg = cfg
        self.observations = []
        self.actions = [{"left_arm_joint_state": np.zeros(6)}]
        self.terminal = None
        self.closed = False

    def reset(self):
        pass

    def update_obs(self, obs):
        self.observations.append(obs)

    def get_action(self):
        return self.actions

    def end_episode(self, obs):
        self.terminal = obs

    def shutdown(self):
        self.closed = True


def agent(tmp_path):
    from robots.robodojo.agents.bridge import SubmissionAgent

    channel = CaseMediaChannel(tmp_path / "media")
    instance = SubmissionAgent(
        tmp_path,
        parameters={"vla_backend": "disabled"},
        media=channel,
        model_factory=Policy,
    )
    instance.reset(task())
    return instance, channel


def test_exact_rgb_and_layout_seed_are_preserved(tmp_path):
    instance, channel = agent(tmp_path)
    obs, rgb = observation(channel)
    result = instance.act(obs)
    np.testing.assert_array_equal(
        instance.model.observations[0]["vision"]["cam_head"]["color"], rgb
    )
    assert instance.model.cfg["seed"] == 2
    assert instance.model.cfg["native_step_limit"] == 9
    assert result.step_index == 0
    assert result.data["left_arm_joint_state"] == [0.0] * 6


def test_stale_or_cross_episode_feedback_is_rejected(tmp_path):
    instance, channel = agent(tmp_path)
    obs, _ = observation(channel)
    instance.act(obs)
    with pytest.raises(ValueError, match="sequence"):
        instance.act(obs)
    wrong, _ = observation(channel, 1, "other-episode")
    with pytest.raises(ValueError, match="episode"):
        instance.act(wrong)


def test_empty_poll_does_not_emit_a_physical_action(tmp_path):
    instance, channel = agent(tmp_path)
    calls = iter([[], [{"left_arm_joint_state": np.zeros(6)}]])
    instance.model.get_action = lambda: next(calls)
    result = instance.act(observation(channel)[0])
    assert result.step_index == 0
    assert len(instance.model.observations) == 1


def test_finish_becomes_stop_and_terminal_feedback_is_delivered(tmp_path):
    instance, channel = agent(tmp_path)
    instance.model.actions = [{"_roborsi_control": "finish"}]
    obs, _ = observation(channel)
    result = instance.act(obs)
    assert isinstance(result, EpisodeStop)
    assert result.step_index == 0
    instance.observe_end(
        StepResult(
            observation=obs, terminated=True, truncated=False, termination_reason="done"
        )
    )
    assert instance.model.terminal is not None
    instance.close()
    instance.close()
    assert instance.model.closed


def test_terminal_feedback_does_not_request_another_action(tmp_path):
    instance, channel = agent(tmp_path)
    instance.act(observation(channel)[0])
    instance.model.get_action = lambda: pytest.fail(
        "no action after native termination"
    )
    instance.observe_end(
        StepResult(
            observation=observation(channel, 1)[0],
            terminated=True,
            truncated=False,
            termination_reason="native_success",
        )
    )
    assert instance.model.terminal is not None


def test_corrupt_media_is_rejected_before_policy(tmp_path):
    instance, channel = agent(tmp_path)
    obs, _ = observation(channel)
    (tmp_path / obs.media[0].ref).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="mismatch"):
        instance.act(obs)
    assert not instance.model.observations


def test_reset_rejects_other_benchmarks_before_starting_policy(tmp_path):
    instance, _ = agent(tmp_path)
    with pytest.raises(ValueError, match="RoboDojo"):
        instance.reset(task().model_copy(update={"benchmark": "other"}))
