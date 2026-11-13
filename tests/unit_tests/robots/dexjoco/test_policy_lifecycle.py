# Copyright 2026 The HEART Authors.
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

"""Exercise policy chunk timing, continuation and score isolation offline."""

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from roborsi.evaluation.messages import EpisodeStop, Observation, StepResult
from robots.dexjoco.agents.baseline.policy import FrozenPolicyAgent
from robots.dexjoco.benchmark_service import make_task

ARCHIVE = (
    Path(__file__).resolve().parents[4]
    / "robots/dexjoco/agents/baseline/resources/reference-code.zip"
)


def agent_fixture(tmp_path, upstream, monkeypatch, *, invalid=False):
    agent = FrozenPolicyAgent(tmp_path, ARCHIVE, tmp_path / "assets")
    # This fixture focuses on lifecycle; real hashes are exercised separately.
    agent.task = make_task(upstream, "water_plant", 100000)
    agent.dual = False
    agent.standard_limit = 2
    agent.task = agent.task.model_copy(update={"max_steps": 20})
    monkeypatch.setattr(agent, "_verify_payload", lambda: None)
    calls = []

    class Session:
        def __init__(self, policy, phase, prompt):
            self.phase = phase
            calls.append(("start", phase))

        def actions(self, decision, remaining):
            calls.append(("actions", self.phase, decision, remaining))
            if invalid:
                raise agent.ref.PolicyError("invalid envelope")
            return [[0.0] * 22] * 2, {}

        def close(self):
            calls.append(("close", self.phase))
            return {"phase": self.phase}

    agent.factory = Session
    return agent, calls


def observation(agent, index, episode="one"):
    output = io.BytesIO()
    Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(output, "PNG")
    media = tuple(
        agent.media.write(episode, index, output.getvalue(), "image/png")
        for _ in range(2)
    )
    return Observation(
        benchmark="dexjoco",
        episode_id=episode,
        step_index=index,
        data={
            "state": [0.0] * 23,
            "prompt": "Task",
            "camera_names": ["random_camera", "wrist"],
            "seed": 100000,
            "success": True,
            "remaining_steps": 99999,
        },
        media=media,
    )


def test_chunk_feedback_and_new_continuation_session(tmp_path, upstream, monkeypatch):
    agent, calls = agent_fixture(tmp_path, upstream, monkeypatch)
    agent.act(observation(agent, 0))
    assert calls == [("start", "initial"), ("actions", "initial", 0, 2)]
    public = json.loads((agent.inputs / "observation.json").read_text())
    assert public["remaining_steps"] == 2
    assert "seed" not in public and "success" not in public
    agent.act(observation(agent, 1))
    assert len(calls) == 2
    assert json.loads((agent.inputs / "observation.json").read_text())["step"] == 0
    agent.act(observation(agent, 2))
    assert calls[2:] == [
        ("close", "initial"),
        ("start", "continuation"),
        ("actions", "continuation", 1, 18),
    ]
    history = (agent.inputs / "first_session_history.jsonl").read_text()
    assert len(history.splitlines()) == 3
    assert (
        "success" not in history and "reward" not in history and "seed" not in history
    )
    agent.close()
    agent.close()
    assert calls.count(("close", "continuation")) == 1


def test_natural_terminal_does_not_open_continuation(tmp_path, upstream, monkeypatch):
    agent, calls = agent_fixture(tmp_path, upstream, monkeypatch)
    agent.act(observation(agent, 0))
    agent.observe_end(
        StepResult(
            observation=observation(agent, 1),
            terminated=True,
            truncated=False,
            termination_reason="native_failure",
            reward=100,
        )
    )
    assert calls == [
        ("start", "initial"),
        ("actions", "initial", 0, 2),
        ("close", "initial"),
    ]
    assert json.loads((agent.inputs / "episode_done.json").read_text()) == {
        "episode_done": True
    }


def test_invalid_policy_is_scored_stop_not_favorable_retry(
    tmp_path, upstream, monkeypatch
):
    agent, calls = agent_fixture(tmp_path, upstream, monkeypatch, invalid=True)
    stop = agent.act(observation(agent, 0))
    assert isinstance(stop, EpisodeStop)
    assert stop.step_index == 0
    agent.observe_end(
        StepResult(
            observation=Observation(
                benchmark="dexjoco", episode_id="one", step_index=0
            ),
            terminated=False,
            truncated=True,
            termination_reason="policy_failure",
        )
    )
    agent.close()
    assert all("continuation" not in call for call in calls)


def test_foreign_or_stale_observation_rejected(tmp_path, upstream, monkeypatch):
    agent, _ = agent_fixture(tmp_path, upstream, monkeypatch)
    agent.act(observation(agent, 0))
    with pytest.raises(ValueError, match="stale or foreign"):
        agent.act(observation(agent, 1, "another"))
    agent.close()
