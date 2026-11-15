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

"""A zero-step native stop has identity, replay and terminal-feedback contracts."""

import pytest

from roborsi.evaluation.adapter_service import AdapterFacade
from roborsi.evaluation.messages import (
    Action,
    EpisodeStop,
    NativeResult,
    Observation,
    StartResult,
    StepResult,
    Task,
)
from roborsi.evaluation.service_protocol import (
    RoboRSIBackendError,
    RoboRSIRequestError,
    _decode,
    _encode,
)
from roborsi.evaluation.services import AdapterClient, AgentFacade
from tests.unit_tests.roborsi.evaluation.test_services import _service


def wire(message, request_id):
    return _encode(message, "demo", "episode", request_id)


class Adapter:
    def __init__(self, invalid=None):
        self.stops = self.closes = 0
        self.invalid = invalid

    def start(self, task, episode_id):
        self.task = task
        self.observation = Observation(
            benchmark="demo", episode_id=episode_id, step_index=0
        )
        return StartResult(task=task, observation=self.observation)

    def stop(self, request):
        self.stops += 1
        return StepResult(
            observation=self.observation.model_copy(
                update={"step_index": 1 if self.invalid == "step" else 0}
            ),
            terminated=self.invalid != "terminal",
            truncated=False,
            termination_reason=request.reason if self.invalid != "terminal" else None,
        )

    def finish(self):
        return NativeResult(
            task=self.task, episode_id="episode", success=False, score=0.25
        )

    def close(self):
        self.closes += 1


def task():
    return Task(
        benchmark="demo", task_id="place", instruction="Place", seed=0, max_steps=2
    )


def stop(step=0):
    return EpisodeStop(
        benchmark="demo", episode_id="episode", step_index=step, reason="give_up"
    )


def test_stop_replay_allows_finish_without_advancing_or_repeating_work():
    adapter = Adapter()
    facade = AdapterFacade(adapter)
    facade.start(wire(task(), "start"))
    stopped = facade.stop(wire(stop(), "stop"))
    assert _decode(stopped, StepResult).observation.step_index == 0
    assert facade.stop(wire(stop(), "stop")) == stopped
    with pytest.raises(RoboRSIRequestError, match="reused"):
        facade.stop(wire(stop().model_copy(update={"reason": "done"}), "stop"))
    with pytest.raises(RoboRSIRequestError, match="ended"):
        facade.step(
            wire(
                Action(benchmark="demo", episode_id="episode", step_index=0, data={}),
                "step",
            )
        )
    result = facade.finish("finish")
    assert _decode(result, NativeResult).score == 0.25
    assert facade.finish("finish") == result
    assert facade.stop(wire(stop(), "stop")) == stopped
    assert adapter.stops == adapter.closes == 1


@pytest.mark.parametrize("invalid", ["step", "terminal"])
def test_invalid_stop_result_closes_and_replays_error_without_retrying_native(invalid):
    adapter = Adapter(invalid)
    facade = AdapterFacade(adapter)
    facade.start(wire(task(), "start"))
    for _ in range(2):
        with pytest.raises(RoboRSIBackendError, match="must"):
            facade.stop(wire(stop(), "stop"))
    assert adapter.stops == adapter.closes == 1
    with pytest.raises(RoboRSIRequestError, match="must end"):
        facade.finish("finish")


def test_stale_stop_does_not_reach_native_owner():
    adapter = Adapter()
    facade = AdapterFacade(adapter)
    facade.start(wire(task(), "start"))
    with pytest.raises(RoboRSIRequestError, match="out of order"):
        facade.stop(wire(stop(1), "stale"))
    assert adapter.stops == adapter.closes == 0
    facade.stop(wire(stop(), "valid"))
    facade.finish("finish")


def test_zero_step_agent_stop_receives_terminal_feedback_once():
    class Agent:
        calls = 0
        feedback = []

        def reset(self, task):
            pass

        def act(self, observation):
            self.calls += 1
            return stop()

        def observe_end(self, step):
            self.feedback.append(step)

        def close(self):
            pass

    agent = Agent()
    facade = AgentFacade(agent)
    facade.reset(wire(task(), "reset"))
    observation = Observation(benchmark="demo", episode_id="episode", step_index=0)
    decision = facade.act(wire(observation, "act"))
    assert _decode(decision, EpisodeStop) == stop()
    assert facade.act(wire(observation, "act")) == decision
    with pytest.raises(RoboRSIRequestError, match="ended"):
        facade.act(wire(observation, "next-act"))
    ended = StepResult(
        observation=observation,
        terminated=True,
        truncated=False,
        termination_reason="give_up",
    )
    assert facade.observe_end(wire(ended, "end")) == facade.observe_end(
        wire(ended, "end")
    )
    assert agent.calls == len(agent.feedback) == 1
    facade.close()


def test_legacy_adapter_rejects_stop_without_losing_its_action_session():
    with _service("adapter") as rpc:
        client = AdapterClient(rpc)
        client.start(task(), "episode", timeout_s=3)
        with pytest.raises(RoboRSIRequestError, match="does not advertise"):
            client.stop(stop(), timeout_s=3)
        ended = client.step(
            Action(
                benchmark="demo",
                episode_id="episode",
                step_index=0,
                data={"native": {"__ndarray__": "literal-field"}},
            ),
            timeout_s=3,
        )
        assert ended.terminated and ended.observation.step_index == 1
        assert client.finish(timeout_s=3).score == 0.75
        client.close()
