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

"""An EmbodiedAgent dialogue proposes RoboRSI actions across observations."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from roborsi.evaluation.embodied_bridge import (
    _EmbodiedToolConnector,
    roborsi_action_tool,
)
from roborsi.evaluation.media import CaseMediaChannel
from roborsi.evaluation.messages import Observation, StepResult, Task

_SCHEMA = {
    "type": "object",
    "properties": {
        "vector": {"type": "array", "items": {"type": "number"}, "minItems": 7}
    },
    "required": ["vector"],
    "additionalProperties": False,
}


def test_heart_agent_compatibility_entrypoint_is_removed() -> None:
    import roborsi.evaluation.embodied_bridge as bridge

    with pytest.raises(AttributeError, match="EmbodiedHeartAgent"):
        getattr(bridge, "EmbodiedHeartAgent")


def _task() -> Task:
    return Task(
        benchmark="libero",
        task_id="3",
        seed=7,
        instruction="Place the block.",
        max_steps=2,
        capabilities={"action_schema": _SCHEMA},
    )


def _observation(channel: CaseMediaChannel, step: int) -> Observation:
    output = io.BytesIO()
    Image.new("RGB", (2, 2), (step + 1, 0, 0)).save(output, format="PNG")
    reference = channel.write("episode-42", step, output.getvalue(), "image/png")
    return Observation(
        benchmark="libero",
        episode_id="episode-42",
        step_index=step,
        data={"states": [float(step)], "media_fields": {"main_images": 0}},
        media=(reference,),
    )


def test_embodied_dialogue_receives_observation_and_proposes_two_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart
    from pydantic_ai.models.function import FunctionModel

    from roborsi.embodied_agent import EmbodiedAgent
    from roborsi.llm import LLMConfig

    turns: list[int] = []

    def model(messages: list[Any], info: Any) -> ModelResponse:
        assert {tool.name for tool in info.function_tools} == {"roborsi_action"}
        returned = sum(
            isinstance(part, ToolReturnPart)
            for message in messages
            for part in message.parts
        )
        turns.append(returned)
        return ModelResponse(
            parts=[ToolCallPart("roborsi_action", {"vector": [float(returned)] * 7})]
        )

    monkeypatch.setattr(LLMConfig, "build_model", lambda self: FunctionModel(model))
    planner = EmbodiedAgent(
        mcp_servers=[],
        local_tools=[roborsi_action_tool(_SCHEMA)],
        output_dir=tmp_path / "agent",
        llm=LLMConfig("openai", "offline", api_key="test"),
        max_turns=3,
    )
    channel = CaseMediaChannel(tmp_path / "media")
    agent = _EmbodiedToolConnector(
        planner,
        media_dir=channel.media_dir,
        system_prompt="Use roborsi_action for each observation.",
        output_dir=tmp_path / "agent",
        action_timeout_s=10.0,
    )
    agent.reset(_task())
    try:
        assert agent.act(_observation(channel, 0)).data == {"vector": [0.0] * 7}
        assert agent.act(_observation(channel, 1)).data == {"vector": [1.0] * 7}
        assert turns == [0, 1]
    finally:
        agent.close()
        agent.close()


def test_agent_rejects_wrong_schema_and_out_of_order_observation(
    tmp_path: Path,
) -> None:
    planner = SimpleNamespace(
        local_tools=[
            SimpleNamespace(name="roborsi_action", input_schema=_SCHEMA, handler=None)
        ]
    )
    agent = _EmbodiedToolConnector(
        planner,
        media_dir=tmp_path / "media",
        system_prompt="Use the tool.",
        output_dir=tmp_path / "agent",
    )
    task = _task().model_copy(
        update={"capabilities": {"action_schema": {"type": "object"}}}
    )
    with pytest.raises(ValueError, match="matching deferred action tool"):
        agent.reset(task)
    agent.reset(_task())
    with pytest.raises(ValueError, match="out of order"):
        agent.act(
            Observation(benchmark="libero", episode_id="episode-42", step_index=1)
        )
    agent.close()


def test_bridge_returns_feedback_before_requesting_next_action(tmp_path: Path) -> None:
    class FakeEpisode:
        def __init__(self) -> None:
            self.calls = iter(
                [
                    SimpleNamespace(
                        name="roborsi_action", arguments={"vector": [0.0] * 7}
                    ),
                    SimpleNamespace(
                        name="roborsi_action", arguments={"vector": [1.0] * 7}
                    ),
                ]
            )
            self.replies = []
            self.closed = False

        def next_call(self, *, timeout: float):
            return next(self.calls)

        def complete(self, call, reply) -> None:
            self.replies.append((call, reply))

        def close(self) -> None:
            self.closed = True

    episode = FakeEpisode()

    class FakePlanner:
        local_tools = [
            SimpleNamespace(name="roborsi_action", input_schema=_SCHEMA, handler=None)
        ]

        def start_episode(self, *args, **kwargs):
            assert args == ("Place the block.",)
            assert kwargs["initial_context"][0].find('"step_index": 0') >= 0
            return episode

    constructed_for = []

    def planner_factory(task: Task) -> FakePlanner:
        constructed_for.append(task)
        return FakePlanner()

    agent = _EmbodiedToolConnector(
        planner_factory,
        media_dir=tmp_path / "media",
        system_prompt="Use roborsi_action.",
        output_dir=tmp_path / "agent",
    )
    agent.reset(_task())
    assert constructed_for == [_task()]
    first = Observation(benchmark="libero", episode_id="episode-42", step_index=0)
    second = Observation(benchmark="libero", episode_id="episode-42", step_index=1)
    assert agent.act(first).data == {"vector": [0.0] * 7}
    assert agent.act(second).data == {"vector": [1.0] * 7}
    assert episode.replies[0][1].result["step_index"] == 1
    agent.close()
    assert episode.replies[1][1].is_finish
    assert episode.closed


def _scripted_connector(tmp_path: Path, arguments, *, schema=None, **options):
    class Episode:
        def __init__(self) -> None:
            self.calls = iter(
                SimpleNamespace(name="roborsi_action", arguments=a) for a in arguments
            )
            self.replies = []

        def next_call(self, *, timeout: float):
            return next(self.calls)

        def complete(self, call, reply) -> None:
            self.replies.append(reply)

        def close(self) -> None:
            pass

    episode = Episode()

    class Planner:
        local_tools = [
            SimpleNamespace(
                name="roborsi_action", input_schema=schema or _SCHEMA, handler=None
            )
        ]

        def start_episode(self, *args, **kwargs):
            return episode

    agent = _EmbodiedToolConnector(
        Planner(),
        media_dir=tmp_path / "media",
        system_prompt="Use roborsi_action.",
        output_dir=tmp_path / "agent",
        **options,
    )
    task = _task()
    if schema is not None:
        task = task.model_copy(update={"capabilities": {"action_schema": schema}})
    agent.reset(task)
    return agent, episode


def test_invalid_action_is_returned_to_the_model_when_enabled(tmp_path: Path) -> None:
    bad, good = {"vector": ["0"] * 7}, {"vector": [0.0] * 7}
    agent, episode = _scripted_connector(
        tmp_path, [bad, good], invalid_action_retries=2
    )
    first = Observation(benchmark="libero", episode_id="episode-42", step_index=0)
    action = agent.act(first)
    assert action.data == good and action.step_index == 0
    (reply,) = episode.replies
    assert reply.result["error"] == "invalid_action"
    assert reply.result["step_index"] == 0
    assert "not executed" in reply.result["message"]


def test_json_text_values_are_decoded_when_retries_are_enabled(tmp_path: Path) -> None:
    agent, episode = _scripted_connector(
        tmp_path, [{"vector": "[0, 0, 0, 0, 0, 0, 0]"}], invalid_action_retries=1
    )
    first = Observation(benchmark="libero", episode_id="episode-42", step_index=0)
    assert agent.act(first).data == {"vector": [0] * 7}
    assert episode.replies == []


def test_invalid_action_reason_names_the_field_of_the_chosen_primitive(
    tmp_path: Path,
) -> None:
    schema = {
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "primitive": {"const": "move"},
                    "xy": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["primitive", "xy"],
            },
            {
                "type": "object",
                "properties": {"primitive": {"const": "look"}},
                "required": ["primitive"],
            },
        ]
    }
    from jsonschema import Draft202012Validator

    from roborsi.evaluation.embodied_bridge import _reason

    errors = list(
        Draft202012Validator(schema).iter_errors({"primitive": "move", "xy": "left"})
    )
    assert _reason(errors).startswith("xy: 'left' is not of type 'array'")


def test_reply_names_only_fields_that_decoding_cannot_fix(tmp_path: Path) -> None:
    schema = {
        "type": "object",
        "properties": {
            "pitch": {"type": "number"},
            "gripper": {"enum": [-1, 0, 1]},
        },
        "required": ["pitch", "gripper"],
    }
    bad = {"pitch": "0.5", "gripper": "hold"}
    good = {"pitch": "0.5", "gripper": "0"}
    agent, episode = _scripted_connector(
        tmp_path, [bad, good], invalid_action_retries=1, schema=schema
    )
    first = Observation(benchmark="libero", episode_id="episode-42", step_index=0)
    assert agent.act(first).data == {"pitch": 0.5, "gripper": 0}
    (reply,) = episode.replies
    assert "gripper" in reply.result["message"]
    assert "pitch" not in reply.result["message"]


def test_invalid_action_ends_the_case_by_default_or_after_retries(
    tmp_path: Path,
) -> None:
    from jsonschema import ValidationError

    bad = {"vector": ["0"] * 7}
    first = Observation(benchmark="libero", episode_id="episode-42", step_index=0)
    agent, episode = _scripted_connector(tmp_path / "default", [bad])
    with pytest.raises(ValidationError):
        agent.act(first)
    assert episode.replies == []
    agent, episode = _scripted_connector(
        tmp_path / "retries", [bad, bad, bad], invalid_action_retries=2
    )
    with pytest.raises(ValidationError):
        agent.act(first)
    assert len(episode.replies) == 2
    with pytest.raises(ValueError, match="invalid_action_retries"):
        _scripted_connector(tmp_path / "x", [], invalid_action_retries=-1)


@pytest.mark.parametrize("truncated", [False, True])
def test_embodied_agent_runs_local_tools_and_receives_terminal_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, truncated: bool
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart
    from pydantic_ai.models.function import FunctionModel

    from roborsi import embodied_agent
    from roborsi.embodied_agent import (
        EmbodiedAgent,
        EmbodiedEpisode,
        LocalToolSpec,
    )
    from roborsi.llm import LLMConfig
    from roborsi.tools.toolkit import ToolResult

    calls = []
    local_calls = []
    feedbacks = []
    complete = EmbodiedEpisode.complete

    def capture_feedback(self, call, result):
        feedbacks.append(result)
        complete(self, call, result)

    monkeypatch.setattr(EmbodiedEpisode, "complete", capture_feedback)
    cleanup_entered, cleanup_release = Event(), Event()
    close_resources = embodied_agent.close_toolkit_trace

    def delayed_cleanup(*args, **kwargs):
        cleanup_entered.set()
        assert cleanup_release.wait(10)
        close_resources(*args, **kwargs)

    monkeypatch.setattr(embodied_agent, "close_toolkit_trace", delayed_cleanup)

    def model(messages, info):
        returns = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        calls.append(len(returns))
        if not returns:
            return ModelResponse(parts=[ToolCallPart("plan", {})])
        assert returns[0].tool_name == "plan"
        return ModelResponse(
            parts=[ToolCallPart("roborsi_action", {"vector": [0.0] * 7})]
        )

    def plan(arguments):
        local_calls.append(arguments)
        return ToolResult("plan", {"target": "block"})

    monkeypatch.setattr(LLMConfig, "build_model", lambda self: FunctionModel(model))
    media = CaseMediaChannel(tmp_path / "media")
    monkeypatch.setenv("ROBORSI_MEDIA_DIR", str(media.media_dir))
    agent = EmbodiedAgent(
        mcp_servers=[],
        output_dir=tmp_path / "agent",
        llm=LLMConfig("openai", "offline", api_key="test"),
        local_tools=[LocalToolSpec("plan", "Plan locally", {"type": "object"}, plan)],
        max_turns=4,
        action_timeout_s=10,
    )
    agent.reset(_task())
    try:
        assert agent.act(_observation(media, 0)).step_index == 0
        assert local_calls == [{}]
        terminal = StepResult(
            observation=_observation(media, 1),
            terminated=not truncated,
            truncated=truncated,
            termination_reason="max_steps" if truncated else "native_success",
            reward=0.25,
            native={"actions": 1},
        )
        with ThreadPoolExecutor(max_workers=1) as pool:
            feedback = pool.submit(agent.observe_end, terminal)
            try:
                assert cleanup_entered.wait(5)
                assert not feedback.done(), (
                    "terminal feedback acknowledged before cleanup"
                )
            finally:
                cleanup_release.set()
            feedback.result(timeout=10)
        episode = agent._evaluation_connector._episode
        result = episode.wait(timeout=0)
        assert calls == [0, 1]
        assert result.finish_result["observation"] == terminal.observation.data
        assert result.finish_result["termination_reason"] == terminal.termination_reason
        assert result.finish_result["native"] == {"actions": 1}
        assert result.finish_result["reward"] == 0.25
        assert result.finish_result["media"] == [{"index": 0, "mime_type": "image/png"}]
        assert len(feedbacks) == 1
        image = feedbacks[0].content_blocks[1]["source"]
        assert base64.b64decode(image["data"]) == media.read(
            terminal.observation.media[0], terminal.observation.episode_id, 1
        )
        with pytest.raises(ValueError, match="active task"):
            agent.act(_observation(media, 2))
    finally:
        agent.close()
    assert not episode._worker.is_alive()


@pytest.mark.parametrize("provider_failed", [True, False])
def test_public_agent_reports_provider_failure_separately_from_missing_tools(
    tmp_path, monkeypatch, provider_failed
):
    from pydantic_ai.exceptions import ModelHTTPError
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel

    from roborsi.embodied_agent import EmbodiedAgent
    from roborsi.llm import LLMConfig

    def model(messages, info):
        if provider_failed:
            raise ModelHTTPError(
                400, "offline", {"message": "Exceeded maximum number of images (50)"}
            )
        return ModelResponse(parts=[TextPart("Task complete.")])

    monkeypatch.setattr(LLMConfig, "build_model", lambda self: FunctionModel(model))
    media = CaseMediaChannel(tmp_path / "media")
    monkeypatch.setenv("ROBORSI_MEDIA_DIR", str(media.media_dir))
    agent = EmbodiedAgent(
        mcp_servers=[],
        output_dir=tmp_path / "agent",
        llm=LLMConfig("openai", "offline", api_key="test"),
        action_timeout_s=10,
    )
    agent.reset(_task())
    try:
        expected = "ModelHTTPError.*400.*Exceeded maximum number of images"
        if not provider_failed:
            expected = "model kept failing to produce a tool call"
        with pytest.raises(RuntimeError, match=expected):
            agent.act(_observation(media, 0))
    finally:
        agent.close()
    assert not agent._evaluation_connector._episode._worker.is_alive()


def test_close_retries_episode_cleanup_after_timeout(tmp_path):
    class Episode:
        attempts = 0

        def close(self):
            self.attempts += 1
            if self.attempts == 1:
                raise TimeoutError("cleanup is still running")

    agent = _EmbodiedToolConnector(
        lambda task: None,
        media_dir=tmp_path / "media",
        system_prompt="Use roborsi_action.",
        output_dir=tmp_path / "agent",
    )
    episode = Episode()
    agent._episode = episode
    with pytest.raises(TimeoutError, match="cleanup"):
        agent.close()
    agent.close()
    assert episode.attempts == 2
    assert agent._closed


def test_terminal_feedback_reports_planner_failure(tmp_path: Path) -> None:
    from roborsi.planner.base import PlannerResult

    call = SimpleNamespace(name="roborsi_action", arguments={"vector": [0.0] * 7})
    completed = []
    episode = SimpleNamespace(
        next_call=lambda **kwargs: call,
        complete=lambda pending, feedback: completed.append(feedback),
        wait=lambda **kwargs: PlannerResult(
            error="SDK terminal feedback was not acknowledged",
            error_kind="infrastructure",
            stop_reason="terminal_feedback_missing",
        ),
        close=lambda: None,
    )
    planner = SimpleNamespace(
        local_tools=[roborsi_action_tool(_SCHEMA)],
        start_episode=lambda *args, **kwargs: episode,
    )
    agent = _EmbodiedToolConnector(
        planner,
        media_dir=tmp_path / "media",
        system_prompt="Use roborsi_action.",
        output_dir=tmp_path / "agent",
    )
    agent.reset(_task())
    try:
        agent.act(
            Observation(benchmark="libero", episode_id="episode-42", step_index=0)
        )
        terminal = StepResult(
            observation=Observation(
                benchmark="libero", episode_id="episode-42", step_index=1
            ),
            terminated=True,
            truncated=False,
            termination_reason="native_success",
        )
        with pytest.raises(RuntimeError, match="Embodied Agent failed: SDK terminal"):
            agent.observe_end(terminal)
        assert completed[0].is_finish
    finally:
        agent.close()
