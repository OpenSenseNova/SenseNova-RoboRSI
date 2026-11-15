# Copyright 2026 The SenseNova-RoboRSI Authors.
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

"""Episode initialization, shared dependency injection and final tool selection."""

import json

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel

from roborsi.embodied_agent import EmbodiedAgent, LocalToolSpec
from roborsi.llm import LLMConfig
from roborsi.runtime import RuntimeConfig, SubAgentConfig
from roborsi.runtime.context_engine import InitialContext
from roborsi.tools.toolkit import ToolResult


@pytest.mark.parametrize("invalid", ["unanswered", "orphan", "duplicate", "wrong_name"])
def test_invalid_initial_exchanges_fail_before_model_and_release_episode(
    tmp_path, invalid
):
    class Engine:
        invalid = True

        def initialize(self, inputs, episode_ctx, history):
            messages = [ModelRequest(parts=[UserPromptPart("new task")])]
            if self.invalid:
                call = ModelResponse(parts=[ToolCallPart("move", {}, "old-call")])
                result = ModelRequest(
                    parts=[ToolReturnPart("move", "old result", "old-call")]
                )
                messages.extend(
                    {
                        "unanswered": [call],
                        "orphan": [result],
                        "duplicate": [call, call, result],
                        "wrong_name": [
                            call,
                            ModelRequest(
                                parts=[ToolReturnPart("other", "result", "old-call")]
                            ),
                        ],
                    }[invalid]
                )
            return InitialContext("system", messages)

        def __call__(self, ctx, messages):
            return messages

    calls = []

    def model(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[ToolCallPart("move", {})])

    engine = Engine()
    agent = EmbodiedAgent(
        mcp_servers=[],
        output_dir=tmp_path,
        local_tools=[
            LocalToolSpec(
                "move",
                "move",
                {"type": "object"},
                handler=lambda _: ToolResult("move", {"_finish": True}),
            )
        ],
        llm=LLMConfig("openai", "offline", model_factory=lambda: FunctionModel(model)),
        runtime=RuntimeConfig(context_engine=engine),
    )
    with pytest.raises(ValueError, match="context contains"):
        agent.run("task", system_prompt="system")
    assert not calls
    engine.invalid = False
    assert agent.run("task", system_prompt="system").error is None
    assert len(calls) == 1


@pytest.mark.parametrize("excluded", [False, True])
def test_unadvertised_tool_preserves_execution_selection_and_request_trace(
    tmp_path, excluded
):
    called = []
    requests = []

    def model(messages, info):
        requests.append([tool.name for tool in info.function_tools])
        return ModelResponse(
            parts=[ToolCallPart("hidden" if len(requests) == 1 else "visible", {})]
        )

    def handler(name):
        def execute(arguments):
            called.append(name)
            return ToolResult(name, {"_finish": True})

        return execute

    result = EmbodiedAgent(
        mcp_servers=[],
        output_dir=tmp_path,
        local_tools=[
            LocalToolSpec(
                name,
                name,
                {"type": "object"},
                handler=handler(name),
                advertise=name != "hidden",
            )
            for name in ("hidden", "visible")
        ],
        llm=LLMConfig("openai", "offline", model_factory=lambda: FunctionModel(model)),
        runtime=RuntimeConfig(
            tool_names=("visible",) if excluded else ("hidden", "visible")
        ),
    ).run("task", system_prompt="system")
    assert result.error is None
    assert called == (["visible"] if excluded else ["hidden"])
    assert requests == [["visible"]] * (2 if excluded else 1)

    events = [
        json.loads(line)
        for line in (tmp_path / "trace/events.jsonl").read_text().splitlines()
    ]
    recorded_requests = [
        event for event in events if event["type"] == "model_request_start"
    ]
    assert len(recorded_requests) == len(requests)
    for event in recorded_requests:
        request = json.loads((tmp_path / event["payload"]["request_ref"]).read_text())
        assert [
            tool["name"]
            for tool in request["model_request_parameters"]["function_tools"]
        ] == ["visible"]


@pytest.mark.parametrize("value", [None, 0, "false"])
def test_local_tool_visibility_requires_boolean(value):
    with pytest.raises(TypeError, match="advertise must be a boolean"):
        LocalToolSpec("hidden", "hidden", {"type": "object"}, advertise=value)


def test_unadvertised_tool_requires_api_planner(tmp_path):
    agent = EmbodiedAgent(
        planner="claude_code",
        mcp_servers=[],
        output_dir=tmp_path,
        local_tools=[
            LocalToolSpec("hidden", "hidden", {"type": "object"}, advertise=False)
        ],
    )
    with pytest.raises(
        ValueError, match="unadvertised local tools require the api planner"
    ):
        agent.run("task", system_prompt="system")


@pytest.mark.parametrize("mode", ["default", "policy", "callback", "factory", "yaml"])
def test_context_configuration_preserves_model_tools(tmp_path, mode):
    """Changing history policy alone keeps the actual request schemas intact."""

    def engine(ctx, messages):
        return messages

    if mode == "yaml":
        path = tmp_path / "runtime.yaml"
        path.write_text("context:\n  strategy: recent_turns\n  keep_turns: 2\n")
        runtime = RuntimeConfig.from_file(path)
    else:
        runtime = RuntimeConfig(
            **{
                "default": {},
                "policy": {"context": {"strategy": "recent_turns", "keep_turns": 2}},
                "callback": {"context_engine": engine},
                "factory": {"context_engine_factory": lambda: engine},
            }[mode]
        )

    requests = []

    def model(messages, info):
        requests.append(
            [(tool.name, tool.parameters_json_schema) for tool in info.function_tools]
        )
        return ModelResponse(parts=[ToolCallPart("move", {"height": 0.8})])

    for name, config in [("baseline", None), (mode, runtime)]:
        result = EmbodiedAgent(
            mcp_servers=[],
            output_dir=tmp_path / name,
            local_tools=[
                LocalToolSpec(
                    "move",
                    "Move the end effector",
                    {
                        "type": "object",
                        "properties": {"height": {"type": "number"}},
                        "required": ["height"],
                    },
                    handler=lambda _: ToolResult("move", {"_finish": True}),
                )
            ],
            llm=LLMConfig(
                "openai", "offline", model_factory=lambda: FunctionModel(model)
            ),
            runtime=config,
        ).run("task", system_prompt="system")
        assert result.error is None
    assert requests[0] == requests[1]
    assert [name for name, _ in requests[1]] == ["move", "finish"]


def test_initialize_once_across_sdk_repair_runs_with_episode_ctx(tmp_path, monkeypatch):
    initialized = []
    factory_calls = []
    requests = []
    contexts = []
    ctx = {"recipe": "recipe text", "watch": "watch frame", "state": []}

    class Engine:
        def initialize(self, inputs, episode_ctx, history):
            initialized.append(episode_ctx)
            assert history == []
            return InitialContext(
                "custom system",
                [
                    ModelRequest(
                        parts=[
                            UserPromptPart(inputs.query + "\n" + episode_ctx["recipe"])
                        ]
                    ),
                    ModelRequest(parts=[UserPromptPart(episode_ctx["watch"])]),
                    ModelRequest(parts=[UserPromptPart("live observation")]),
                ],
            )

        def __call__(self, run_ctx, messages, *, episode_ctx):
            assert run_ctx.model is not None
            assert episode_ctx is ctx
            contexts.append(episode_ctx)
            return messages

    def factory():
        factory_calls.append(True)
        return Engine()

    def model(messages, info):
        assert info.instructions == "custom system"
        requests.append(messages.copy())
        if len(requests) == 1:
            return ModelResponse(parts=[TextPart("thinking")])
        return ModelResponse(parts=[ToolCallPart("move", {})])

    monkeypatch.setattr(LLMConfig, "build_model", lambda _: FunctionModel(model))
    agent = EmbodiedAgent(
        mcp_servers=[],
        output_dir=tmp_path,
        local_tools=[LocalToolSpec("move", "move", {"type": "object"})],
        llm=LLMConfig("openai", "offline", api_key="test"),
        runtime=RuntimeConfig(context_engine_factory=factory, dynamic_subagents=False),
    )
    with agent.start_episode(
        "task", system_prompt="unused", deferred_tools=["move"], episode_ctx=ctx
    ) as episode:
        call = episode.next_call(timeout=10)
        assert call is not None
        episode.complete(call, ToolResult("move", {"_finish": True}))
        result = episode.wait(timeout=10)
    assert result.error is None
    assert factory_calls == [True]
    assert initialized == [ctx]
    assert len(contexts) == 2
    assert [
        p.content for m in requests[0] for p in m.parts if isinstance(p, UserPromptPart)
    ] == ["task\nrecipe text", "watch frame", "live observation"]
    assert (
        sum(
            p.content == "watch frame"
            for m in requests[1]
            for p in m.parts
            if isinstance(p, UserPromptPart)
        )
        == 1
    )


def test_tool_selection_covers_static_delegates_skills_and_finish(
    tmp_path, monkeypatch
):
    skill = tmp_path / "skills" / "helper"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: helper\ndescription: helper\n---\ncontent"
    )
    selected = []

    def model(messages, info):
        selected.append([tool.name for tool in info.function_tools])
        return ModelResponse(parts=[ToolCallPart("move", {})])

    monkeypatch.setattr(LLMConfig, "build_model", lambda _: FunctionModel(model))
    result = EmbodiedAgent(
        mcp_servers=[],
        output_dir=tmp_path / "run",
        local_tools=[
            LocalToolSpec(
                "move",
                "move",
                {"type": "object"},
                handler=lambda _: ToolResult("move", {"_finish": True}),
            )
        ],
        llm=LLMConfig("openai", "offline", api_key="test"),
        runtime=RuntimeConfig(
            dynamic_subagents=False,
            subagents={"helper": SubAgentConfig(instructions="help")},
            skill_paths=(skill,),
            tool_names=("move",),
        ),
    ).run("task", system_prompt="system")
    assert result.error is None
    assert selected == [["move"]]


def test_embodied_factory_keeps_child_processors_isolated(tmp_path, monkeypatch):
    instances = []

    class Engine:
        def __init__(self):
            self.calls = 0
            instances.append(self)

        def __call__(self, run_ctx, messages):
            self.calls += 1
            return messages

    def model(messages, info):
        if info.instructions == "CHILD":
            return ModelResponse(parts=[TextPart("child result")])
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "delegate_task", {"agent_name": "child", "task": "help"}
                    )
                ]
            )
        return ModelResponse(
            parts=[ToolCallPart("finish", {"status": "success", "summary": "done"})]
        )

    monkeypatch.setattr(LLMConfig, "build_model", lambda _: FunctionModel(model))
    result = EmbodiedAgent(
        mcp_servers=[],
        output_dir=tmp_path,
        local_tools=[
            LocalToolSpec(
                "read",
                "read",
                {"type": "object"},
                handler=lambda _: ToolResult("read", {}),
            )
        ],
        llm=LLMConfig("openai", "offline", api_key="test"),
        runtime=RuntimeConfig(
            context_engine_factory=Engine,
            dynamic_subagents=True,
            subagents={"child": SubAgentConfig(instructions="CHILD")},
        ),
    ).run("task", system_prompt="ROOT")
    assert result.error is None
    assert len(instances) == 2
    assert [engine.calls for engine in instances] == [2, 1]
