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

"""Behavioral contracts for dynamically defined delegate tasks."""

from __future__ import annotations

import asyncio

import pytest
from pydantic_ai import Tool, models
from pydantic_ai.messages import (
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel


@pytest.fixture(autouse=True)
def offline_models(monkeypatch):
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)


def _prompts(messages):
    return [
        part.content
        for message in messages
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ]


def _returns(messages):
    return [
        part
        for message in messages
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]


def _build(model, config, tools=()):
    from roborsi.runtime.factory import build_runtime_agent

    return build_runtime_agent(
        model=model,
        system_prompt="ROOT private instructions",
        tools=list(tools),
        max_tokens=321,
        capabilities=[],
        runtime=config,
    )


def test_delegate_task_creates_unconfigured_child_with_isolated_context():
    """An unseen name is materialized from the tool call, without parent history."""

    from roborsi.runtime import RuntimeConfig

    child_calls = []

    def model(messages, info):
        if info.instructions.startswith("DYNAMIC CHILD"):
            child_calls.append((_prompts(messages), info.instructions))
            assert _prompts(messages) == ["dynamic task"]
            assert "ROOT private instructions" not in info.instructions
            return ModelResponse(parts=[TextPart("dynamic result")])

        completed = _returns(messages)
        if not completed:
            assert any(tool.name == "delegate_task" for tool in info.function_tools)
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "delegate_task",
                        {
                            "agent_name": "unconfigured",
                            "task": "dynamic task",
                            "instructions": "DYNAMIC CHILD: answer only the task",
                            "description": "Created by the parent for this task",
                        },
                        "delegate-1",
                    )
                ]
            )
        assert [part.content for part in completed] == ["dynamic result"]
        return ModelResponse(parts=[TextPart("parent done")])

    result = asyncio.run(
        _build(FunctionModel(model), RuntimeConfig(dynamic_subagents=True)).run(
            "ROOT secret query"
        )
    )
    assert result.output == "parent done"
    assert child_calls == [(["dynamic task"], "DYNAMIC CHILD: answer only the task")]


def test_dynamic_delegate_uses_requested_configured_skill_and_reader(tmp_path):
    """Dynamic declarations are bounded by the parent's configured catalogs."""

    from roborsi.runtime import RuntimeConfig

    skill_dir = tmp_path / "skills" / "scene"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: scene\ndescription: inspect the scene\n---\nscene instructions",
        encoding="utf-8",
    )
    note = tmp_path / "observation.txt"
    note.write_text("observation", encoding="utf-8")
    seen = []

    def read_text_file(path: str) -> str:
        return (tmp_path / path).read_text(encoding="utf-8")

    def model(messages, info):
        if info.instructions.startswith("DYNAMIC CHILD"):
            seen.append(info)
            names = {tool.name for tool in info.function_tools}
            assert names == {"read_text_file", "read_skill"}
            returned = _returns(messages)
            if not returned:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "read_text_file",
                            {"path": "observation.txt"},
                            "text-1",
                        )
                    ]
                )
            if len(returned) == 1:
                assert returned[-1].content == "observation"
                return ModelResponse(
                    parts=[ToolCallPart("read_skill", {"name": "scene"}, "skill-1")]
                )
            assert "scene instructions" in returned[-1].content["content"]
            return ModelResponse(parts=[TextPart("child inspected")])

        if not _returns(messages):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "delegate_task",
                        {
                            "agent_name": "scene_reader",
                            "task": "inspect this observation",
                            "instructions": "DYNAMIC CHILD: inspect the supplied scene",
                            "skills": ["scene"],
                            "tools": ["read_text_file", "read_skill"],
                        },
                        "delegate-1",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart("parent done")])

    runtime = RuntimeConfig(dynamic_subagents=True, skill_paths=[skill_dir])
    result = asyncio.run(
        _build(
            FunctionModel(model),
            runtime,
            [Tool(read_text_file)],
        ).run("ROOT observation")
    )
    assert result.output == "parent done"
    assert len(seen) == 3


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "agent_name": "bad_tool",
                "task": "try it",
                "instructions": "DYNAMIC CHILD",
                "tools": ["move"],
            },
            "move",
        ),
        (
            {
                "agent_name": "missing_reader",
                "task": "try it",
                "instructions": "DYNAMIC CHILD",
                "tools": ["read_text_file"],
            },
            "read_text_file",
        ),
        (
            {
                "agent_name": "bad_skill",
                "task": "try it",
                "instructions": "DYNAMIC CHILD",
                "skills": ["missing"],
            },
            "missing",
        ),
        (
            {"agent_name": "missing_instructions", "task": "try it"},
            "instructions",
        ),
    ],
)
def test_dynamic_delegate_rejects_untrusted_definitions(payload, message):
    """Invalid dynamic definitions fail before a child model request."""

    from roborsi.runtime import RuntimeConfig

    def model(messages, info):
        if not _returns(messages):
            return ModelResponse(
                parts=[ToolCallPart("delegate_task", payload, "delegate-1")]
            )
        pytest.fail("invalid dynamic delegate unexpectedly returned to the parent")

    with pytest.raises(ValueError, match=message):
        asyncio.run(
            _build(FunctionModel(model), RuntimeConfig(dynamic_subagents=True)).run(
                "ROOT query"
            )
        )


def test_configured_read_skill_requires_child_skill_paths():
    """Configured children cannot request a reader with no skill catalog."""

    from roborsi.runtime import RuntimeConfig, SubAgentConfig

    runtime = RuntimeConfig(
        subagents={
            "reader": SubAgentConfig(
                instructions="read selected skills",
                tools=["read_skill"],
            )
        }
    )
    with pytest.raises(ValueError, match="read_skill"):
        _build(FunctionModel(lambda messages, info: TextPart("unused")), runtime)


def test_delegate_task_name_is_reserved_for_runtime():
    """A caller cannot shadow the runtime's model-facing delegation tool."""

    from roborsi.runtime import RuntimeConfig

    def delegate_task() -> str:
        return "shadow"

    with pytest.raises(ValueError, match="reserved"):
        _build(
            FunctionModel(lambda messages, info: TextPart("unused")),
            RuntimeConfig(dynamic_subagents=True),
            [Tool(delegate_task)],
        )


def test_dynamic_delegation_can_be_disabled():
    """The compatibility switch removes the parent delegation tool."""

    from roborsi.runtime import RuntimeConfig

    def model(messages, info):
        assert all(tool.name != "delegate_task" for tool in info.function_tools)
        return ModelResponse(parts=[TextPart("done")])

    result = asyncio.run(
        _build(FunctionModel(model), RuntimeConfig(dynamic_subagents=False)).run(
            "ROOT query"
        )
    )
    assert result.output == "done"


def test_dynamic_delegation_limit_is_shared_by_parallel_calls():
    """Parallel model tool calls consume one root-level delegate budget."""

    from roborsi.runtime import RuntimeConfig

    def model(messages, info):
        if info.instructions.startswith("DYNAMIC CHILD"):
            return ModelResponse(parts=[TextPart("child done")])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "delegate_task",
                    {
                        "agent_name": "first_child",
                        "task": "first",
                        "instructions": "DYNAMIC CHILD",
                    },
                    "delegate-1",
                ),
                ToolCallPart(
                    "delegate_task",
                    {
                        "agent_name": "second_child",
                        "task": "second",
                        "instructions": "DYNAMIC CHILD",
                    },
                    "delegate-2",
                ),
            ]
        )

    with pytest.raises(ValueError, match="limit exceeded"):
        asyncio.run(
            _build(
                FunctionModel(model),
                RuntimeConfig(dynamic_subagents=True, max_dynamic_subagents=1),
            ).run("ROOT query")
        )
