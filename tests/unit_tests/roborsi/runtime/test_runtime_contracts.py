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

from __future__ import annotations

import asyncio
import subprocess
import sys

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


def _returns(messages):
    return [
        part
        for message in messages
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]


def _prompts(messages):
    return [
        part.content
        for message in messages
        for part in message.parts
        if isinstance(part, UserPromptPart)
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


def test_repeated_delegation_isolates_child_context_and_reloads_skills(tmp_path):
    from roborsi.runtime import RuntimeConfig, SubAgentConfig

    skill = tmp_path / "SKILL.md"
    skill.write_text("First skill content", encoding="utf-8")
    config = RuntimeConfig(
        subagents={
            "reviewer": SubAgentConfig(instructions="CHILD", skills=(skill,)),
        }
    )
    child_prompts = []
    child_instructions = []

    def model(messages, info):
        if info.instructions.startswith("CHILD"):
            child_prompts.append(_prompts(messages))
            child_instructions.append(info.instructions)
            assert info.function_tools == []
            assert info.model_settings["max_tokens"] == 321
            return ModelResponse(parts=[TextPart("review accepted")])
        completed = _returns(messages)
        if len(completed) < 2:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "delegate_task",
                        {
                            "agent_name": "reviewer",
                            "task": f"Explicit task {len(completed)}",
                        },
                        f"delegate-{len(completed)}",
                    )
                ]
            )
        assert [part.content for part in completed] == ["review accepted"] * 2
        return ModelResponse(parts=[TextPart("done")])

    for skill_text in ("First skill content", "Updated skill content"):
        skill.write_text(skill_text, encoding="utf-8")
        result = asyncio.run(
            _build(FunctionModel(model), config).run("ROOT private memory")
        )
        assert result.output == "done"
        assert all(skill_text in text for text in child_instructions[-2:])
    assert child_prompts == [["Explicit task 0"], ["Explicit task 1"]] * 2
    assert all("ROOT" not in text for text in child_instructions)


def test_parallel_delegates_serialize_shared_readers():
    from roborsi.runtime import RuntimeConfig, SubAgentConfig

    async def scenario():
        started = set()
        both_started = asyncio.Event()
        active = 0
        peak = 0
        reads = []

        async def read_text_file(path: str) -> str:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            reads.append(path)
            active -= 1
            return f"evidence:{path}"

        async def model(messages, info):
            returned = _returns(messages)
            if info.instructions == "CHILD":
                task = _prompts(messages)[0]
                assert {tool.name for tool in info.function_tools} == {"read_text_file"}
                if not returned:
                    started.add(task)
                    if len(started) == 2:
                        both_started.set()
                    await asyncio.wait_for(both_started.wait(), 5)
                    return ModelResponse(
                        parts=[
                            ToolCallPart(
                                "read_text_file", {"path": task}, f"read-{task}"
                            )
                        ]
                    )
                return ModelResponse(parts=[TextPart(returned[0].content)])
            if not returned:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "delegate_task",
                            {"agent_name": "reader", "task": task},
                            task,
                        )
                        for task in ("a", "b")
                    ]
                )
            assert {part.content for part in returned} == {"evidence:a", "evidence:b"}
            return ModelResponse(parts=[TextPart("done")])

        config = RuntimeConfig(
            subagents={
                "reader": SubAgentConfig(
                    instructions="CHILD", tools=("read_text_file",)
                )
            }
        )
        result = await _build(
            FunctionModel(model), config, [Tool(read_text_file, sequential=True)]
        ).run("Read both")
        assert result.output == "done"
        assert sorted(reads) == ["a", "b"]
        assert peak == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("tool", ["finish", "move", "missing", "read_text_file"])
def test_invalid_child_tool_fails_before_any_model_call(tool):
    from roborsi.runtime import RuntimeConfig, SubAgentConfig

    def unexpected_request(messages, info):
        pytest.fail("invalid child configuration reached a model")

    config = RuntimeConfig(
        subagents={"child": SubAgentConfig(instructions="CHILD", tools=(tool,))}
    )
    with pytest.raises(ValueError, match=tool):
        _build(FunctionModel(unexpected_request), config)


def test_child_model_override_uses_existing_model_builder(monkeypatch):
    from roborsi.planner import base
    from roborsi.runtime import RuntimeConfig, SubAgentConfig

    def parent(messages, info):
        if info.instructions == "CHILD":
            pytest.fail("child ignored its configured model")
        returned = _returns(messages)
        if returned:
            return ModelResponse(parts=[TextPart(returned[0].content)])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "delegate_task", {"agent_name": "child", "task": "Review"}, "d"
                )
            ]
        )

    def child(messages, info):
        assert info.instructions == "CHILD"
        assert _prompts(messages) == ["Review"]
        return ModelResponse(parts=[TextPart("specialist result")])

    def resolve(model, base_url=None):
        assert model == "openai:specialist"
        return FunctionModel(child)

    monkeypatch.setattr(base, "build_api_model", resolve)
    config = RuntimeConfig(
        subagents={
            "child": SubAgentConfig(instructions="CHILD", model="openai:specialist")
        }
    )
    result = asyncio.run(_build(FunctionModel(parent), config).run("ROOT query"))
    assert result.output == "specialist result"


@pytest.mark.parametrize("stop", ["cancel", "sibling_failure"])
def test_delegated_work_is_drained_when_parent_stops(stop):
    from roborsi.runtime import RuntimeConfig, SubAgentConfig

    async def scenario():
        entered = asyncio.Event()
        exited = asyncio.Event()

        async def model(messages, info):
            if info.instructions == "CHILD":
                task = _prompts(messages)[0]
                if task == "fail":
                    await entered.wait()
                    raise RuntimeError("specialist failed")
                try:
                    entered.set()
                    await asyncio.Event().wait()
                finally:
                    exited.set()
            tasks = ["wait", "fail"] if stop == "sibling_failure" else ["wait"]
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "delegate_task", {"agent_name": "child", "task": task}, task
                    )
                    for task in tasks
                ]
            )

        config = RuntimeConfig(
            subagents={"child": SubAgentConfig(instructions="CHILD")}
        )
        task = asyncio.create_task(
            _build(FunctionModel(model), config).run("ROOT query")
        )
        await asyncio.wait_for(entered.wait(), 5)
        if stop == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(RuntimeError, match="specialist failed"):
                await task
        assert exited.is_set(), "parent returned while a child was still running"

    asyncio.run(scenario())


def test_runtime_file_resolves_skills_relative_to_itself(tmp_path, monkeypatch):
    from roborsi.runtime import RuntimeConfig

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "guide.md").write_text(
        "Skill loaded from config directory", encoding="utf-8"
    )
    path = config_dir / "runtime.yaml"
    path.write_text(
        "subagents:\n  reader:\n    instructions: CHILD\n    skills: [guide.md]\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    config = RuntimeConfig.from_file(path)
    seen = []

    def model(messages, info):
        if info.instructions.startswith("CHILD"):
            seen.append(info.instructions)
            return ModelResponse(parts=[TextPart("read")])
        if _returns(messages):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "delegate_task", {"agent_name": "reader", "task": "Read"}, "d"
                )
            ]
        )

    assert (
        asyncio.run(_build(FunctionModel(model), config).run("Task")).output == "done"
    )
    assert "Skill loaded from config directory" in seen[0]


@pytest.mark.parametrize(
    "document",
    [
        "[]",
        "subagents: [",
        "subagents: null",
        "subagents: {reader: null}",
        "subagents: {reader: {instructions: CHILD, tools: read_image}}",
        "subagents: {reader: {instructions: CHILD, skills: guide.md}}",
        "subagents: {reader: {instructions: CHILD, typo: true}}",
        "subagents: {reader: {instructions: CHILD, model: ''}}",
        "subagents: {'': {instructions: CHILD}}",
        "unexpected: true",
    ],
)
def test_invalid_file_configuration_is_rejected(tmp_path, document):
    from roborsi.runtime import RuntimeConfig

    path = tmp_path / "runtime.yaml"
    path.write_text(document, encoding="utf-8")
    with pytest.raises(ValueError):
        RuntimeConfig.from_file(path)


def test_configuration_import_does_not_load_model_or_harness_sdks():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import roborsi.runtime; assert 'pydantic_ai' not in sys.modules; assert 'pydantic_ai_harness' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_disk_agents_are_not_discovered(tmp_path, monkeypatch):
    from roborsi.runtime import RuntimeConfig, SubAgentConfig

    directory = tmp_path / ".agents" / "agents"
    directory.mkdir(parents=True)
    (directory / "unexpected.md").write_text("Unexpected local agent", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    def model(messages, info):
        assert "unexpected" not in info.instructions
        assert "configured" in info.instructions
        return ModelResponse(parts=[TextPart("done")])

    config = RuntimeConfig(
        subagents={"configured": SubAgentConfig(instructions="CHILD")}
    )
    assert (
        asyncio.run(_build(FunctionModel(model), config).run("Task")).output == "done"
    )


@pytest.mark.parametrize("budget,success", [(0, False), (1, False), (2, True)])
@pytest.mark.parametrize("role", ["root", "child", "dynamic_child", "dynamic_new"])
def test_output_retry_budget_preserves_completed_tools(budget, success, role):
    from pydantic_ai.exceptions import UnexpectedModelBehavior
    from pydantic_ai.messages import RetryPromptPart

    from roborsi.runtime import RuntimeConfig, SubAgentConfig

    delegated = False
    calls = []
    child_results = []
    outputs = iter(
        [
            ModelResponse(parts=[TextPart("")]),
            ModelResponse(
                parts=[ToolCallPart("read_text_file", {"path": "proof"}, "read")]
            ),
            ModelResponse(parts=[TextPart("")]),
            ModelResponse(parts=[TextPart("done")]),
        ]
    )

    def read_text_file(path: str) -> str:
        calls.append(path)
        return "observed"

    def model(messages, info):
        nonlocal delegated
        if role == "root" or info.instructions.startswith("CHILD"):
            return next(outputs)
        if not delegated:
            delegated = True
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "delegate_task",
                        {
                            "agent_name": "reader",
                            "task": "read proof",
                            **(
                                {"instructions": "CHILD", "tools": ["read_text_file"]}
                                if role == "dynamic_new"
                                else {}
                            ),
                        },
                        "delegate",
                    )
                ]
            )
        child_results.extend(
            part.content
            for message in messages
            for part in message.parts
            if isinstance(part, (ToolReturnPart, RetryPromptPart))
        )
        return ModelResponse(parts=[TextPart("done")])

    config = (
        RuntimeConfig(output_retries=budget, dynamic_subagents=role == "dynamic_new")
        if role in {"root", "dynamic_new"}
        else RuntimeConfig(
            dynamic_subagents=role == "dynamic_child",
            subagents={
                "reader": SubAgentConfig(
                    instructions="CHILD",
                    tools=("read_text_file",),
                    output_retries=budget,
                )
            },
        )
    )
    agent = _build(FunctionModel(model), config, [Tool(read_text_file)])
    if success or role == "child":
        assert asyncio.run(agent.run("start")).output == "done"
    else:
        with pytest.raises(UnexpectedModelBehavior, match="maximum output retries"):
            asyncio.run(agent.run("start"))
    if role == "child":
        assert len(child_results) == 1
        if success:
            assert child_results[0] == "done"
        else:
            assert "maximum output retries" in child_results[0]
    assert calls == ([] if budget == 0 else ["proof"])


@pytest.mark.parametrize("value", [-1, True, 1.5, "2", None])
def test_invalid_output_retry_budget_is_rejected(value):
    from roborsi.runtime import RuntimeConfig, SubAgentConfig

    with pytest.raises(ValueError, match="output_retries"):
        RuntimeConfig(output_retries=value)
    with pytest.raises(ValueError, match="output_retries"):
        SubAgentConfig(instructions="child", output_retries=value)


def test_output_retry_budget_loads_from_user_config(tmp_path):
    from roborsi.runtime import RuntimeConfig

    path = tmp_path / "runtime.yaml"
    path.write_text(
        "output_retries: 8\nsubagents:\n  reader:\n    instructions: read\n    output_retries: 2\n"
    )
    config = RuntimeConfig.from_file(path)
    assert config.output_retries == 8
    assert config.subagents["reader"].output_retries == 2
    assert RuntimeConfig().output_retries == 1
