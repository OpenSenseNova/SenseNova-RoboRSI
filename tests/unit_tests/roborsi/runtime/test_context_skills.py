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

import pytest
from pydantic_ai import Tool, ToolReturn
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import (
    CachePoint,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RunUsage, UsageLimits

from roborsi.runtime import RuntimeConfig, SubAgentConfig
from roborsi.runtime.factory import build_runtime_agent


def build(model, runtime, tools=(), capabilities=()):
    return build_runtime_agent(
        model=model,
        runtime=runtime,
        tools=tools,
        capabilities=capabilities,
        system_prompt="ROOT",
        max_tokens=200,
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_engine_produces_final_context_after_explicit_processors(asynchronous):
    seen = []

    def engine(ctx, messages):
        seen.append(("engine", len(messages)))
        messages[0].parts[0].content = "task"
        return messages

    async def async_engine(ctx, messages):
        return engine(ctx, messages)

    def images(messages):
        seen.append(("images", len(messages)))
        return messages

    def model(messages, info):
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("read", {}, "r")])
        return ModelResponse(parts=[TextPart("done")])

    def read() -> str:
        return "evidence"

    result = asyncio.run(
        build(
            FunctionModel(model),
            RuntimeConfig(context_engine=async_engine if asynchronous else engine),
            [Tool(read)],
            [ProcessHistory(images)],
        ).run("task")
    )
    assert result.output == "done"
    assert seen == [("images", 1), ("engine", 1), ("images", 3), ("engine", 3)]


def test_recent_turns_keeps_parallel_tool_groups_and_current_task():
    from roborsi.runtime.context_engine import ContextPolicy, recent_turns

    messages = [ModelRequest(parts=[UserPromptPart("task")])]
    for turn in range(3):
        messages += [
            ModelResponse(
                parts=[
                    ToolCallPart("read", {}, f"{turn}a"),
                    ToolCallPart("read", {}, f"{turn}b"),
                ]
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart("read", "a", f"{turn}a"),
                    ToolReturnPart("read", "b", f"{turn}b"),
                ]
            ),
        ]
    selected = recent_turns(None, messages, keep_turns=1)
    assert selected == [messages[0], messages[-2], messages[-1]]
    assert ContextPolicy(strategy="recent_turns", keep_turns=1).keep_turns == 1


def test_context_result_selects_user_and_tool_cache_boundaries():
    from roborsi.runtime import CacheBreakpoint, ContextResult

    seen = []

    def engine(ctx, messages):
        del ctx
        if len(messages) == 1:
            return ContextResult(messages, (CacheBreakpoint(0, 0, 0),))
        return ContextResult(messages, (CacheBreakpoint(2, 0),))

    def model(messages, info):
        seen.append((messages, dict(info.model_settings or {})))
        if len(seen) == 1:
            return ModelResponse(parts=[ToolCallPart("read", {}, "result-1")])
        return ModelResponse(parts=[TextPart("done")])

    result = asyncio.run(
        build(
            FunctionModel(model),
            RuntimeConfig(context_engine=engine),
            [Tool(lambda: "observation", name="read")],
        ).run("task")
    )
    assert result.output == "done"
    assert seen[0][0][0].parts[0].content == ["task", CachePoint()]
    assert "roborsi_cache_tool_call_ids" not in seen[0][1]
    assert seen[1][1]["roborsi_cache_tool_call_ids"] == ("result-1",)


def test_context_engine_output_is_not_pruned_after_callback(monkeypatch):
    from pydantic_ai import BinaryContent

    from roborsi.planner import api_loop

    monkeypatch.setattr(api_loop, "_MAX_HISTORY_IMAGE_BYTES", 1)
    images = [
        BinaryContent(data=f"frame-{index}".encode(), media_type="image/png")
        for index in range(3)
    ]

    def engine(ctx, messages):
        del ctx
        return messages

    def model(messages, info):
        del info
        assert messages[0].parts[0].content == ["task", *images]
        return ModelResponse(parts=[TextPart("done")])

    result = asyncio.run(
        build(FunctionModel(model), RuntimeConfig(context_engine=engine)).run(
            ["task", *images]
        )
    )
    assert result.output == "done"


def test_built_in_context_policy_selects_cache_boundaries():
    from pydantic_ai import BinaryContent

    from roborsi.runtime import ContextPolicy

    image = BinaryContent(data=b"frame", media_type="image/png")
    seen = []

    def model(messages, info):
        seen.append((messages, dict(info.model_settings or {})))
        if len(seen) == 1:
            return ModelResponse(parts=[ToolCallPart("read", {}, "tool-1")])
        return ModelResponse(parts=[TextPart("done")])

    def read() -> str:
        return "observation"

    result = asyncio.run(
        build(
            FunctionModel(model),
            RuntimeConfig(
                context=ContextPolicy(
                    cache_initial_task=True,
                    cache_initial_context=True,
                    cache_latest_tool_feedback=True,
                )
            ),
            [Tool(read)],
        ).run(["task", image])
    )
    assert result.output == "done"
    assert seen[0][0][0].parts[0].content == ["task", CachePoint(), image, CachePoint()]
    assert seen[1][1]["roborsi_cache_tool_call_ids"] == ("tool-1",)


def test_legacy_image_settings_cannot_override_custom_context():
    from roborsi.llm.client import LLMConfig

    with pytest.raises(ValueError, match="custom context_engine"):
        RuntimeConfig(
            llm=LLMConfig("openai", "test", image_history_groups=2),
            context_engine=lambda ctx, messages: messages,
        )


def test_tool_cache_breakpoint_rejects_unsupported_model():
    from roborsi.runtime import ContextPolicy

    def model(messages, info):
        del info
        return ModelResponse(parts=[ToolCallPart("read", {}, "tool-1")])

    def read() -> str:
        return "observation"

    unsupported = FunctionModel(model)
    unsupported._system = "anthropic"
    with pytest.raises(ValueError, match="RoboRSI OpenAI Responses"):
        asyncio.run(
            build(
                unsupported,
                RuntimeConfig(context=ContextPolicy(cache_latest_tool_feedback=True)),
                [Tool(read)],
            ).run("task")
        )


@pytest.mark.parametrize("drop", ["input", "feedback", "call"])
def test_invalid_context_fails_before_model_request(drop):
    calls = 0

    def engine(ctx, messages):
        if drop == "input":
            return []
        if len(messages) > 1:
            return messages[:-1] if drop == "feedback" else [messages[0], messages[-1]]
        return messages

    def model(messages, info):
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[ToolCallPart("read", {}, "r")])

    def read() -> str:
        return "new evidence"

    with pytest.raises(ValueError, match="context"):
        asyncio.run(
            build(
                FunctionModel(model), RuntimeConfig(context_engine=engine), [Tool(read)]
            ).run("task")
        )
    assert calls == (0 if drop == "input" else 1)


def test_context_factory_isolated_for_repeated_parallel_child_runs():
    instances = []

    def factory():
        steps = []
        instances.append(steps)

        def engine(ctx, messages):
            steps.append(len(messages))
            return messages

        return engine

    def model(messages, info):
        if info.instructions == "CHILD":
            return ModelResponse(parts=[TextPart("child done")])
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "delegate_task", {"agent_name": "child", "task": task}, task
                    )
                    for task in ("a", "b")
                ]
            )
        return ModelResponse(parts=[TextPart("done")])

    agent = build(
        FunctionModel(model),
        RuntimeConfig(
            subagents={
                "child": SubAgentConfig(
                    instructions="CHILD", context_engine_factory=factory
                )
            }
        ),
    )

    async def scenario():
        await agent.run("task one")
        await agent.run("task two")

    asyncio.run(scenario())
    assert instances == [[1], [1], [1], [1]]


def test_summary_consumes_shared_request_budget_without_recursing():
    from roborsi.runtime.context_engine import summarize_history

    engines = 0
    calls = []

    async def engine(ctx, messages):
        nonlocal engines
        engines += 1
        assert await summarize_history(ctx, messages) == "summary"
        return messages

    def model(messages, info):
        calls.append(info.instructions)
        return ModelResponse(parts=[TextPart("summary")])

    from pydantic_ai.exceptions import UsageLimitExceeded

    usage = RunUsage()
    with pytest.raises(UsageLimitExceeded):
        asyncio.run(
            build(FunctionModel(model), RuntimeConfig(context_engine=engine)).run(
                "task", usage=usage, usage_limits=UsageLimits(request_limit=1)
            )
        )
    assert engines == 1
    assert len(calls) == usage.requests == 1


def test_skill_catalog_is_lazy_fresh_and_confined(tmp_path):
    from roborsi.runtime.skills import SkillCatalog

    folder = tmp_path / "camera"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        "---\nname: inspect\ndescription: Inspect images.\n---\nPRIVATE BODY",
        encoding="utf-8",
    )
    catalog = SkillCatalog([folder], max_bytes=128)
    assert (
        "inspect" in catalog.instructions and "Inspect images." in catalog.instructions
    )
    assert "PRIVATE BODY" not in catalog.instructions
    (folder / "notes.md").write_text("new notes", encoding="utf-8")
    result = catalog.read_skill("inspect", "notes.md")
    assert result["content"] == "new notes" and result["source"].endswith("notes.md")
    (folder / "notes.md").write_text("updated notes", encoding="utf-8")
    assert catalog.read_skill("inspect", "notes.md")["content"] == "updated notes"
    assert (
        catalog.read_skill("inspect", "../outside")["error"]["code"] == "outside_skill"
    )
    assert catalog.read_skill("missing")["error"]["code"] == "unknown_skill"
    (folder / "notes.md").write_text("x" * 129, encoding="utf-8")
    assert catalog.read_skill("inspect", "notes.md")["error"] == {
        "code": "too_large",
        "size_bytes": 129,
        "max_bytes": 128,
    }
    with pytest.raises(ValueError, match="duplicate"):
        SkillCatalog([folder, folder])


def test_child_skill_tool_uses_its_catalog_not_parent(tmp_path):
    root = tmp_path / "root"
    child = tmp_path / "child"
    for folder in [root, child]:
        folder.mkdir()
        (folder / "SKILL.md").write_text(f"{folder.name} guide", encoding="utf-8")

    def model(messages, info):
        if info.instructions.startswith("CHILD"):
            assert "root guide" not in info.instructions
            if len(messages) == 1:
                return ModelResponse(
                    parts=[ToolCallPart("read_skill", {"name": "child"}, "s")]
                )
            assert messages[-1].parts[0].content["content"] == "child guide"
            return ModelResponse(parts=[TextPart("read")])
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "delegate_task", {"agent_name": "child", "task": "read"}, "d"
                    )
                ]
            )
        return ModelResponse(parts=[TextPart("done")])

    runtime = RuntimeConfig(
        skill_paths=[root],
        subagents={
            "child": SubAgentConfig(
                instructions="CHILD", tools=["read_skill"], skill_paths=[child]
            )
        },
    )
    assert (
        asyncio.run(build(FunctionModel(model), runtime).run("task")).output == "done"
    )


def test_full_config_resolves_paths_and_rejects_conflicting_models(tmp_path):
    (tmp_path / "skill").mkdir()
    (tmp_path / "skill" / "SKILL.md").write_text("Guide.", encoding="utf-8")
    path = tmp_path / "runtime.yaml"
    path.write_text(
        "llm: {provider: openai, model: root, retry: {max_retries: 3}}\ncontext: {strategy: recent_turns, keep_turns: 2}\nskill_paths: [skill]\nsubagents:\n  child:\n    instructions: CHILD\n    llm: {provider: anthropic, model: specialist}\n",
        encoding="utf-8",
    )
    config = RuntimeConfig.from_file(path)
    assert config.llm.model == "root" and config.llm.retry.max_retries == 3
    assert config.skill_paths == (tmp_path / "skill",)
    assert config.subagents["child"].llm.provider == "anthropic"
    with pytest.raises(ValueError, match="model.*llm|llm.*model"):
        SubAgentConfig(instructions="CHILD", model="openai:x", llm=config.llm)


def test_resource_validation_reports_missing_skill_before_execution(tmp_path):
    runtime = RuntimeConfig(skill_paths=[tmp_path / "missing"])
    with pytest.raises(FileNotFoundError):
        runtime.validate_resources()
    child = RuntimeConfig(
        subagents={
            "child": SubAgentConfig(
                instructions="CHILD", skills=[tmp_path / "missing.md"]
            )
        }
    )
    with pytest.raises(FileNotFoundError):
        child.validate_resources()


@pytest.mark.parametrize(
    "llm",
    [
        {"provider": "openai", "model": 123},
        {"provider": "openai", "model": "model", "retry": "three"},
    ],
)
def test_malformed_model_configuration_fails_at_runtime_boundary(llm):
    with pytest.raises(ValueError, match="llm"):
        RuntimeConfig(llm=llm)


def test_full_child_llm_uses_its_endpoint_retry_and_settings(monkeypatch):
    from roborsi.llm.client import LLMConfig
    from roborsi.llm.retry import RetryPolicy

    config = LLMConfig(
        provider="openai",
        model="specialist",
        base_url="https://example.invalid/v1",
        retry=RetryPolicy(max_retries=0),
        parallel_tool_calls=False,
    )

    def child(messages, info):
        assert info.model_settings["parallel_tool_calls"] is False
        return ModelResponse(parts=[TextPart("specialist answer")])

    def make_model(self):
        assert self.base_url == "https://example.invalid/v1"
        return FunctionModel(child, settings={"parallel_tool_calls": False})

    monkeypatch.setattr(LLMConfig, "build_model", make_model)

    def parent(messages, info):
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "delegate_task", {"agent_name": "child", "task": "review"}, "d"
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(messages[-1].parts[0].content)])

    runtime = RuntimeConfig(
        subagents={"child": SubAgentConfig(instructions="CHILD", llm=config)}
    )
    assert (
        asyncio.run(build(FunctionModel(parent), runtime).run("task")).output
        == "specialist answer"
    )


def test_skill_symlink_cannot_read_outside_catalog(tmp_path):
    from roborsi.runtime.skills import SkillCatalog

    directory = tmp_path / "skill"
    directory.mkdir()
    (directory / "SKILL.md").write_text("Guide.", encoding="utf-8")
    outside = tmp_path / "private.md"
    outside.write_text("private content", encoding="utf-8")
    (directory / "escape.md").symlink_to(outside)
    assert (
        SkillCatalog([directory]).read_skill("skill", "escape.md")["error"]["code"]
        == "outside_skill"
    )


def test_child_llm_image_policy_overrides_parent_window(monkeypatch):
    from functools import partial

    from pydantic_ai import BinaryContent

    from roborsi.llm.client import LLMConfig
    from roborsi.planner.api_loop import _prune_history_images

    image_counts = []

    def child(messages, info):
        images = [
            item
            for message in messages
            for part in message.parts
            if isinstance(part, UserPromptPart) and isinstance(part.content, list)
            for item in part.content
            if isinstance(item, BinaryContent)
        ]
        image_counts.append(len(images))
        if len(image_counts) <= 3:
            return ModelResponse(
                parts=[ToolCallPart("read_image", {}, str(len(image_counts)))]
            )
        return ModelResponse(parts=[TextPart("done")])

    monkeypatch.setattr(LLMConfig, "build_model", lambda self: FunctionModel(child))

    def parent(messages, info):
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "delegate_task", {"agent_name": "child", "task": "inspect"}, "d"
                    )
                ]
            )
        return ModelResponse(parts=[TextPart("done")])

    def read_image() -> ToolReturn:
        return ToolReturn(
            return_value="view",
            content=[BinaryContent(data=b"image", media_type="image/png")],
        )

    config = RuntimeConfig(
        subagents={
            "child": SubAgentConfig(
                instructions="CHILD",
                tools=["read_image"],
                llm=LLMConfig(provider="openai", model="child", image_history_groups=1),
            )
        }
    )
    asyncio.run(
        build(
            FunctionModel(parent),
            config,
            [Tool(read_image)],
            [ProcessHistory(partial(_prune_history_images, max_groups=2))],
        ).run("task")
    )
    assert image_counts == [0, 1, 1, 1]
