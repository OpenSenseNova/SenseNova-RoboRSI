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

"""Exercise the user-facing facade across a real local MCP transport."""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import BinaryContent
from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from roborsi.data_convert import TextDocument
from roborsi.embodied_agent import EmbodiedAgent, LocalToolSpec, McpServer
from roborsi.llm import LLMConfig
from roborsi.planner.base import PlannerResult
from roborsi.planner.utils.http_mcp_server import HttpMcpServer
from roborsi.runtime import RuntimeConfig, SubAgentConfig
from roborsi.tools.toolkit import ToolResult


class _RobotTools:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get_tools_spec(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "move_eef",
                "description": "Move to a target pose and return a camera frame.",
                "input_schema": {
                    "type": "object",
                    "properties": {"pose": {"type": "string"}},
                    "required": ["pose"],
                },
            },
            {
                "name": "snapshot",
                "description": "Capture the current camera frame.",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> ToolResult:
        self.calls.append((name, input_dict))
        if name == "move_eef":
            if input_dict["pose"] == "invalid":
                return ToolResult(name=name, result={"error": "pose unreachable"})
            return ToolResult(name=name, result={"pose": input_dict["pose"]})
        return ToolResult(name=name, result={"camera": "front", "_image_bytes": b"png"})


def test_mcp_allowlist_hides_and_rejects_unselected_tools(tmp_path: Path) -> None:
    from roborsi.embodied_agent import _McpToolkit

    robot = _RobotTools()
    server = HttpMcpServer(robot)
    server.start()
    toolkit = _McpToolkit(
        [McpServer(name="robot", url=server.url, tools=("snapshot",))], tmp_path
    )
    try:
        toolkit.start()
        assert {item["name"] for item in toolkit.get_tools_spec()} == {
            "robot__snapshot",
            "finish",
        }
        assert (
            "unknown tool"
            in toolkit.execute_tool("robot__move_eef", {"pose": "target"}).result[
                "error"
            ]
        )
        assert not robot.calls
        toolkit.execute_tool("robot__snapshot", {})
        assert robot.calls == [("snapshot", {})]
    finally:
        toolkit.close()
        server.stop()


def test_embodied_runtime_delegates_before_executing_robot_tool(tmp_path, monkeypatch):
    from pydantic_ai.messages import (
        ModelResponse,
        TextPart,
        ToolCallPart,
        ToolReturnPart,
    )
    from pydantic_ai.models.function import FunctionModel

    from roborsi.planner import base

    def model(messages, info):
        if info.instructions == "CHILD":
            assert info.function_tools == []
            return ModelResponse(parts=[TextPart("target")])
        returned = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not returned:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "delegate_task",
                        {"agent_name": "reviewer", "task": "Choose a pose"},
                        "d",
                    )
                ]
            )
        if len(returned) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "robot__move_eef", {"pose": returned[0].content}, "move"
                    )
                ]
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "finish", {"status": "success", "summary": "placed"}, "finish"
                )
            ]
        )

    monkeypatch.setattr(base, "build_api_model", lambda *a, **kw: FunctionModel(model))
    robot = _RobotTools()
    server = HttpMcpServer(robot)
    server.start()
    try:
        agent = EmbodiedAgent(
            mcp_servers=[McpServer(name="robot", url=server.url)],
            output_dir=tmp_path / "episode",
            model="openai:offline",
            runtime=RuntimeConfig(
                subagents={"reviewer": SubAgentConfig(instructions="CHILD")}
            ),
        )
        result = agent.run("Place the block", system_prompt="ROOT")
    finally:
        server.stop()
    assert result.error is None
    assert result.finish_result["status"] == "success"
    assert result.stats["requests"] == 4
    assert robot.calls == [("move_eef", {"pose": "target"})]


@pytest.mark.parametrize("planner", ["codex", "claude_code"])
def test_embodied_runtime_rejected_before_mcp_or_output_creation(tmp_path, planner):
    output = tmp_path / "unused"
    agent = EmbodiedAgent(
        mcp_servers=[McpServer(name="robot", url="http://127.0.0.1:1/mcp")],
        output_dir=output,
        planner=planner,
        runtime=RuntimeConfig(),
    )
    with pytest.raises(ValueError, match="runtime.*api"):
        agent.run("Task", system_prompt="Instructions")
    assert not output.exists()


@pytest.mark.parametrize("with_context", [False, True])
def test_embodied_agent_discovers_calls_and_preserves_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_context: bool
) -> None:
    robot = _RobotTools()
    server = HttpMcpServer(robot)
    server.start()
    skill = tmp_path / "SKILL.md"
    skill.write_text("Use the front camera after each motion.", encoding="utf-8")
    seen: dict[str, Any] = {}

    class _Planner:
        def solve(self, **kwargs: Any) -> PlannerResult:
            seen.update(kwargs)
            toolkit = kwargs["toolkit"]
            specs = {item["name"]: item for item in toolkit.get_tools_spec()}
            assert specs["robot__move_eef"]["input_schema"]["required"] == ["pose"]
            assert toolkit.execute_tool(
                "robot__move_eef", {"pose": "target"}
            ).result == {"pose": "target"}
            rejected = toolkit.execute_tool("robot__move_eef", {"pose": "invalid"})
            assert rejected.result["error"] == "pose unreachable"
            snapshot = toolkit.execute_tool("robot__snapshot", {})
            assert snapshot.content_blocks[0]["type"] == "text"
            assert snapshot.content_blocks[1]["type"] == "image"
            assert (
                base64.b64decode(snapshot.content_blocks[1]["source"]["data"]) == b"png"
            )
            finish = toolkit.execute_tool(
                "finish", {"status": "success", "summary": "placed"}
            )
            assert finish.is_finish
            return PlannerResult(
                finish_result=finish.result,
                stats={
                    "total_input_tokens": 10,
                    "total_output_tokens": 5,
                    "cache_read_tokens": 3,
                    "cache_write_tokens": 2,
                    "requests": 1,
                },
            )

    planner_kwargs: dict[str, Any] = {}

    def build_planner(*args: Any, **kwargs: Any) -> _Planner:
        planner_kwargs.update(kwargs)
        return _Planner()

    monkeypatch.setattr("roborsi.embodied_agent.build_planner", build_planner)
    llm = LLMConfig("openai", "gpt-4o", api_key="test")
    agent = EmbodiedAgent(
        mcp_servers=[McpServer(name="robot", url=server.url)],
        output_dir=tmp_path / "episode",
        llm=llm,
    )
    image = BinaryContent(data=b"initial-png", media_type="image/png")
    context_args = (
        {
            "memory": [TextDocument("grasp", "Use the top grasp.", "memory/grasp.md")],
            "initial_context": ["Initial camera", image],
        }
        if with_context
        else {}
    )
    try:
        result = agent.run(
            "Place the block.",
            system_prompt="Use safe poses.",
            skills=[skill],
            **context_args,
        )
    finally:
        server.stop()

    assert result.finish_result["status"] == "success"
    assert planner_kwargs["llm_config"] is llm
    assert result.stats["llm_usage"]["total_tokens"] == 15
    assert result.stats["llm_usage"]["cache_read_tokens"] == 3
    if with_context:
        assert seen["user_message"] == [
            "Place the block.\n\n## Memory: grasp\nSource: memory/grasp.md\n\nUse the top grasp.",
            "Initial camera",
            image,
        ]
        assert seen["user_message"][2] is image
    else:
        assert seen["user_message"] == "Place the block."
    assert seen["system_prompt"] == (
        f"Use safe poses.\n\n## Skill: {tmp_path.name}\n\n"
        "Use the front camera after each motion."
    )
    assert robot.calls == [
        ("move_eef", {"pose": "target"}),
        ("move_eef", {"pose": "invalid"}),
        ("snapshot", {}),
    ]


def test_missing_skill_fails_before_mcp_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_start(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("MCP must not start before context is loaded")

    monkeypatch.setattr("roborsi.embodied_agent._McpToolkit.start", reject_start)
    agent = EmbodiedAgent(
        mcp_servers=[McpServer(name="robot", url="http://localhost/mcp")],
        output_dir=tmp_path / "episode",
    )
    with pytest.raises(FileNotFoundError):
        agent.run("task", system_prompt="rules", skills=[tmp_path / "missing.md"])
    assert not (tmp_path / "episode").exists()


def test_episode_hands_motion_to_benchmark_and_returns_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    robot = _RobotTools()
    server = HttpMcpServer(robot)
    server.start()
    requests = 0

    def model(messages: list[Any], info: Any) -> ModelResponse:
        nonlocal requests
        requests += 1
        assert {tool.name for tool in info.function_tools} == {
            "move_eef",
            "snapshot",
        }
        if requests == 2:
            assert "executed" in str(messages)
            assert "BinaryContent" in str(messages)
        return ModelResponse(
            parts=[ToolCallPart("move_eef", {"pose": "target"})],
            usage=RequestUsage(input_tokens=10, output_tokens=2),
        )

    monkeypatch.setattr(LLMConfig, "build_model", lambda self: FunctionModel(model))
    agent = EmbodiedAgent(
        mcp_servers=[McpServer(name="robot", url=server.url, expose_unprefixed=True)],
        output_dir=tmp_path / "episode",
        llm=LLMConfig("openai", "offline", api_key="test"),
        max_turns=3,
    )
    try:
        with agent.start_episode(
            "Pick up the object.",
            system_prompt="Call move_eef once.",
            deferred_tools=["move_eef"],
        ) as episode:
            call = episode.next_call(timeout=10)
            assert call is not None
            assert call.name == "move_eef"
            assert call.arguments == {"pose": "target"}
            with pytest.raises(ValueError, match="name does not match"):
                episode.complete(call, ToolResult("robot__snapshot", {}))
            episode.complete(
                call,
                ToolResult(
                    call.name,
                    {"status": "executed", "_image_bytes": b"png"},
                ),
            )
            next_call = episode.next_call(timeout=10)
            assert next_call is not None
            episode.complete(
                next_call,
                ToolResult(next_call.name, {"_finish": True, "status": "native_end"}),
            )
            result = episode.wait(timeout=10)
    finally:
        server.stop()

    assert requests == 2
    assert result.finish_result is not None
    assert result.finish_result["status"] == "native_end"
    assert result.stats["requests"] == 2
    assert robot.calls == []


def test_local_tool_runs_without_mcp_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def model(messages: list[Any], info: Any) -> ModelResponse:
        assert {tool.name for tool in info.function_tools} == {"move_eef", "finish"}
        if calls:
            return ModelResponse(
                parts=[ToolCallPart("finish", {"status": "success", "summary": "done"})]
            )
        return ModelResponse(parts=[ToolCallPart("move_eef", {"x": 0.2})])

    monkeypatch.setattr(LLMConfig, "build_model", lambda self: FunctionModel(model))

    def move(arguments: dict[str, Any]) -> ToolResult:
        calls.append(arguments)
        return ToolResult("move_eef", {"state": "arrived", "_image_bytes": b"png"})

    agent = EmbodiedAgent(
        mcp_servers=[],
        local_tools=[
            LocalToolSpec(
                "move_eef",
                "Move the end effector.",
                {"type": "object", "properties": {"x": {"type": "number"}}},
                handler=move,
            )
        ],
        output_dir=tmp_path / "episode",
        llm=LLMConfig("openai", "offline", api_key="test"),
    )
    result = agent.run("Move the block.", system_prompt="Use move_eef.")
    assert calls == [{"x": 0.2}]
    assert result.finish_result is not None
    assert result.error is None


def test_local_tool_defers_to_benchmark_thread_without_mcp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def model(messages: list[Any], info: Any) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart("move_eef", {"x": 0.2})])

    monkeypatch.setattr(LLMConfig, "build_model", lambda self: FunctionModel(model))
    agent = EmbodiedAgent(
        mcp_servers=[],
        local_tools=[
            LocalToolSpec(
                "move_eef",
                "Move the end effector.",
                {"type": "object", "properties": {"x": {"type": "number"}}},
            )
        ],
        output_dir=tmp_path / "episode",
        llm=LLMConfig("openai", "offline", api_key="test"),
    )
    with agent.start_episode(
        "Move the block.",
        system_prompt="Use move_eef.",
        deferred_tools=["move_eef"],
    ) as episode:
        call = episode.next_call(timeout=10)
        assert call is not None
        assert call.arguments == {"x": 0.2}
        episode.complete(call, ToolResult("move_eef", {"_finish": True}))
        assert episode.wait(timeout=10).finish_result == {"_finish": True}


def test_task_selected_skills_change_without_rebuilding_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("pick", "place"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "SKILL.md").write_text(f"{name} instructions", encoding="utf-8")
    seen: list[str] = []

    def model(messages: list[Any], info: Any) -> ModelResponse:
        name = "pick" if "- pick:" in info.instructions else "place"
        assert f"- {name}:" in info.instructions
        assert f"- {'place' if name == 'pick' else 'pick'}:" not in info.instructions
        if any(
            isinstance(part, ToolReturnPart)
            for message in messages
            for part in message.parts
        ):
            seen.append(name)
            return ModelResponse(
                parts=[ToolCallPart("finish", {"status": "success", "summary": name})]
            )
        return ModelResponse(parts=[ToolCallPart("read_skill", {"name": name})])

    monkeypatch.setattr(LLMConfig, "build_model", lambda self: FunctionModel(model))
    agent = EmbodiedAgent(
        mcp_servers=[],
        local_tools=[
            LocalToolSpec(
                "observe",
                "Read state",
                {"type": "object", "properties": {}},
                handler=lambda _: ToolResult("observe", {"state": "ok"}),
            )
        ],
        output_dir=tmp_path / "unused",
        llm=LLMConfig("openai", "offline", api_key="test"),
    )
    for name in ("pick", "place"):
        result = agent.run(
            "Do the task.",
            system_prompt="Use skills.",
            skill_paths=[tmp_path / name],
            output_dir=tmp_path / f"run-{name}",
        )
        assert result.finish_result["summary"] == name
        assert (tmp_path / f"run-{name}" / "trace/manifest.json").exists()
    assert seen == ["pick", "place"]


def test_embodied_agent_rejects_invalid_server_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        McpServer(name="robot")
    with pytest.raises(ValueError, match="exactly one"):
        McpServer(name="robot", url="http://localhost/mcp", command="python")
    with pytest.raises(ValueError, match="unique"):
        EmbodiedAgent(
            mcp_servers=[
                McpServer(name="robot", url="http://localhost/a"),
                McpServer(name="robot", url="http://localhost/b"),
            ],
            output_dir=tmp_path,
        ).run("task", system_prompt="rules")


def test_embodied_agent_starts_stdio_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "robot_server.py"
    script.write_text(
        "from mcp.server.fastmcp import FastMCP\n"
        "server = FastMCP('robot')\n"
        "@server.tool()\n"
        "def snapshot() -> str:\n"
        "    return 'camera ready'\n"
        "server.run(transport='stdio')\n",
        encoding="utf-8",
    )

    class _Planner:
        def solve(self, **kwargs: Any) -> PlannerResult:
            toolkit = kwargs["toolkit"]
            result = toolkit.execute_tool("robot__snapshot", {})
            assert result.content_blocks == [{"type": "text", "text": "camera ready"}]
            return PlannerResult()

    monkeypatch.setattr(
        "roborsi.embodied_agent.build_planner", lambda *a, **kw: _Planner()
    )
    agent = EmbodiedAgent(
        mcp_servers=[
            McpServer(name="robot", command=sys.executable, args=(str(script),))
        ],
        output_dir=tmp_path / "episode",
    )
    agent.run("Observe the scene.", system_prompt="Use snapshot.")


def test_episode_close_waits_for_mcp_action_before_releasing_connection(
    tmp_path, monkeypatch
):
    from threading import Event

    from roborsi.embodied_agent import _McpToolkit

    entered, release = Event(), Event()
    connected = []
    original_start = _McpToolkit.start

    def start(toolkit):
        connected.append(toolkit)
        original_start(toolkit)

    class BlockingRobot(_RobotTools):
        def execute_tool(self, name, input_dict):
            entered.set()
            assert release.wait(10)
            return super().execute_tool(name, input_dict)

    def model(messages, info):
        return ModelResponse(
            parts=[ToolCallPart("robot__move_eef", {"pose": "target"})]
        )

    monkeypatch.setattr(_McpToolkit, "start", start)
    monkeypatch.setattr(LLMConfig, "build_model", lambda self: FunctionModel(model))
    robot = BlockingRobot()
    server = HttpMcpServer(robot)
    server.start()
    agent = EmbodiedAgent(
        mcp_servers=[McpServer(name="robot", url=server.url)],
        local_tools=[LocalToolSpec("observe", "Observe", {"type": "object"})],
        output_dir=tmp_path,
        llm=LLMConfig("openai", "offline", api_key="test"),
    )
    episode = agent.start_episode(
        "task", system_prompt="rules", deferred_tools=["observe"]
    )
    try:
        assert entered.wait(5)
        with pytest.raises(TimeoutError, match="cleanup"):
            episode.close(timeout=0.1)
        assert not episode.done
        assert connected[0]._thread.is_alive()
    finally:
        release.set()
        try:
            episode.close()
        finally:
            server.stop()
    assert robot.calls == [("move_eef", {"pose": "target"})]
    assert episode.done and not agent._run_lock.locked()
    assert connected[0]._thread is None and connected[0]._loop is None
