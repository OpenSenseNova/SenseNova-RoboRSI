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

"""Episode shutdown prevents new actions and exposes pending cleanup."""

import asyncio
from threading import Event

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from roborsi.embodied_agent import (
    EmbodiedAgent,
    LocalToolSpec,
    _McpToolkit,
    _RemoteTool,
    _ToolBridge,
)
from roborsi.llm import LLMConfig
from roborsi.planner.base import PlannerFailure
from roborsi.tools.toolkit import ToolResult


def _agent(tmp_path, monkeypatch, model, handler=None):
    monkeypatch.setattr(LLMConfig, "build_model", lambda self: FunctionModel(model))
    return EmbodiedAgent(
        mcp_servers=[],
        output_dir=tmp_path,
        llm=LLMConfig("openai", "offline", api_key="test"),
        local_tools=[
            LocalToolSpec("observe", "Observe", {"type": "object"}),
            LocalToolSpec("move", "Move", {"type": "object"}, handler),
        ],
    )


def test_close_cancels_async_model_and_releases_lock(tmp_path, monkeypatch):
    entered, cancelled = Event(), Event()

    async def model(messages, info):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    agent = _agent(tmp_path, monkeypatch, model, lambda _: ToolResult("move", {}))
    episode = agent.start_episode(
        "task", system_prompt="rules", deferred_tools=["observe"]
    )
    assert entered.wait(5)
    episode.close()
    assert cancelled.is_set()
    assert episode.closed and episode.done
    assert not agent._run_lock.locked()
    assert episode.wait().stop_reason == "closed"
    assert episode.next_call(timeout=1) is None


def test_late_sync_model_cannot_dispatch_after_close(tmp_path, monkeypatch):
    entered, release = Event(), Event()
    actions = []

    def model(messages, info):
        entered.set()
        assert release.wait(10)
        return ModelResponse(parts=[ToolCallPart("move", {})])

    def move(args):
        actions.append(args)
        return ToolResult("move", {"_finish": True})

    agent = _agent(tmp_path, monkeypatch, model, move)
    episode = agent.start_episode(
        "task", system_prompt="rules", deferred_tools=["observe"]
    )
    try:
        assert entered.wait(5)
        with pytest.raises(TimeoutError, match="cleanup"):
            episode.close(timeout=0.1)
        assert episode.closed and not episode.done
        assert agent._run_lock.locked()
    finally:
        release.set()
        episode.close()
    assert episode.done
    assert actions == []
    assert not agent._run_lock.locked()


def test_started_handler_finishes_before_cleanup(tmp_path, monkeypatch):
    entered, release, finished = Event(), Event(), Event()

    def model(messages, info):
        return ModelResponse(parts=[ToolCallPart("move", {})])

    def move(args):
        entered.set()
        assert release.wait(10)
        finished.set()
        return ToolResult("move", {"_finish": True})

    agent = _agent(tmp_path, monkeypatch, model, move)
    episode = agent.start_episode(
        "task", system_prompt="rules", deferred_tools=["observe"]
    )
    try:
        assert entered.wait(5)
        with pytest.raises(TimeoutError, match="cleanup"):
            episode.close(timeout=0.1)
        assert not episode.done
        assert not finished.is_set()
    finally:
        release.set()
        episode.close()
    assert finished.is_set() and episode.done
    assert not agent._run_lock.locked()


@pytest.mark.parametrize("name", ["move", "observe", "server__move", "finish"])
def test_closed_dispatch_rejects_all_tool_routes(tmp_path, name):
    bridge = _ToolBridge()
    calls = []
    toolkit = _McpToolkit(
        [],
        tmp_path,
        bridge=bridge,
        deferred_tools=["observe"],
        local_tools=[LocalToolSpec("move", "Move", {"type": "object"}, calls.append)],
    )
    toolkit._tools["server__move"] = _RemoteTool("server", "move", {})
    bridge.close()
    with pytest.raises(PlannerFailure, match="closed"):
        toolkit.execute_tool(name, {})
    assert calls == []
    assert bridge.calls.get_nowait() is None
    toolkit.close()


@pytest.mark.parametrize(
    "names", [("move", "move"), ("finish", "move"), ("move", "finish")]
)
def test_terminal_feedback_skips_remaining_calls_and_preserves_trace(
    tmp_path, monkeypatch, names
):
    import json

    from roborsi.runtime import RuntimeConfig

    requests = []

    def model(messages, info):
        requests.append(messages)
        return ModelResponse(
            parts=[
                ToolCallPart(
                    name,
                    {"status": "success", "summary": "done"}
                    if name == "finish"
                    else {},
                    tool_call_id=f"call-{i}",
                )
                for i, name in enumerate(names)
            ]
        )

    agent = _agent(tmp_path, monkeypatch, model)
    agent.runtime = RuntimeConfig()
    episode = agent.start_episode(
        "task",
        system_prompt="rules",
        deferred_tools=["observe", "move"],
        include_finish=True,
    )
    with episode:
        if names[0] == "move":
            call = episode.next_call(timeout=5)
            assert call.name == "move"
            episode.complete(
                call,
                ToolResult(
                    "move",
                    {
                        "_finish": True,
                        "observation": "terminal",
                        "_image_bytes": b"terminal-camera",
                    },
                ),
            )
        result = episode.wait(timeout=5)
        assert result.error is None
        assert result.stop_reason == "finish"
        assert len(requests) == 1
        assert episode.next_call(timeout=1) is None
    feedback = [m for m in result.messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in feedback] == ["call-0", "call-1"]
    assert json.loads(feedback[1]["content"])["skipped"] is True
    if names[0] == "move":
        assert result.finish_result["observation"] == "terminal"
        assert any(
            p.read_bytes() == b"terminal-camera"
            for p in (tmp_path / "trace/content").iterdir()
        )
    manifest = json.loads((tmp_path / "trace/manifest.json").read_text())
    assert manifest["status"] == "completed"
    events = [
        json.loads(line)
        for line in (tmp_path / "trace/events.jsonl").read_text().splitlines()
    ]
    results = [event for event in events if event["type"] == "tool_end"]
    assert len(results) == 2
    assert [event["tool_call_id"] for event in results] == ["call-0", "call-1"]


def test_close_releases_pending_deferred_call_and_records_cancellation(
    tmp_path, monkeypatch
):
    import json

    from roborsi.runtime import RuntimeConfig

    def model(messages, info):
        return ModelResponse(parts=[ToolCallPart("move", {})])

    agent = _agent(tmp_path, monkeypatch, model)
    agent.runtime = RuntimeConfig()
    episode = agent.start_episode(
        "task", system_prompt="rules", deferred_tools=["observe", "move"]
    )
    assert episode.next_call(timeout=5).name == "move"
    episode.close()
    assert episode.done
    result = episode.wait(timeout=1)
    assert result.error is None
    assert result.stop_reason == "closed"
    assert episode.next_call(timeout=0) is None
    assert episode.next_call(timeout=0) is None
    manifest = json.loads((tmp_path / "trace/manifest.json").read_text())
    assert manifest["status"] == "cancelled"
    events = [
        json.loads(line)
        for line in (tmp_path / "trace/events.jsonl").read_text().splitlines()
    ]
    tool_end = next(event for event in events if event["type"] == "tool_end")
    assert tool_end["payload"]["status"] == "cancelled"
