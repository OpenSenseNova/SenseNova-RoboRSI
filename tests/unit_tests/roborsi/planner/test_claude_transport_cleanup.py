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

"""Exercise planner cleanup through the real SDK and in-process MCP server."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import claude_agent_sdk._internal.client
import claude_agent_sdk._internal.transport.subprocess_cli
import pytest
from claude_agent_sdk import Transport

from tests.unit_tests.roborsi.planner.test_claude_contracts import (
    FakeToolkit,
    RecordingSink,
    make_planner,
)


class TerminalTransport(Transport):
    """Replace only the external CLI; keep SDK parsing, MCP and cleanup real."""

    def __init__(self, events: list[str], *, fail_close: bool) -> None:
        self.events = events
        self.fail_close = fail_close
        self.messages: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def connect(self) -> None:
        pass

    async def write(self, data: str) -> None:
        message = json.loads(data)
        if message["type"] == "control_request":
            await self.messages.put(
                {
                    "type": "control_response",
                    "response": {
                        "subtype": "success",
                        "request_id": message["request_id"],
                        "response": {},
                    },
                }
            )
        elif message["type"] == "user":
            await self.messages.put(
                {
                    "type": "assistant",
                    "message": {
                        "model": "offline",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "finish-1",
                                "name": "mcp__roborsi__finish",
                                "input": {"status": "success", "summary": "done"},
                            }
                        ],
                    },
                }
            )
            await self.messages.put(
                {
                    "type": "control_request",
                    "request_id": "initialize-mcp",
                    "request": {
                        "subtype": "mcp_message",
                        "server_name": "roborsi",
                        "message": {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "initialize",
                            "params": {
                                "protocolVersion": "2024-11-05",
                                "capabilities": {},
                                "clientInfo": {"name": "offline", "version": "1"},
                            },
                        },
                    },
                }
            )
        elif message["type"] == "control_response":
            response = message["response"]
            assert response["subtype"] == "success", response
            if response["request_id"] == "initialize-mcp":
                await self.messages.put(
                    {
                        "type": "control_request",
                        "request_id": "initialized-mcp",
                        "request": {
                            "subtype": "mcp_message",
                            "server_name": "roborsi",
                            "message": {
                                "jsonrpc": "2.0",
                                "method": "notifications/initialized",
                            },
                        },
                    }
                )
                return
            if response["request_id"] == "initialized-mcp":
                await self.messages.put(
                    {
                        "type": "control_request",
                        "request_id": "finish-mcp",
                        "request": {
                            "subtype": "mcp_message",
                            "server_name": "roborsi",
                            "message": {
                                "jsonrpc": "2.0",
                                "id": 2,
                                "method": "tools/call",
                                "params": {
                                    "name": "finish",
                                    "arguments": {
                                        "status": "success",
                                        "summary": "done",
                                    },
                                },
                            },
                        },
                    }
                )
                return
            result = response["response"]["mcp_response"]["result"]
            await self.messages.put(
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "finish-1",
                                "content": result["content"],
                            }
                        ],
                    },
                }
            )

    async def read_messages(self):
        while True:
            yield await self.messages.get()

    async def end_input(self) -> None:
        pass

    def is_ready(self) -> bool:
        return True

    async def close(self) -> None:
        self.events.append("sdk-cleanup-started")
        await asyncio.sleep(0.02)
        self.events.append("sdk-cleanup-completed")
        if self.fail_close:
            raise RuntimeError("transport close failed")


@pytest.mark.parametrize("fail_close", [False, True])
def test_terminal_result_waits_for_real_sdk_transport_cleanup(
    tmp_path, monkeypatch, fail_close
):
    # Returning before the client context exits used to abandon Transport.close,
    # reporting finish even when cleanup had not completed or would fail.
    events = []
    transport = TerminalTransport(events, fail_close=fail_close)
    for module in (
        claude_agent_sdk._internal.transport.subprocess_cli,
        claude_agent_sdk._internal.client,
    ):
        monkeypatch.setattr(
            module, "SubprocessCLITransport", lambda **kwargs: transport
        )
    toolkit = FakeToolkit({"_finish": True, "status": "success", "summary": "done"})
    toolkit.cancel_active_and_wait = lambda: events.append("toolkit-cleanup")

    result = make_planner(tmp_path, RecordingSink(), timeout_s=3).solve(
        system_prompt="",
        user_message="finish the task",
        toolkit=toolkit,
        max_turns=3,
    )

    assert toolkit.calls == [("finish", {"status": "success", "summary": "done"})]
    assert events == [
        "sdk-cleanup-started",
        "sdk-cleanup-completed",
        "toolkit-cleanup",
    ]
    assert result.finish_result == {
        "_finish": True,
        "status": "success",
        "summary": "done",
    }
    if fail_close:
        assert result.error_kind == "infrastructure"
        assert result.stop_reason == "cleanup_failed"
        assert "transport close failed" in result.error
    else:
        assert result.error is None
        assert result.stop_reason == "finish"
