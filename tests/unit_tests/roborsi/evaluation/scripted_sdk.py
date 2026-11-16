# Copyright 2026 The SenseNova-RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script only provider decisions; keep real episode, MCP and Agent services."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace


def install(planner: str, directory: Path, action_prompt: str) -> None:
    """Install child-process SDK fakes with inspectable input and feedback."""
    audit = {"inputs": [], "feedback": [], "closed": False}

    def save():
        (directory / "sdk-audit.json").write_text(json.dumps(audit))

    def feedback(content):
        blocks = [
            part.model_dump() if hasattr(part, "model_dump") else part
            for part in content
        ]
        audit["feedback"].append(blocks)
        save()
        return any(
            part.get("type") == "text" and json.loads(part["text"]).get("_finish")
            for part in blocks
        )

    if planner == "claude_code":
        import claude_agent_sdk as sdk

        def server(*, name, version, tools):
            return {"type": "sdk", "name": name, "tools": tools}

        async def messages(prompt, options):
            if isinstance(prompt, str):
                audit["inputs"].append(prompt)
            else:
                async for message in prompt:
                    audit["inputs"].append(message)
            save()
            tool = next(
                tool
                for tool in options.mcp_servers["roborsi"]["tools"]
                if tool.name == "roborsi_action"
            )
            for step in range(3):
                call_id = f"action-{step}"
                yield sdk.AssistantMessage(
                    model="scripted",
                    content=[
                        sdk.ToolUseBlock(
                            id=call_id,
                            name="mcp__roborsi__roborsi_action",
                            input={"prompt": action_prompt},
                        )
                    ],
                )
                result = await tool.handler({"prompt": action_prompt})
                done = feedback(result["content"])
                yield sdk.UserMessage(
                    content=[
                        sdk.ToolResultBlock(
                            tool_use_id=call_id, content=result["content"]
                        )
                    ]
                )
                if done:
                    return

        class ClaudeSDKClient:
            def __init__(self, *, options):
                assert Path(options.cwd) == directory.resolve()
                self.options = options
                self.stream = None

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                if self.stream is not None:
                    await self.stream.aclose()
                audit["closed"] = True
                save()

            async def query(self, prompt):
                self.stream = messages(prompt, self.options)

            def receive_messages(self):
                assert self.stream is not None
                return self.stream

        sdk.create_sdk_mcp_server = server
        sdk.ClaudeSDKClient = ClaudeSDKClient
        return

    import openai_codex
    from openai_codex.generated.v2_all import ItemCompletedNotification

    async def call_tool(url):
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        async with streamable_http_client(url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(
                    "roborsi_action", {"prompt": action_prompt}
                )

    class Turn:
        def __init__(self, url):
            self.url = url
            self.stopped = False

        def interrupt(self):
            self.stopped = True

        def stream(self):
            for step in range(3):
                if self.stopped:
                    return
                result = asyncio.run(call_tool(self.url))
                done = feedback(result.content)
                yield SimpleNamespace(
                    method="item/completed",
                    payload=ItemCompletedNotification.model_validate(
                        {
                            "completedAtMs": 0,
                            "threadId": "scripted-thread",
                            "turnId": "scripted-turn",
                            "item": {
                                "type": "mcpToolCall",
                                "id": f"action-{step}",
                                "server": "roborsi",
                                "tool": "roborsi_action",
                                "status": "completed",
                                "arguments": {"prompt": action_prompt},
                                "result": result.model_dump(),
                            },
                        }
                    ),
                )
                if done:
                    break
            yield {
                "method": "turn/completed",
                "payload": {"turn": {"status": "completed"}},
            }

    class Thread:
        def __init__(self, url):
            self.url = url

        def turn(self, prompt, **kwargs):
            audit["inputs"].append(
                [
                    asdict(item)
                    if is_dataclass(item)
                    else item.model_dump()
                    if hasattr(item, "model_dump")
                    else item
                    for item in prompt
                ]
                if isinstance(prompt, list)
                else prompt
            )
            save()
            return Turn(self.url)

    class Codex:
        def __init__(self, config):
            assert Path(config.cwd) == directory.resolve()
            overrides = config.config_overrides
            self.url = next(
                json.loads(value.split("=", 1)[1])
                for value in overrides
                if value.startswith("mcp_servers.roborsi.url=")
            )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def thread_start(self, **kwargs):
            return Thread(self.url)

        def close(self):
            audit["closed"] = True
            save()

    openai_codex.Codex = Codex
