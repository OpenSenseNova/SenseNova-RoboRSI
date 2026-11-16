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

"""The shared episode boundary passes media and cancellation to each planner."""

import threading

import pytest
from pydantic_ai import BinaryContent

import roborsi.embodied_agent as module
from roborsi.embodied_agent import EmbodiedAgent, LocalToolSpec
from roborsi.planner.base import PlannerResult
from roborsi.tools.toolkit import ToolResult


@pytest.mark.parametrize("planner", ["codex", "claude_code"])
def test_sdk_episode_preserves_initial_media_and_deferred_feedback(
    tmp_path, monkeypatch, planner
):
    image = BinaryContent(data=b"camera", media_type="image/png")
    received = []

    class Planner:
        def solve(
            self, *, system_prompt, user_message, toolkit, max_turns, cancel_event
        ):
            assert user_message == ["Move", "state zero", image]
            assert system_prompt == "Rules"
            assert isinstance(cancel_event, threading.Event)
            received.append(toolkit.execute_tool("action", {"distance": 1}))
            return PlannerResult(finish_result=received[-1].result)

    monkeypatch.setattr(module, "build_planner", lambda *args, **kwargs: Planner())
    agent = EmbodiedAgent(
        [],
        tmp_path,
        planner=planner,
        local_tools=[LocalToolSpec("action", "Move", {"type": "object"})],
    )
    with agent.start_episode(
        "Move",
        system_prompt="Rules",
        deferred_tools=["action"],
        initial_context=["state zero", image],
    ) as episode:
        call = episode.next_call(timeout=2)
        assert call.arguments == {"distance": 1}
        feedback = ToolResult("action", {"_finish": True, "native": "complete"})
        episode.complete(call, feedback)
        assert episode.wait(timeout=2).finish_result == feedback.result
    assert received == [feedback]


def test_cancel_pending_wakes_deferred_call_before_waiting_for_dispatch(tmp_path):
    bridge = module._ToolBridge()
    toolkit = module._McpToolkit(
        [],
        tmp_path,
        local_tools=[LocalToolSpec("action", "Move", {"type": "object"})],
        bridge=bridge,
        deferred_tools=["action"],
        include_finish=False,
    )
    errors = []

    def invoke():
        try:
            toolkit.execute_tool("action", {})
        except Exception as error:
            errors.append(error)

    worker = threading.Thread(target=invoke, daemon=True)
    worker.start()
    try:
        bridge.calls.get(timeout=2)
        toolkit.cancel_pending()
        worker.join(timeout=2)
        assert not worker.is_alive()
        toolkit.cancel_active_and_wait()
        assert errors[0].kind == "cancelled"
    finally:
        bridge.close()
        worker.join(timeout=2)
        toolkit.close()
