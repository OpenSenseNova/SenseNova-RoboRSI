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

import asyncio
import threading
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import BinaryContent

import roborsi.planner.sdk_episode as sdk_episode
from roborsi.embodied_agent import LocalToolSpec, _McpToolkit, _ToolBridge
from roborsi.planner.sdk_episode import SdkEpisode, prompt_parts
from roborsi.tools.toolkit import ToolResult


class Toolkit:
    def execute_tool(self, name, args):
        return ToolResult(name, {"_finish": True, "status": "success"})

    def cancel_active_and_wait(self):
        raise RuntimeError("cleanup failed")


def test_prior_same_tool_response_cannot_acknowledge_terminal_feedback():
    episode = SdkEpisode(Toolkit(), None)
    terminal = episode.execute_tool("roborsi_action", {})
    episode.acknowledge(
        "roborsi_action",
        {"content": [{"type": "text", "text": '{"value": "previous"}'}]},
    )
    assert not episode.should_shutdown()
    episode.acknowledge("other_tool", {"_finish": True})
    assert not episode.should_shutdown()
    episode.acknowledge("roborsi_action", terminal.content_blocks)
    assert episode.should_shutdown()


def test_cleanup_failure_retains_terminal_evidence_and_error():
    episode = SdkEpisode(Toolkit(), None)

    async def consume():
        terminal = episode.execute_tool("roborsi_action", {})
        episode.acknowledge("roborsi_action", terminal.content_blocks)

    with pytest.raises(RuntimeError, match="cleanup failed"):
        asyncio.run(episode.run(consume(), 1))
    assert episode.finish_result["status"] == "success"
    assert episode.failure.kind == "infrastructure"
    assert episode.failure.reason == "cleanup_failed"


def test_cancel_during_terminal_ack_wait_stops_immediately():
    cancel = threading.Event()
    episode = SdkEpisode(Toolkit(), cancel)
    episode.execute_tool("roborsi_action", {})
    cancel.set()
    assert episode.should_shutdown()
    assert episode.failure.kind == "cancelled"
    assert episode.failure.reason == "closed"
    assert episode.finish_result["status"] == "success"


def test_non_image_binary_input_is_rejected_at_sdk_boundary():
    with pytest.raises(ValueError, match="image"):
        prompt_parts("system", [BinaryContent(data=b"audio", media_type="audio/wav")])


def test_terminal_ack_block_preserves_original_tool_result():
    toolkit = Toolkit()
    original = toolkit.execute_tool("roborsi_action", {})
    toolkit.execute_tool = lambda name, args: original
    episode = SdkEpisode(toolkit, None)
    delivered = episode.execute_tool("roborsi_action", {})
    assert delivered is not original
    assert delivered.result is original.result
    assert delivered.content_blocks[:-1] == original.content_blocks
    assert len(original.content_blocks) == 1
    assert "_roborsi_terminal_ack" not in original.result
    episode.acknowledge("roborsi_action", {"_roborsi_terminal_ack": "previous"})
    assert not episode.acknowledged.is_set()
    episode.acknowledge("roborsi_action", delivered.content_blocks)
    assert episode.acknowledged.is_set()


def test_terminal_ack_deadline_preserves_missing_feedback_cause():
    episode = SdkEpisode(Toolkit(), None)
    episode.execute_tool("roborsi_action", {})
    episode._terminal_at -= 6
    assert episode.should_shutdown()
    assert episode.failure.reason == "terminal_feedback_timeout"
    assert str(episode.failure) == "SDK did not acknowledge terminal tool feedback"


@pytest.mark.parametrize(
    "exit_kind",
    ["sdk_error", "model_end", "sdk_cancel", "caller_cancel", "timeout", "terminal"],
)
def test_run_releases_deferred_tool_on_every_exit(tmp_path, monkeypatch, exit_kind):
    clock = SimpleNamespace(now=0.0)
    if exit_kind == "timeout":
        monkeypatch.setattr(
            sdk_episode, "time", SimpleNamespace(monotonic=lambda: clock.now)
        )
    bridge = _ToolBridge()
    toolkit = _McpToolkit(
        [],
        tmp_path,
        local_tools=[LocalToolSpec("action", "Move", {"type": "object"})],
        bridge=bridge,
        deferred_tools=["action"],
        include_finish=False,
    )
    episode = SdkEpisode(toolkit, None)
    sdk_error = RuntimeError("SDK stream disconnected")
    tool_task = None
    run_task = None
    released_on_exit = False
    forced_release = threading.Event()

    def release_stuck_worker():
        forced_release.set()
        bridge.close()

    async def consume():
        nonlocal tool_task
        tool_task = asyncio.create_task(
            asyncio.to_thread(episode.execute_tool, "action", {})
        )
        call = await asyncio.to_thread(bridge.calls.get, True, 2)
        assert call.name == "action"
        if exit_kind == "sdk_error":
            raise sdk_error
        if exit_kind == "model_end":
            return "SDK completed"
        if exit_kind == "sdk_cancel":
            raise asyncio.CancelledError
        if exit_kind == "caller_cancel":
            run_task.cancel()
        if exit_kind == "timeout":
            # Advance only the episode clock after a deferred call is waiting.
            clock.now = 1.0
        if exit_kind == "terminal":
            bridge.respond(
                call, ToolResult("action", {"_finish": True, "status": "success"})
            )
            terminal = await tool_task
            episode.acknowledge("action", terminal.content_blocks)
        await asyncio.Event().wait()

    async def exercise():
        nonlocal released_on_exit, run_task
        timeout_s = 0.1 if exit_kind == "timeout" else None
        run_task = asyncio.create_task(episode.run(consume(), timeout_s))
        try:
            if exit_kind == "sdk_error":
                with pytest.raises(RuntimeError) as raised:
                    await run_task
                assert raised.value is sdk_error
            elif exit_kind in {"sdk_cancel", "caller_cancel"}:
                with pytest.raises(asyncio.CancelledError):
                    await run_task
            else:
                result = await run_task
                assert result == ("SDK completed" if exit_kind == "model_end" else None)
        finally:
            released_on_exit = bridge.cancelled.is_set()
            # Release the real deferred worker even when the regression fails.
            bridge.close()
            if tool_task is not None:
                await asyncio.wait_for(tool_task, timeout=2)

    watchdog = threading.Timer(5, release_stuck_worker)
    watchdog.start()
    try:
        asyncio.run(exercise())
    finally:
        watchdog.cancel()
        watchdog.join(timeout=2)
        bridge.close()
    assert not forced_release.is_set(), "SDK shutdown needed the test watchdog"
    assert released_on_exit, "SDK exit left a deferred tool waiting for feedback"
    if exit_kind == "timeout":
        assert episode.failure.reason == "timeout"
    else:
        assert episode.failure is None
    if exit_kind == "terminal":
        assert episode.finish_result["status"] == "success"
        assert episode.acknowledged.is_set()


@pytest.mark.parametrize("error_type", [ValueError, asyncio.CancelledError])
def test_cleanup_failure_does_not_replace_sdk_error(error_type):
    episode = SdkEpisode(Toolkit(), None)
    sdk_error = error_type("SDK stream disconnected")

    async def consume():
        raise sdk_error

    with pytest.raises(error_type) as raised:
        asyncio.run(episode.run(consume(), 1))
    propagated = raised.value
    if error_type is asyncio.CancelledError:
        # Python 3.10 wraps cancellation at Task boundaries; the original SDK
        # exception must survive through a chain containing only cancellations.
        while propagated.__context__ is not None:
            assert isinstance(propagated, asyncio.CancelledError)
            propagated = propagated.__context__
    assert propagated is sdk_error
    assert episode.failure is not None
    assert episode.failure.reason == "cleanup_failed"
    assert "cleanup failed" in str(episode.failure)


def test_cleanup_failure_after_model_end_is_reported():
    episode = SdkEpisode(Toolkit(), None)

    async def consume():
        return "SDK completed"

    with pytest.raises(RuntimeError, match="cleanup failed"):
        asyncio.run(episode.run(consume(), 1))
    assert episode.failure.reason == "cleanup_failed"


def test_caller_cancellation_waits_for_client_cleanup(tmp_path):
    episode = SdkEpisode(_McpToolkit([], tmp_path), None)

    async def exercise():
        cleanup_started = asyncio.Event()
        cleanup_allowed = asyncio.Event()
        cleanup_finished = asyncio.Event()

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                cleanup_started.set()
                await cleanup_allowed.wait()
                cleanup_finished.set()

        async def consume():
            async with episode.async_client_context(Client()):
                pass

        run_task = asyncio.create_task(episode.run(consume(), 1))
        try:
            await asyncio.wait_for(cleanup_started.wait(), timeout=2)
            run_task.cancel()
            await asyncio.sleep(0)
            cleanup_allowed.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(run_task, timeout=2)
            assert cleanup_finished.is_set()
        finally:
            cleanup_allowed.set()
            if not run_task.done():
                run_task.cancel()
            await asyncio.wait_for(
                asyncio.gather(run_task, return_exceptions=True), timeout=2
            )

    asyncio.run(exercise())
