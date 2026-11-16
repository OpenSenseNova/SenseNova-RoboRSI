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

"""User policies inspect prior requests and explicitly choose the next markers."""

import asyncio
import json
from dataclasses import replace
from decimal import Decimal

import pytest
from pydantic_ai import BinaryContent, Tool
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelHTTPError, ModelRetry
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from pydantic_ai.usage import RequestUsage

from roborsi.llm.retry import RetryLoggingModel, RetryPolicy
from roborsi.runtime import CacheBreakpoint, ContextResult, RuntimeConfig
from roborsi.runtime.context_engine import recent_turns
from roborsi.runtime.factory import build_runtime_agent
from roborsi.runtime.trace import TraceRecorder


def build(model, engine, *, capabilities=(), factory=False):
    def read() -> str:
        return "observation"

    return build_runtime_agent(
        model=model,
        runtime=RuntimeConfig(
            **{"context_engine_factory" if factory else "context_engine": engine}
        ),
        tools=[Tool(read)],
        capabilities=capabilities,
        system_prompt="fixed",
        max_tokens=100,
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_user_can_keep_move_and_delete_previous_markers(asynchronous):
    states = []
    wire_ids = []

    def policy(ctx, messages, *, cache_state):
        states.append(cache_state)
        if cache_state.previous is None:
            return ContextResult(messages, (CacheBreakpoint(0, 0, 0),))
        assert cache_state.previous.status == "completed"
        assert cache_state.previous.usage.cache_read_tokens == 4
        assert cache_state.previous.usage.cache_write_tokens is None
        matches = cache_state.resolve(messages)
        assert all(match.reusable_prefix for match in matches)
        if len(states) == 2:
            # Drop the task marker and select the latest feedback instead.
            return ContextResult(messages, (CacheBreakpoint(2, 0),))
        assert matches[0].boundary.tool_call_id == "call-1"
        assert matches[0].position == CacheBreakpoint(2, 0)
        # Returning an ordinary list removes all markers.
        return messages

    async def async_policy(ctx, messages, *, cache_state):
        return policy(ctx, messages, cache_state=cache_state)

    def model(messages, info):
        wire_ids.append(
            (info.model_settings or {}).get("roborsi_cache_tool_call_ids", ())
        )
        return ModelResponse(
            parts=[ToolCallPart("read", {}, f"call-{len(wire_ids)}")]
            if len(wire_ids) < 3
            else [TextPart("done")],
            usage=RequestUsage(
                input_tokens=10,
                output_tokens=1,
                cache_read_tokens=4,
                cost=Decimal("0.001"),
            ),
        )

    result = build(
        FunctionModel(model), async_policy if asynchronous else policy
    ).run_sync("task")
    assert result.output == "done"
    assert wire_ids == [(), ("call-1",), ()]
    assert states[0].previous is None
    assert [state.previous.sequence for state in states[1:]] == [1, 2]


def test_relocation_and_prefix_changes_are_visible_to_user():
    calls = 0

    def policy(ctx, messages, *, cache_state):
        if len(messages) < 5:
            return ContextResult(
                messages,
                (
                    CacheBreakpoint(
                        len(messages) - 1, 0, 0 if len(messages) == 1 else None
                    ),
                ),
            )
        original = cache_state.resolve(messages)[0]
        assert original.reusable_prefix
        # Remove the initial task: the tool identity survives, the prefix does not.
        shifted = cache_state.resolve(messages[1:])[0]
        assert shifted.position == CacheBreakpoint(1, 0)
        assert shifted.prefix_unchanged is False
        edited = list(messages)
        edited[0] = ModelRequest(
            parts=[
                UserPromptPart(
                    ["task", BinaryContent(b"changed", media_type="image/png")]
                )
            ]
        )
        assert cache_state.resolve(edited)[0].prefix_unchanged is False
        edited = list(messages)
        edited[2] = replace(
            messages[2],
            parts=[replace(messages[2].parts[0], content="changed observation")],
        )
        assert cache_state.resolve(edited)[0].prefix_unchanged is False
        # A deleted boundary must not accidentally bind to its old array index.
        trimmed = recent_turns(ctx, messages, keep_turns=1)
        assert cache_state.resolve(trimmed)[0].position is None
        return messages

    def model(messages, info):
        nonlocal calls
        calls += 1
        return ModelResponse(
            parts=[ToolCallPart("read", {}, f"call-{calls}")]
            if calls < 3
            else [TextPart("done")]
        )

    build(FunctionModel(model), policy).run_sync(
        ["task", BinaryContent(b"original", media_type="image/png")]
    )


def test_changed_instructions_invalidate_configuration():
    class ChangeInstructions(AbstractCapability):
        async def before_model_request(self, ctx, request_context):
            if len(request_context.messages) > 1:
                parts = request_context.model_request_parameters.instruction_parts
                request_context.model_request_parameters = replace(
                    request_context.model_request_parameters,
                    instruction_parts=[replace(parts[0], content="new instructions")],
                )
            return request_context

    def policy(ctx, messages, *, cache_state):
        if cache_state.previous:
            match = cache_state.resolve(messages)[0]
            assert match.prefix_unchanged is True
            assert match.configuration_unchanged is False
            assert not match.reusable_prefix
        return ContextResult(messages, (CacheBreakpoint(0, 0, 0),))

    def model(messages, info):
        return ModelResponse(
            parts=[ToolCallPart("read", {}, "r")]
            if len(messages) == 1
            else [TextPart("done")]
        )

    build(FunctionModel(model), policy, capabilities=[ChangeInstructions()]).run_sync(
        "task"
    )


def test_retries_are_one_logical_request_and_failure_is_not_a_cache_write():
    attempts = 0
    selected = []

    def policy(ctx, messages, *, cache_state):
        selected.append(cache_state)
        return ContextResult(messages, (CacheBreakpoint(0, 0, 0),))

    def model(messages, info):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ModelHTTPError(503, "test")
        return ModelResponse(
            parts=[ToolCallPart("read", {}, "r")]
            if len(messages) == 1
            else [TextPart("done")]
        )

    retry = RetryLoggingModel(
        FunctionModel(model),
        policy=RetryPolicy(max_retries=1, initial_delay_s=0, max_delay_s=0),
    )
    build(retry, policy).run_sync("task")
    assert attempts == 3
    assert len(selected) == 2
    assert selected[1].previous.sequence == 1
    assert selected[1].previous.status == "completed"
    assert selected[1].previous.usage.cache_write_tokens is None


def test_failed_request_remains_visible_after_error_recovery():
    class Recover(AbstractCapability):
        async def on_model_request_error(self, ctx, *, request_context, error):
            return ModelResponse(parts=[ToolCallPart("read", {}, "recovered")])

    seen = []

    def policy(ctx, messages, *, cache_state):
        seen.append(cache_state)
        return ContextResult(messages, (CacheBreakpoint(0, 0, 0),))

    def model(messages, info):
        if len(messages) == 1:
            raise ModelHTTPError(400, "test")
        return ModelResponse(parts=[TextPart("done")])

    build(FunctionModel(model), policy, capabilities=[Recover()]).run_sync("task")
    assert seen[1].previous.status == "failed"
    assert seen[1].previous.usage is None


def test_concurrent_runs_do_not_share_cache_state():
    histories = []

    def factory():
        states = []
        histories.append(states)

        async def policy(ctx, messages, *, cache_state):
            await asyncio.sleep(0)
            states.append(cache_state)
            return ContextResult(messages, (CacheBreakpoint(0, 0, 0),))

        return policy

    def model(messages, info):
        return ModelResponse(
            parts=[ToolCallPart("read", {}, "same-id")]
            if len(messages) == 1
            else [TextPart("done")]
        )

    agent = build(FunctionModel(model), factory, factory=True)

    async def run():
        await asyncio.gather(agent.run("first task"), agent.run("second task"))

    asyncio.run(run())
    assert len(histories) == 2
    assert all(states[0].previous is None for states in histories)
    assert (
        histories[0][1].previous.boundaries[0].content_id
        != histories[1][1].previous.boundaries[0].content_id
    )


def test_cancellation_is_visible_in_final_cache_record(tmp_path):
    async def scenario():
        started = asyncio.Event()

        async def model(messages, info):
            started.set()
            await asyncio.Event().wait()

        def policy(ctx, messages, *, cache_state):
            return ContextResult(messages, (CacheBreakpoint(0, 0, 0),))

        recorder = TraceRecorder(tmp_path)
        with recorder.activate():
            task = asyncio.create_task(build(FunctionModel(model), policy).run("task"))
            await asyncio.wait_for(started.wait(), timeout=5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        recorder.close()

    asyncio.run(scenario())
    records = [
        event["payload"]
        for line in (tmp_path / "trace/events.jsonl").read_text().splitlines()
        if (event := json.loads(line))["type"] == "context_cache_request"
    ]
    assert len(records) == 1
    assert records[0]["status"] == "cancelled"
    assert records[0]["usage"] is None
    assert len(records[0]["boundaries"]) == 1


@pytest.mark.parametrize("stream_failure", [False, True])
def test_streamed_requests_keep_prior_outcome_and_final_selected_boundaries(
    tmp_path, stream_failure
):
    states = []

    def policy(ctx, messages, *, cache_state):
        states.append(cache_state)
        return ContextResult(messages, (CacheBreakpoint(0, 0, 0),))

    async def stream(messages, info):
        if len(messages) == 1:
            yield {0: DeltaToolCall(name="read", json_args="{}", tool_call_id="r")}
        else:
            yield "done"
            if stream_failure:
                raise ConnectionError("stream interrupted")

    recorder = TraceRecorder(tmp_path)
    agent = build(FunctionModel(stream_function=stream), policy)

    async def run():
        with recorder.activate():
            async with agent.run_stream("task") as result:
                await result.get_output()

    if stream_failure:
        with pytest.raises(ConnectionError, match="stream interrupted"):
            asyncio.run(run())
    else:
        asyncio.run(run())
    recorder.close()
    assert len(states) == 2
    assert states[0].previous is None
    assert states[1].previous.status == "completed"
    assert states[1].previous.sequence == 1
    assert len(states[1].previous.boundaries) == 1
    records = [
        event["payload"]
        for line in (tmp_path / "trace/events.jsonl").read_text().splitlines()
        if (event := json.loads(line))["type"] == "context_cache_request"
    ]
    assert [record["sequence"] for record in records] == [1, 2]
    assert [record["status"] for record in records] == [
        "completed",
        "failed" if stream_failure else "completed",
    ]
    assert all(len(record["boundaries"]) == 1 for record in records)
    if stream_failure:
        assert records[-1]["usage"] is None


def test_response_hook_retry_keeps_the_received_provider_usage():
    class RetryResponse(AbstractCapability):
        async def after_model_request(self, ctx, *, request_context, response):
            if ctx.run_step == 1:
                raise ModelRetry("check again")
            return response

    states = []

    def policy(ctx, messages, *, cache_state):
        states.append(cache_state)
        return ContextResult(messages, (CacheBreakpoint(0, 0, 0),))

    def model(messages, info):
        return ModelResponse(
            parts=[TextPart("done")],
            usage=RequestUsage(input_tokens=10, cache_read_tokens=4),
        )

    assert (
        build(FunctionModel(model), policy, capabilities=[RetryResponse()])
        .run_sync("task")
        .output
        == "done"
    )
    assert len(states) == 2
    assert states[0].previous is None
    assert states[1].previous.status == "completed"
    assert states[1].previous.usage.cache_read_tokens == 4


@pytest.mark.parametrize("reject_step", [1, 2])
def test_rejected_context_does_not_record_an_undispatched_request(
    tmp_path, reject_step
):
    calls = 0

    def policy(ctx, messages, *, cache_state):
        if ctx.run_step == reject_step:
            raise ValueError("context rejected")
        return ContextResult(messages, (CacheBreakpoint(0, 0, 0),))

    def model(messages, info):
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[ToolCallPart("read", {}, "r")])

    recorder = TraceRecorder(tmp_path)
    with recorder.activate(), pytest.raises(ValueError, match="context rejected"):
        build(FunctionModel(model), policy).run_sync("task")
    recorder.close(status="error")
    records = [
        event["payload"]
        for line in (tmp_path / "trace/events.jsonl").read_text().splitlines()
        if (event := json.loads(line))["type"] == "context_cache_request"
    ]
    assert calls == reject_step - 1
    assert len(records) == calls
    assert all(record["status"] == "completed" for record in records)


@pytest.mark.parametrize("configured_child", [False, True])
def test_dynamic_child_has_one_isolated_cache_policy(configured_child):
    from roborsi.runtime import SubAgentConfig

    histories = []

    def factory():
        states = []
        histories.append(states)

        def policy(ctx, messages, *, cache_state):
            states.append(cache_state)
            return ContextResult(messages, (CacheBreakpoint(0, 0, 0),))

        return policy

    def model(messages, info):
        if info.instructions == "child":
            return ModelResponse(
                parts=[ToolCallPart("read_text_file", {}, "read-child")]
                if len(messages) == 1
                else [TextPart("child done")]
            )
        if len(messages) == 1:
            args = {"agent_name": "reader", "task": "child task"}
            if not configured_child:
                args.update(instructions="child", tools=["read_text_file"])
            return ModelResponse(
                parts=[ToolCallPart("delegate_task", args, "delegate")]
            )
        return ModelResponse(parts=[TextPart("root done")])

    def read_text_file() -> str:
        return "observed"

    config = RuntimeConfig(
        dynamic_subagents=True,
        context_engine_factory=factory,
        subagents={
            "reader": SubAgentConfig(instructions="child", tools=("read_text_file",))
        }
        if configured_child
        else {},
    )
    agent = build_runtime_agent(
        model=FunctionModel(model),
        runtime=config,
        tools=[Tool(read_text_file)],
        capabilities=[],
        system_prompt="root",
        max_tokens=100,
    )
    assert agent.run_sync("root task").output == "root done"
    assert len(histories) == 2
    assert [len(states) for states in histories] == [2, 2]
    assert all(states[0].previous is None for states in histories)
    assert (
        histories[0][1].previous.boundaries[0].content_id
        != histories[1][1].previous.boundaries[0].content_id
    )
