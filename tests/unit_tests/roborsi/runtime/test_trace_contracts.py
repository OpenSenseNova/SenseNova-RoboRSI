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

"""Observable runtime traces survive pruning, retries and concurrent agents."""

import asyncio
import json
import queue
from pathlib import Path

import pytest
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage, UsageLimits

from roborsi.llm.retry import RetryLoggingModel, RetryPolicy
from roborsi.runtime.trace import (
    RuntimeTraceCapability,
    TraceConfig,
    TraceRecorder,
    current_trace,
    trace_scope,
)


def _events(root: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (root / "trace/events.jsonl").read_text().splitlines()
    ]


def test_full_trace_keeps_original_binary_and_actual_pruned_input(
    tmp_path: Path,
) -> None:
    def model(messages, info):
        return ModelResponse(
            parts=[TextPart("done")],
            usage=RequestUsage(input_tokens=8, output_tokens=2),
        )

    def prune(messages):
        from dataclasses import replace

        return [
            replace(
                messages[-1], parts=[replace(messages[-1].parts[-1], content="short")]
            )
        ]

    recorder = TraceRecorder(tmp_path)
    agent = Agent(
        RetryLoggingModel(FunctionModel(model)),
        instructions="fixed instructions",
        capabilities=[RuntimeTraceCapability(), ProcessHistory(prune)],
    )
    with recorder.activate():
        assert current_trace() is recorder
        assert (
            agent.run_sync(
                [
                    "secret original",
                    BinaryContent(b"original-image", media_type="image/png"),
                ]
            ).output
            == "done"
        )
    assert current_trace() is None
    recorder.close()
    events = _events(tmp_path)
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
    assert events[0]["type"] == "run_start"
    assert events[-1]["type"] == "run_end"
    request = next(event for event in events if event["type"] == "model_request_start")
    actual = json.loads((tmp_path / request["payload"]["request_ref"]).read_text())
    assert "short" in json.dumps(actual)
    assert "secret original" not in json.dumps(actual)
    assert "fixed instructions" in json.dumps(actual)
    raw = [event for event in events if event["type"] == "message_received"]
    assert any(
        "secret original" in (tmp_path / event["payload"]["message_ref"]).read_text()
        for event in raw
    )
    assert len(list((tmp_path / "trace/content").iterdir())) == 1
    assert (
        next((tmp_path / "trace/content").iterdir()).read_bytes() == b"original-image"
    )
    assert (
        json.loads((tmp_path / "trace/manifest.json").read_text())["status"]
        == "completed"
    )


@pytest.mark.parametrize("mode", ["metadata", "off"])
def test_modes_never_persist_prompt_or_binary(tmp_path: Path, mode: str) -> None:
    recorder = TraceRecorder(tmp_path, TraceConfig(mode=mode))
    agent = Agent(
        RetryLoggingModel(
            FunctionModel(
                lambda messages, info: ModelResponse(
                    parts=[TextPart("sensitive response")]
                )
            )
        ),
        capabilities=[RuntimeTraceCapability()],
    )
    with recorder.activate():
        agent.run_sync(
            [
                "sensitive prompt",
                BinaryContent(b"sensitive binary", media_type="image/png"),
            ]
        )
    recorder.close()
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert b"sensitive" not in path.read_bytes()
    assert not (tmp_path / "trace/requests").exists()
    if mode == "off":
        assert not (tmp_path / "trace").exists()


def test_tool_result_recorded_before_pruning_and_turn_closes_after_tool(
    tmp_path: Path,
) -> None:
    calls = 0

    def model(messages, info):
        nonlocal calls
        calls += 1
        return ModelResponse(
            parts=[ToolCallPart("read", {}, "call-1")]
            if calls == 1
            else [TextPart("done")]
        )

    def read() -> str:
        return "fresh tool result"

    recorder = TraceRecorder(tmp_path)
    agent = Agent(
        RetryLoggingModel(FunctionModel(model)),
        tools=[read],
        capabilities=[RuntimeTraceCapability()],
    )
    with recorder.activate():
        agent.run_sync("go")
    recorder.close()
    events = _events(tmp_path)
    tool_end = next(event for event in events if event["type"] == "tool_end")
    turn_end = next(event for event in events if event["type"] == "turn_end")
    assert tool_end["seq"] < turn_end["seq"]
    assert tool_end["turn_id"] == turn_end["turn_id"]
    assert tool_end["tool_call_id"] == "call-1"
    assert (
        "fresh tool result"
        in (tmp_path / tool_end["payload"]["result_ref"]).read_text()
    )


def test_parallel_children_have_distinct_scopes_and_inherit_parent(
    tmp_path: Path,
) -> None:
    model = RetryLoggingModel(
        FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart("done")]))
    )
    child = Agent(model, name="child", capabilities=[RuntimeTraceCapability()])

    async def delegate() -> str:
        await asyncio.gather(child.run("a"), child.run("b"))
        return "delegated"

    calls = 0

    def root_model(messages, info):
        nonlocal calls
        calls += 1
        return ModelResponse(
            parts=[ToolCallPart("delegate", {}, "delegate-1")]
            if calls == 1
            else [TextPart("done")]
        )

    root = Agent(
        RetryLoggingModel(FunctionModel(root_model)),
        name="root",
        tools=[delegate],
        capabilities=[RuntimeTraceCapability()],
    )
    recorder = TraceRecorder(tmp_path)
    with recorder.activate():
        root.run_sync("start")
    recorder.close()
    events = _events(tmp_path)
    starts = [event for event in events if event["type"] == "agent_start"]
    assert len(starts) == 3
    parent = starts[0]["agent_id"]
    assert {event["parent_agent_id"] for event in starts[1:]} == {parent}
    assert len({event["agent_id"] for event in starts}) == 3
    for event in starts:
        requests = [
            item
            for item in events
            if item["agent_id"] == event["agent_id"]
            and item["type"] == "model_request_start"
        ]
        assert requests
        assert all(item["turn_id"] for item in requests)


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (RuntimeError("broken"), "error"),
        (TimeoutError(), "timeout"),
        (asyncio.CancelledError(), "cancelled"),
    ],
)
def test_exception_closes_agent_and_turn(
    tmp_path: Path, failure: BaseException, expected: str
) -> None:
    async def model(messages, info):
        raise failure

    recorder = TraceRecorder(tmp_path)
    agent = Agent(
        RetryLoggingModel(FunctionModel(model), policy=RetryPolicy(max_retries=0)),
        capabilities=[RuntimeTraceCapability()],
    )
    with recorder.activate(), pytest.raises(type(failure)):
        agent.run_sync("go")
    recorder.close(status=expected)
    events = _events(tmp_path)
    for event_type in (
        "agent_end",
        "turn_end",
        "model_request_end",
        "model_attempt_end",
    ):
        ends = [event for event in events if event["type"] == event_type]
        assert len(ends) == 1
        assert ends[0]["payload"]["status"] == expected


def test_default_retries_three_times_and_usage_is_attempt_owned(tmp_path: Path) -> None:
    calls = 0

    def model(messages, info):
        nonlocal calls
        calls += 1
        if calls <= 3:
            raise ConnectionError("temporary")
        return ModelResponse(
            parts=[TextPart("ok")], usage=RequestUsage(input_tokens=10, output_tokens=2)
        )

    recorder = TraceRecorder(tmp_path)
    policy = RetryPolicy(initial_delay_s=0, max_delay_s=0)
    agent = Agent(
        RetryLoggingModel(FunctionModel(model), policy=policy),
        capabilities=[RuntimeTraceCapability()],
    )
    with recorder.activate(), trace_scope(purpose="context_compression"):
        agent.run_sync("go", usage_limits=UsageLimits(request_limit=1))
    recorder.close()
    events = _events(tmp_path)
    assert calls == 4
    attempts = [event for event in events if event["type"] == "model_attempt_end"]
    assert [event["attempt"] for event in attempts] == [1, 2, 3, 4]
    assert len({event["request_id"] for event in attempts}) == 1
    assert all(event["payload"]["usage"] is None for event in attempts[:3])
    assert attempts[-1]["payload"]["usage"]["input_tokens"] == 10
    assert attempts[-1]["payload"]["usage"]["cache_read_tokens"] is None
    assert attempts[-1]["payload"]["purpose"] == "context_compression"
    assert all(
        "usage" not in event["payload"]
        for event in events
        if event["type"] == "model_request_end"
    )


def test_partial_stream_failure_keeps_usage_chunks_and_is_never_replayed(
    tmp_path: Path,
) -> None:
    calls = 0

    async def stream(messages, info):
        nonlocal calls
        calls += 1
        yield "visible partial response"
        raise ConnectionError("stream broke")

    recorder = TraceRecorder(tmp_path)
    model = RetryLoggingModel(
        FunctionModel(stream_function=stream),
        policy=RetryPolicy(initial_delay_s=0, max_delay_s=0),
        log_path=tmp_path / "llm_errors.jsonl",
    )
    agent = Agent(model, capabilities=[RuntimeTraceCapability()])

    async def run():
        with recorder.activate():
            async with agent.run_stream("go") as result:
                async for _ in result.stream_text():
                    pass

    with pytest.raises(ConnectionError):
        asyncio.run(run())
    recorder.close(status="error")
    events = _events(tmp_path)
    assert calls == 1
    attempt = next(event for event in events if event["type"] == "model_attempt_end")
    assert attempt["payload"]["status"] == "error"
    assert attempt["payload"]["usage"]["output_tokens"] > 0
    response = (tmp_path / attempt["payload"]["response_ref"]).read_text()
    assert "visible partial response" in response
    assert any(event["type"] == "model_response_chunk" for event in events)
    legacy = json.loads((tmp_path / "llm_requests.jsonl").read_text())
    assert legacy["request_id"] == attempt["request_id"]
    assert (
        legacy["usage"]["output_tokens"] == attempt["payload"]["usage"]["output_tokens"]
    )


def test_sink_failure_does_not_skip_tool_cleanup_and_marks_incomplete(
    tmp_path: Path,
) -> None:
    class BrokenSink:
        def emit(self, event):
            if event.type == "tool_start":
                raise OSError("sink unavailable")

    cleaned = []
    calls = 0

    def model(messages, info):
        nonlocal calls
        calls += 1
        return ModelResponse(
            parts=[ToolCallPart("action", {}, "action-1")]
            if calls == 1
            else [TextPart("done")]
        )

    def action() -> str:
        try:
            return "acted"
        finally:
            cleaned.append(True)

    recorder = TraceRecorder(tmp_path, sinks=[BrokenSink()])
    agent = Agent(
        RetryLoggingModel(FunctionModel(model)),
        tools=[action],
        capabilities=[RuntimeTraceCapability()],
    )
    with recorder.activate():
        agent.run_sync("go")
    recorder.close()
    assert cleaned == [True]
    manifest = json.loads((tmp_path / "trace/manifest.json").read_text())
    assert manifest["status"] == "incomplete"
    assert manifest["complete"] is False


def test_shared_request_limit_closes_without_second_provider_request(
    tmp_path: Path,
) -> None:
    calls = 0

    def model(messages, info):
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[ToolCallPart("again", {}, f"call-{calls}")])

    def again() -> str:
        return "continue"

    from pydantic_ai.exceptions import UsageLimitExceeded

    recorder = TraceRecorder(tmp_path)
    agent = Agent(
        RetryLoggingModel(FunctionModel(model)),
        tools=[again],
        capabilities=[RuntimeTraceCapability()],
    )
    with recorder.activate(), pytest.raises(UsageLimitExceeded):
        agent.run_sync("go", usage_limits=UsageLimits(request_limit=1))
    recorder.close(status="limit")
    assert calls == 1
    events = _events(tmp_path)
    assert (
        next(event for event in events if event["type"] == "agent_end")["payload"][
            "status"
        ]
        == "limit"
    )


def test_existing_trace_cannot_be_replaced_or_appended(tmp_path: Path) -> None:
    first = TraceRecorder(tmp_path)
    first.close()
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError, match="trace already exists"):
        TraceRecorder(tmp_path)
    assert {
        path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    } == before


def test_snapshot_write_failure_keeps_model_working_and_marks_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from roborsi.runtime import trace

    write = trace.write_json_atomic

    def fail_message(path, value):
        if Path(path).parent.name == "messages":
            raise OSError("disk unavailable")
        return write(path, value)

    monkeypatch.setattr(trace, "write_json_atomic", fail_message)
    recorder = TraceRecorder(tmp_path)
    agent = Agent(
        RetryLoggingModel(
            FunctionModel(
                lambda messages, info: ModelResponse(parts=[TextPart("done")])
            )
        ),
        capabilities=[RuntimeTraceCapability()],
    )
    with recorder.activate():
        assert agent.run_sync("still execute").output == "done"
    recorder.close()
    assert (
        json.loads((tmp_path / "trace/manifest.json").read_text())["status"]
        == "incomplete"
    )


def test_context_failure_closes_context_without_calling_provider(
    tmp_path: Path,
) -> None:
    def prune(messages):
        raise ValueError("bad context policy")

    recorder = TraceRecorder(tmp_path)
    agent = Agent(
        RetryLoggingModel(
            FunctionModel(
                lambda messages, info: ModelResponse(parts=[TextPart("unreachable")])
            )
        ),
        capabilities=[RuntimeTraceCapability(), ProcessHistory(prune)],
    )
    with recorder.activate(), pytest.raises(ValueError, match="bad context policy"):
        agent.run_sync("task")
    recorder.close(status="error")
    events = _events(tmp_path)
    context_end = [event for event in events if event["type"] == "context_end"]
    assert len(context_end) == 1
    assert context_end[0]["payload"]["status"] == "error"
    assert not any(event["type"] == "model_request_start" for event in events)


def test_final_event_sink_reads_final_manifest_and_media(tmp_path: Path) -> None:
    observed = []

    class Sink:
        def emit(self, event):
            if event.type in {"run_end", "artifact_written"}:
                observed.append(
                    (event, json.loads((tmp_path / "trace/manifest.json").read_text()))
                )

    recorder = TraceRecorder(tmp_path, sinks=[Sink()])
    video = tmp_path / "episode.mp4"
    video.write_bytes(b"video fixture")
    recorder.record_artifact(video)
    recorder.close()
    assert len(observed) == 2
    assert observed[0][1]["media"][0]["artifact_ref"] == "episode.mp4"
    event, manifest = observed[-1]
    assert manifest["status"] == "completed"
    assert manifest["complete"] is True
    assert manifest["last_seq"] == event.seq


@pytest.mark.parametrize(
    ("stop", "status"),
    [("finish", "completed"), ("max_turns", "limit"), ("user_quit", "cancelled")],
)
def test_api_planner_explicit_stop_reason_survives_sdk_iterator_cleanup(
    tmp_path: Path, stop: str, status: str
) -> None:
    from roborsi.dashboard.events import NullDashboardEventSink
    from roborsi.planner.api_loop import ApiAgentLoop
    from roborsi.runtime import RuntimeConfig
    from roborsi.runtime.lifecycle import (
        activate_trace,
        close_toolkit_trace,
        create_trace,
    )
    from roborsi.tools.toolkit import ToolResult

    inputs = queue.Queue()
    name = "finish" if stop == "finish" else "action"
    arguments = {"status": "success", "summary": "done"} if stop == "finish" else {}
    calls = []

    class Toolkit:
        state = None

        def get_tools_spec(self):
            return [
                {
                    "name": name,
                    "description": "Act",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string"},
                            "summary": {"type": "string"},
                        },
                    },
                }
            ]

        def execute_tool(self, tool_name, args):
            calls.append(tool_name)
            if stop == "user_quit":
                inputs.put(None)
            return ToolResult(tool_name, {"ok": True})

        def cancel_active_and_wait(self):
            pass

        def close(self):
            pass

    recorder = create_trace(tmp_path, RuntimeConfig(), NullDashboardEventSink())
    planner = ApiAgentLoop(
        RetryLoggingModel(
            FunctionModel(
                lambda messages, info: ModelResponse(
                    parts=[ToolCallPart(name, arguments, "call-1")]
                )
            )
        ),
        dashboard_events=NullDashboardEventSink(),
    )
    toolkit = Toolkit()
    with activate_trace(recorder):
        result = planner.solve(
            system_prompt="Act",
            user_message="go",
            toolkit=toolkit,
            max_turns=1 if stop == "max_turns" else 10,
            input_queue=inputs if stop == "user_quit" else None,
        )
    close_toolkit_trace(toolkit, recorder, error=result.error)
    assert calls == [name]
    assert result.error is None
    events = _events(tmp_path)
    for kind in ("turn_end", "agent_end", "run_end"):
        final = next(event for event in reversed(events) if event["type"] == kind)
        assert final["payload"]["status"] == status
    assert (
        json.loads((tmp_path / "trace/manifest.json").read_text())["status"] == status
    )
