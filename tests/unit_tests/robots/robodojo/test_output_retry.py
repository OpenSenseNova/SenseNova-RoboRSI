# Copyright 2026 The RoboRSI Contributors.
# Licensed under the Apache License, Version 2.0.

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from roborsi.planner.provider_capture import ProviderCaptureConfig
from robots.robodojo.agents.submission.output_retry import GuardedCapture
from robots.robodojo.agents.submission.turn_output_retry import classify, output_error

REQUEST = {
    "model": "fixture",
    "input": [],
    "stream": False,
    "tools": [
        {
            "type": "function",
            "name": "act",
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {"type": "integer", "minimum": 1, "maximum": 50}
                },
                "required": ["steps"],
                "additionalProperties": False,
            },
        }
    ],
}


def response(output=None, **patch):
    return {
        "id": "response-fixture",
        "object": "response",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "m1",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "valid"}],
            }
        ]
        if output is None
        else output,
        "usage": {"input_tokens": 10, "output_tokens": 2},
        **patch,
    }


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        [],
        {"id": "r", "status": "incomplete", "incomplete_details": "bad"},
        response([]),
        response([{"type": "reasoning", "encrypted_content": "opaque"}]),
        response(
            [
                {
                    "type": "function_call",
                    "call_id": "c",
                    "name": "unknown",
                    "arguments": "{}",
                }
            ]
        ),
        response(
            [
                {
                    "type": "function_call",
                    "call_id": "c",
                    "name": "act",
                    "arguments": "broken",
                }
            ]
        ),
        response(
            [
                {
                    "type": "function_call",
                    "call_id": "c",
                    "name": "act",
                    "arguments": '{"steps":51}',
                }
            ]
        ),
        response(
            [
                {
                    "type": "function_call",
                    "call_id": "c",
                    "name": "act",
                    "arguments": '{"steps":true}',
                }
            ]
        ),
        response([{"type": "message", "content": "bad"}]),
    ],
)
def test_known_invalid_outputs_regenerate(invalid):
    decision = classify(200, invalid, REQUEST, transport_complete=True)
    assert decision["retryable"] and not decision["valid"] and not decision["unknown"]


def test_compaction_needs_public_summary_and_no_tool_execution():
    request = {**REQUEST, "client_metadata": {"request_kind": "compaction"}}
    assert output_error(request, response([])) == "compaction_public_summary_missing"
    assert output_error(request, response()) is None
    call = {
        "type": "function_call",
        "call_id": "c",
        "name": "act",
        "arguments": '{"steps":1}',
    }
    assert output_error(request, response([call])) == "compaction_contains_tool_call"
    assert output_error(REQUEST, response([call, call])) == "tool_call_id_invalid"


@pytest.mark.parametrize(
    "status,value,complete",
    [
        (403, {"error": {"type": "forbidden"}}, True),
        (200, {"error": {"code": "content_filter"}}, True),
        (200, response([{"type": "message", "content": [{"type": "refusal"}]}]), True),
    ],
)
def test_explicit_rejection_never_regenerates(status, value, complete):
    assert not classify(status, value, REQUEST, transport_complete=complete)[
        "retryable"
    ]


def test_caller_cancellation_does_not_retry_unknown_transport():
    decision = classify(
        200,
        response(),
        REQUEST,
        transport_complete=False,
        transport_error="cancelled_or_disconnected",
    )
    assert not decision["retryable"] and decision["unknown"]


def run_buffer(tmp_path, replies, *, storage="raw", case_deadline=None):
    directory = tmp_path / "case"
    directory.mkdir()
    output = directory / "api"
    output.mkdir()
    capture = GuardedCapture(
        ProviderCaptureConfig("http://fixture.invalid/v1", "fixture"),
        output,
        env={},
        case_dir=directory,
        model="fixture",
        model_label="fixture",
        cohort_root=directory,
        expected_effort="medium",
        enforce_paid_cache=False,
        storage_mode=storage,
        case_deadline=case_deadline,
    )
    calls = []

    class Body(httpx.AsyncByteStream):
        def __init__(self, data):
            self.data = data

        async def __aiter__(self):
            yield self.data

    def handler(request):
        calls.append(request.content)
        status, value = replies[min(len(calls) - 1, len(replies) - 1)]
        if isinstance(value, Exception):
            raise value
        return httpx.Response(
            status,
            headers={
                "content-type": "text/event-stream"
                if isinstance(value, bytes) and value.startswith(b"event:")
                else "application/json"
            },
            stream=Body(
                value if isinstance(value, bytes) else json.dumps(value).encode()
            ),
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            capture._upstream_stream = lambda unused, method, url, **kw: client.stream(
                method, url, **kw
            )
            return await capture._buffered_forward(
                {"path": "/v1/responses", "headers": [], "query_string": b""},
                json.dumps(REQUEST).encode(),
            )

    messages = asyncio.run(run())
    return capture, calls, messages


@pytest.mark.parametrize("storage", ["raw", "training"])
def test_five_invalid_attempts_then_valid_only_delivers_valid_output(tmp_path, storage):
    capture, calls, messages = run_buffer(
        tmp_path,
        [
            (
                200,
                response(
                    [],
                    status="incomplete",
                    incomplete_details={"reason": "max_output_tokens"},
                ),
            )
        ]
        * 5
        + [(200, response())],
        storage=storage,
    )
    assert len(calls) == 6 and len(set(calls)) == 1
    assert len(capture.rows) == 6
    assert [r["completed"] for r in capture.rows] == [False] * 5 + [True]
    assert sum(r["usage"]["output_tokens"] for r in capture.rows) == 12
    payload = b"".join(m.get("body", b"") for m in messages)
    assert json.loads(payload)["output"][0]["content"][0]["text"] == "valid"
    assert not (capture.case / "dispatch_hold.json").exists()
    assert len((capture.case / "output-retries.jsonl").read_text().splitlines()) == 6
    assert len(list(capture.output_dir.glob("*.response.bin"))) >= 5


def test_malformed_tool_output_is_not_committed_or_delivered(tmp_path):
    wrong = response(
        [
            {
                "type": "function_call",
                "call_id": "bad",
                "name": "act",
                "arguments": '{"steps":500}',
            }
        ]
    )
    capture, calls, messages = run_buffer(
        tmp_path, [(200, wrong), (200, response())], storage="training"
    )
    assert len(calls) == 2 and [r["completed"] for r in capture.rows] == [False, True]
    assert b'"steps":500' not in b"".join(m.get("body", b"") for m in messages)


def test_six_invalid_attempts_exhaust_one_shared_budget(tmp_path):
    capture, calls, messages = run_buffer(tmp_path, [(200, response([]))])
    assert len(calls) == 6 and not any(r["completed"] for r in capture.rows)
    assert messages[0]["status"] == 503
    assert (capture.case / "dispatch_hold.json").exists()


@pytest.mark.parametrize("storage", ["raw", "training"])
def test_unknown_upstream_retries_share_budget_and_keep_unknown_usage(
    tmp_path, storage
):
    capture, calls, messages = run_buffer(
        tmp_path,
        [(502, {"error": {"type": "cci_request_timeout", "outcome_unknown": True}})] * 5
        + [(200, response())],
        storage=storage,
    )
    assert len(calls) == 6 and len(set(calls)) == 1
    assert [r["completed"] for r in capture.rows] == [False] * 5 + [True]
    assert [r["usage"] for r in capture.rows[:5]] == [None] * 5
    retries = [
        json.loads(line)
        for line in (capture.case / "output-retries.jsonl").read_text().splitlines()
    ]
    assert all(
        r["unknown"] and not r["budget_reset"] and not r["failed_output_forwarded"]
        for r in retries[:5]
    )
    assert not (capture.case / "dispatch_hold.json").exists()
    assert messages[0]["status"] == 200


def test_unknown_timeouts_exhaust_exactly_five_additional_retries(tmp_path):
    capture, calls, messages = run_buffer(
        tmp_path,
        [(502, {"error": {"type": "cci_request_timeout", "outcome_unknown": True}})],
    )
    assert len(calls) == len(capture.rows) == 6
    assert not any(r["completed"] for r in capture.rows)
    assert messages[0]["status"] == 503


def test_read_timeout_retries_before_any_output_delivery(tmp_path):
    capture, calls, messages = run_buffer(
        tmp_path,
        [(200, httpx.ReadTimeout("fixture timeout")), (200, response())],
    )
    assert len(calls) == 2 and len(set(calls)) == 1
    assert capture.rows[0]["usage"] is None
    assert not capture.rows[0]["completed"] and capture.rows[1]["completed"]
    assert messages[0]["status"] == 200


def test_expired_case_budget_prevents_even_the_first_attempt(tmp_path):
    capture, calls, messages = run_buffer(
        tmp_path, [(200, response())], case_deadline=0
    )
    assert not calls and not capture.rows
    assert messages[0]["status"] == 503


def test_different_output_problems_share_the_same_five_retries(tmp_path):
    bad_call = response(
        [{"type": "function_call", "call_id": "c", "name": "act", "arguments": "wrong"}]
    )
    capture, calls, messages = run_buffer(
        tmp_path,
        [
            (502, {"error": {"type": "cci_request_timeout", "outcome_unknown": True}}),
            (200, response([])),
            (200, b"malformed JSON"),
            (200, bad_call),
            (200, response([{"type": "reasoning", "encrypted_content": "opaque"}])),
            (200, response()),
        ],
    )
    assert len(calls) == 6 and len(set(calls)) == 1
    assert [r["completed"] for r in capture.rows] == [False] * 5 + [True]
    assert messages[0]["status"] == 200


def test_native_tool_free_compaction_does_not_require_robot_declarations(tmp_path):
    from robots.robodojo.agents.submission import agent_track
    from robots.robodojo.agents.submission.execution_profile import RobotCodeModeProfile

    policy = __import__("robots.robodojo.agents.submission", fromlist=["__file__"])
    profile = agent_track.load_profile(
        Path(policy.__file__).parent,
        "cover_blocks",
        tmp_path,
        vla_backend="simate_beta",
    )
    wrapped = RobotCodeModeProfile(
        profile,
        [
            {"name": name, "input_schema": {"type": "object"}}
            for name in profile.expected_tools
        ],
    )
    request = {
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": profile.core}],
            },
            {"type": "additional_tools", "tools": []},
        ],
        "client_metadata": {"request_kind": "compaction"},
    }
    actual, evidence = wrapped.transform(request)
    assert evidence["compact_is_explicitly_tool_free"]
    assert actual["input"][-1]["tools"] == []
    assert profile.expected_tools  # No mutation of the ordinary actor profile.
    request.pop("client_metadata")
    with pytest.raises(ValueError, match="declarations missing"):
        wrapped.transform(request)


@pytest.mark.parametrize("storage", ["raw", "training"])
def test_sse_output_is_validated_before_client_delivery(tmp_path, storage):
    from robots.robodojo.agents.submission.cci_protocol import sse

    invalid = sse(
        response(
            [], status="incomplete", incomplete_details={"reason": "max_output_tokens"}
        )
    )
    valid = sse(response())
    capture, calls, messages = run_buffer(
        tmp_path, [(200, invalid), (200, valid)], storage=storage
    )
    assert len(calls) == 2 and [r["completed"] for r in capture.rows] == [False, True]
    assert b"".join(m.get("body", b"") for m in messages) == valid


def test_missing_usage_preserves_output_without_another_paid_call(tmp_path):
    capture, calls, messages = run_buffer(tmp_path, [(200, response(usage=None))])
    assert len(calls) == 1 and not capture.rows[0]["completed"]
    assert messages[0]["status"] == 503


@pytest.mark.parametrize("unknown", [True, False])
def test_cci_distinguishes_unknown_timeout_from_known_malformed_output(
    tmp_path, monkeypatch, unknown
):
    from robots.robodojo.agents.submission import cci_protocol

    class Reply:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return b"malformed upstream response"

    def upstream(*args, **kwargs):
        if unknown:
            raise TimeoutError("fixture timeout")
        return Reply()

    monkeypatch.setattr(cci_protocol.urllib.request, "urlopen", upstream)
    adapter = cci_protocol.CciAdapter(
        "http://fixture.invalid/v1", "fixture", tmp_path / "wire", save_raw=False
    )
    adapter.start()
    try:
        reply = httpx.post(
            adapter.base_url + "/responses",
            json={"model": "fixture", "input": [], "tools": []},
            trust_env=False,
        )
        assert reply.status_code == 502
        assert reply.json()["error"]["outcome_unknown"] is unknown
        retained = adapter.directory / "000001.invalid-upstream.response.json"
        assert retained.exists() is not unknown
        if not unknown:
            assert retained.read_bytes() == b"malformed upstream response"
            assert (
                json.loads((adapter.directory / "000001.receipt.json").read_text())[
                    "status"
                ]
                == 200
            )
    finally:
        adapter.stop()
