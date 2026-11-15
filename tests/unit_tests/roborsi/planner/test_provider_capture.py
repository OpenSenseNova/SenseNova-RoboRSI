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

"""Provider capture checks use real local HTTP and streamed upstream bodies."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import httpx
import pytest

from roborsi.planner.provider_capture import (
    ProviderCapture,
    ProviderCaptureConfig,
    parse_response_body,
    response_is_complete,
)


@contextmanager
def upstream(
    body: bytes,
    *,
    content_type="application/json",
    status=200,
    parts=None,
    claimed_length=None,
    bind_host="127.0.0.1",
):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_CONNECT(self):
            requests.append((self.path, dict(self.headers), b""))
            self.send_response(502)
            self.end_headers()

        def do_POST(self):
            request = self.rfile.read(int(self.headers["content-length"]))
            requests.append((self.path, dict(self.headers), request))
            self.send_response(status)
            if content_type is not None:
                self.send_header("Content-Type", content_type)
            self.send_header(
                "Content-Length",
                str(len(body) if claimed_length is None else claimed_length),
            )
            self.end_headers()
            for part in parts or [body]:
                self.wfile.write(part)
                self.wfile.flush()

        def log_message(self, *_args):
            pass

    class Server(ThreadingHTTPServer):
        address_family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET

    server = Server((bind_host, 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url_host = f"[{bind_host}]" if ":" in bind_host else bind_host
        yield f"http://{url_host}:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def response_body(encrypted="opaque+/=世界"):
    return json.dumps(
        {
            "id": "resp_1",
            "status": "completed",
            "output": [
                {"type": "reasoning", "id": "rs_1", "encrypted_content": encrypted}
            ],
        },
        ensure_ascii=False,
    ).encode()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def clean_proxy_environment(monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)


@pytest.mark.parametrize("streaming", [False, True])
def test_environment_proxy_preserves_response_bytes_and_opaque_reasoning(
    tmp_path, monkeypatch, clean_proxy_environment, streaming
):
    body = response_body()
    content_type = "application/json"
    if streaming:
        body = (
            b"data: "
            + json.dumps(
                {"type": "response.completed", "response": json.loads(body)},
                ensure_ascii=False,
            ).encode()
            + b"\n\ndata: [DONE]\n\n"
        )
        content_type = "text/event-stream"
    with upstream(body, content_type=content_type) as (proxy_url, requests):
        authenticated_proxy = proxy_url.removesuffix("/v1").replace(
            "://", "://proxy-user:PROXY-SECRET@"
        )
        monkeypatch.setenv("HTTP_PROXY", authenticated_proxy)
        environment = dict(os.environ)
        with ProviderCapture(
            ProviderCaptureConfig("http://127.0.0.1:1/v1", "custom"), tmp_path
        ) as capture:
            response = httpx.post(
                capture.base_url + "/responses",
                json={"model": "original"},
                trust_env=False,
            )
        assert response.status_code == 200
        assert response.content == body
        assert requests[0][0] == "http://127.0.0.1:1/v1/responses"
        headers = {key.lower(): value for key, value in requests[0][1].items()}
        assert headers["proxy-authorization"].startswith("Basic ")
        assert dict(os.environ) == environment
    assert (tmp_path / "000001.response.bin").read_bytes() == body
    assert rows(tmp_path / "reasoning.jsonl")[0]["item"]["encrypted_content"] == (
        "opaque+/=世界"
    )
    assert capture.summary["complete_responses"] == 1
    for path in tmp_path.iterdir():
        assert b"PROXY-SECRET" not in path.read_bytes()
        assert b"proxy-user" not in path.read_bytes()
        assert headers["proxy-authorization"].encode() not in path.read_bytes()


@pytest.mark.parametrize(
    "proxy_key", ["HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"]
)
def test_explicit_proxy_environment_is_snapshotted(
    tmp_path, monkeypatch, clean_proxy_environment, proxy_key
):
    with upstream(response_body()) as (proxy_url, requests):
        env = {proxy_key: proxy_url.removesuffix("/v1")}
        capture = ProviderCapture(
            ProviderCaptureConfig("http://127.0.0.1:1/v1", "custom"),
            tmp_path,
            env=env,
        )
        env[proxy_key] = "http://127.0.0.1:2"
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:2")
        with capture:
            response = httpx.post(
                capture.base_url + "/responses", json={}, trust_env=False
            )
        assert response.status_code == 200
        assert response.content == response_body()
        assert len(requests) == 1


def test_explicit_lowercase_proxy_overrides_uppercase(tmp_path):
    with upstream(response_body()) as (proxy_url, requests):
        with ProviderCapture(
            ProviderCaptureConfig("http://127.0.0.1:1/v1", "custom"),
            tmp_path,
            env={"HTTP_PROXY": "http://127.0.0.1:2", "http_proxy": proxy_url},
        ) as capture:
            response = httpx.post(
                capture.base_url + "/responses", json={}, trust_env=False
            )
    assert response.content == response_body()
    assert len(requests) == 1


@pytest.mark.parametrize("proxy_key", ["HTTPS_PROXY", "https_proxy", "ALL_PROXY"])
def test_https_proxy_receives_connect_for_fixed_upstream(tmp_path, proxy_key):
    with upstream(response_body()) as (proxy_url, requests):
        with ProviderCapture(
            ProviderCaptureConfig("https://127.0.0.1:1/v1", "custom"),
            tmp_path,
            env={proxy_key: proxy_url.removesuffix("/v1")},
        ) as capture:
            response = httpx.post(
                capture.base_url + "/responses", json={}, trust_env=False
            )
    assert response.status_code == 502
    assert [request[0] for request in requests] == ["127.0.0.1:1"]
    assert capture.summary["records"][0]["transport_error"] == "ProxyError"


@pytest.mark.parametrize(
    "bypass",
    [
        {"NO_PROXY": "127.0.0.1"},
        {"no_proxy": "127.0.0.1", "NO_PROXY": "elsewhere.invalid"},
        {"NO_PROXY": "*"},
        {"NO_PROXY": "<authority>"},
        {"http_proxy": ""},
    ],
)
def test_no_proxy_and_explicit_empty_proxy_bypass_upstream_proxy(tmp_path, bypass):
    with upstream(response_body("direct")) as (url, direct_requests):
        bypass = {
            name: value.replace("<authority>", urlsplit(url).netloc)
            for name, value in bypass.items()
        }
        with upstream(response_body("wrong-proxy")) as (proxy_url, proxy_requests):
            with ProviderCapture(
                ProviderCaptureConfig(url, "custom"),
                tmp_path,
                env={"HTTP_PROXY": proxy_url.removesuffix("/v1"), **bypass},
            ) as capture:
                response = httpx.post(
                    capture.base_url + "/responses", json={}, trust_env=False
                )
    assert response.content == response_body("direct")
    assert len(direct_requests) == 1
    assert proxy_requests == []


@pytest.mark.parametrize(
    "bypass", [{"NO_PROXY": "127.0.0.1:1"}, {"NO_PROXY": "*", "no_proxy": ""}]
)
def test_nonmatching_or_cleared_no_proxy_keeps_selected_proxy(tmp_path, bypass):
    with upstream(response_body("wrong-direct")) as (url, direct_requests):
        with upstream(response_body("proxy")) as (proxy_url, proxy_requests):
            with ProviderCapture(
                ProviderCaptureConfig(url, "custom"),
                tmp_path,
                env={"HTTP_PROXY": proxy_url.removesuffix("/v1"), **bypass},
            ) as capture:
                response = httpx.post(
                    capture.base_url + "/responses", json={}, trust_env=False
                )
    assert response.content == response_body("proxy")
    assert len(proxy_requests) == 1
    assert direct_requests == []


@pytest.mark.parametrize(
    "no_proxy,bypass",
    [("::1", True), ("[::1]", True), ("[::1]:<port>", True), ("[::1]:1", False)],
)
def test_ipv6_no_proxy_matches_literal_and_optional_port(tmp_path, no_proxy, bypass):
    with upstream(response_body("direct"), bind_host="::1") as (url, direct_requests):
        with upstream(response_body("proxy")) as (proxy_url, proxy_requests):
            with ProviderCapture(
                ProviderCaptureConfig(url, "custom"),
                tmp_path,
                env={
                    "HTTP_PROXY": proxy_url.removesuffix("/v1"),
                    "NO_PROXY": no_proxy.replace("<port>", str(urlsplit(url).port)),
                },
            ) as capture:
                response = httpx.post(
                    capture.base_url + "/responses", json={}, trust_env=False
                )
    assert response.content == response_body("direct" if bypass else "proxy")
    assert len(direct_requests) == (1 if bypass else 0)
    assert len(proxy_requests) == (0 if bypass else 1)


def test_explicit_empty_environment_does_not_inherit_parent_proxy(
    tmp_path, monkeypatch, clean_proxy_environment
):
    with upstream(response_body("direct")) as (url, direct_requests):
        with upstream(response_body("wrong-proxy")) as (proxy_url, proxy_requests):
            monkeypatch.setenv("HTTP_PROXY", proxy_url.removesuffix("/v1"))
            with ProviderCapture(
                ProviderCaptureConfig(url, "custom"), tmp_path, env={}
            ) as capture:
                response = httpx.post(
                    capture.base_url + "/responses", json={}, trust_env=False
                )
    assert response.content == response_body("direct")
    assert len(direct_requests) == 1
    assert proxy_requests == []


def test_proxy_connection_failure_does_not_persist_proxy_credentials(tmp_path):
    with ProviderCapture(
        ProviderCaptureConfig("http://127.0.0.1:2/v1", "custom"),
        tmp_path,
        env={"HTTP_PROXY": "http://proxy-user:PROXY-SECRET@127.0.0.1:1"},
    ) as capture:
        response = httpx.post(capture.base_url + "/responses", json={}, trust_env=False)
    assert response.status_code == 502
    assert capture.summary["records"][0]["transport_error"] == "ConnectError"
    assert b"PROXY-SECRET" not in response.content
    for path in tmp_path.iterdir():
        assert b"PROXY-SECRET" not in path.read_bytes()
        assert b"proxy-user" not in path.read_bytes()


def test_transparent_json_auth_and_opaque_reasoning_are_preserved(tmp_path):
    body = response_body()
    request = b'{ "model":"original", "store": true, "reasoning":{"effort":"max"}}'
    with upstream(body) as (url, requests):
        with ProviderCapture(ProviderCaptureConfig(url, "custom"), tmp_path) as capture:
            response = httpx.post(
                capture.base_url + "/responses",
                content=request,
                headers={
                    "Authorization": "Bearer TOP-SECRET",
                    "Cookie": "SECRET-COOKIE",
                },
                trust_env=False,
            )
            assert response.status_code == 200
            assert response.content == body
        assert requests[0][0] == "/v1/responses"
        assert requests[0][2] == request
        headers = {k.lower(): v for k, v in requests[0][1].items()}
        assert headers["authorization"] == "Bearer TOP-SECRET"
    assert (tmp_path / "000001.response.bin").read_bytes() == body
    saved = rows(tmp_path / "reasoning.jsonl")
    assert saved[0]["item"]["encrypted_content"] == "opaque+/=世界"
    assert capture.summary["nonempty_encrypted_items"] == 1
    assert capture.summary["complete_responses"] == 1
    for path in tmp_path.iterdir():
        assert b"TOP-SECRET" not in path.read_bytes()
        assert b"SECRET-COOKIE" not in path.read_bytes()


def test_explicit_mutation_records_original_and_forwarded_requests(tmp_path):
    request = {
        "model": "original",
        "reasoning": {"effort": "max"},
        "prompt_cache_key": "episode-7",
        "include": ["other.field"],
        "store": True,
    }
    with upstream(response_body()) as (url, requests):
        with ProviderCapture(
            ProviderCaptureConfig(url, "custom", request_encrypted_content=True),
            tmp_path,
        ) as capture:
            assert httpx.post(capture.base_url + "/responses", json=request).is_success
    forwarded = json.loads(requests[0][2])
    assert forwarded == {
        **request,
        "store": False,
        "include": ["other.field", "reasoning.encrypted_content"],
    }
    saved = json.loads((tmp_path / "000001.request.json").read_text())
    assert saved["original"] == request
    assert saved["forwarded"] == forwarded
    assert saved["mutated"] is True


def test_sse_split_events_deduplicate_without_changing_raw_body(tmp_path):
    item = json.loads(response_body())["output"][0]
    events = [
        {"type": "response.created", "response": {"id": "resp_1"}},
        {"type": "response.output_item.done", "item": item, "output_index": 0},
        {"type": "response.completed", "response": json.loads(response_body())},
    ]
    body = b"".join(b"data: " + json.dumps(e).encode() + b"\r\n\r\n" for e in events)
    body += b"data: [DONE]\r\n\r\n"
    parts = [body[:7], body[7:51], body[51:74], body[74:]]
    with upstream(body, content_type="text/event-stream", parts=parts) as (url, _):
        with ProviderCapture(ProviderCaptureConfig(url, "custom"), tmp_path) as capture:
            assert httpx.post(capture.base_url + "/responses", json={}).content == body
    assert (tmp_path / "000001.response.bin").read_bytes() == body
    assert len(rows(tmp_path / "reasoning.jsonl")) == 1
    assert capture.summary["nonempty_encrypted_items"] == 1
    assert capture.summary["complete_responses"] == 1


@pytest.mark.parametrize("content_type", [None, ""])
def test_sse_without_content_type_preserves_reasoning_and_completion(
    tmp_path, content_type
):
    body = (
        b": keepalive\r\nevent: response.completed\r\ndata: "
        + json.dumps(
            {"type": "response.completed", "response": json.loads(response_body())}
        ).encode()
        + b"\r\n\r\n"
    )
    with upstream(body, content_type=content_type) as (url, _):
        with ProviderCapture(ProviderCaptureConfig(url, "custom"), tmp_path) as capture:
            assert httpx.post(capture.base_url + "/responses", json={}).content == body
    record = capture.summary["records"][0]
    assert record["content_type"] == ""
    assert record["transport_complete"] is True
    assert record["complete"] is True
    assert record["response_format"] == "sse"
    assert (tmp_path / "000001.response.bin").read_bytes() == body
    assert rows(tmp_path / "reasoning.jsonl")[0]["item"]["encrypted_content"] == (
        "opaque+/=世界"
    )


@pytest.mark.parametrize("content_type", [None, "text/event-stream"])
@pytest.mark.parametrize(
    "body_kind,complete",
    [
        ("completed", True),
        ("item_only", False),
        ("truncated", False),
        ("malformed", False),
        ("conflicting", False),
    ],
)
def test_client_close_after_terminal_event_preserves_content_and_transport_state(
    tmp_path, content_type, body_kind, complete
):
    response = json.loads(response_body())
    event = {"type": "response.completed", "response": response}
    if body_kind == "item_only":
        event = {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": response["output"][0],
        }
    body = b"data: " + json.dumps(event).encode() + b"\n\n"
    if body_kind == "truncated":
        body = body[:-1]
    if body_kind == "malformed":
        body += b"data: invalid-json\n\n"
    if body_kind == "conflicting":
        body = b'data: {"type":"response.created","response":{"id":"other"}}\n\n' + body
    released = threading.Event()

    def parts():
        yield body
        assert released.wait(10)

    with upstream(
        body,
        content_type=content_type,
        parts=parts(),
        claimed_length=len(body) + 100,
    ) as (url, _):
        capture = ProviderCapture(ProviderCaptureConfig(url, "custom"), tmp_path)
        capture.start()
        try:
            with httpx.stream(
                "POST", capture.base_url + "/responses", json={}, timeout=3
            ) as stream:
                assert next(stream.iter_raw()) == body
                assert not released.is_set()
            capture.stop(timeout_s=2)
        finally:
            released.set()
            capture.stop()
    record = capture.summary["records"][0]
    assert record["transport_complete"] is False
    assert record["transport_error"] == "cancelled_or_disconnected"
    assert record["complete"] is complete
    assert capture.summary["complete_responses"] == int(complete)
    assert (tmp_path / "000001.response.bin").read_bytes() == body
    assert rows(tmp_path / "reasoning.jsonl")[0]["complete"] is complete
    assert rows(tmp_path / "reasoning.jsonl")[0]["item"]["encrypted_content"] == (
        "opaque+/=世界"
    )


@pytest.mark.parametrize("content_type", ["application/json", "text/plain"])
def test_explicit_non_sse_content_type_is_not_overridden(content_type):
    body = (
        b'data: {"type":"response.completed","response":' + response_body() + b"}\n\n"
    )
    parsed = parse_response_body(body, content_type, include_response=True)
    assert parsed["parse_status"] == "malformed"
    assert not response_is_complete(
        parsed,
        status_code=200,
        transport_complete=False,
        transport_error="cancelled_or_disconnected",
    )


@pytest.mark.parametrize(
    "first_status,last_status,changed_output,conflict",
    [
        ("completed", "completed", True, True),
        ("completed", "failed", False, True),
        ("failed", "completed", False, True),
        ("completed", "completed", False, False),
    ],
)
def test_strict_sse_rejects_conflicting_terminal_responses(
    first_status, last_status, changed_output, conflict
):
    first = {**json.loads(response_body()), "status": first_status}
    last = {**json.loads(response_body()), "status": last_status}
    if changed_output:
        last["output"][0]["encrypted_content"] = "different-ciphertext"
    body = b"".join(
        b"data: "
        + json.dumps(
            {"type": "response." + response["status"], "response": response}
        ).encode()
        + b"\n\n"
        for response in (first, last)
    )
    parsed = parse_response_body(body, "", include_response=True)
    assert parsed["parse_status"] == ("malformed" if conflict else "parsed")
    assert (
        "Conflicting terminal responses" in parsed["response_conflicts"]
    ) is conflict
    assert response_is_complete(
        parsed,
        status_code=200,
        transport_complete=False,
        transport_error="cancelled_or_disconnected",
    ) is (not conflict)
    assert "response_conflicts" not in parse_response_body(body, "")


@pytest.mark.parametrize(
    "streaming,status,transport_error,complete",
    [
        (True, 200, "cancelled_or_disconnected", True),
        (False, 200, "cancelled_or_disconnected", False),
        (True, 429, "cancelled_or_disconnected", False),
        (True, 200, "RemoteProtocolError", False),
        (True, 200, "ReadError", False),
        (True, 200, None, False),
    ],
)
def test_terminal_event_exception_only_applies_to_successful_sse_cancellation(
    streaming, status, transport_error, complete
):
    body = response_body()
    if streaming:
        body = b'data: {"type":"response.completed","response":' + body + b"}\n\n"
    parsed = parse_response_body(body, "", include_response=True)
    assert parsed["parse_status"] == "parsed"
    assert (
        response_is_complete(
            parsed,
            status_code=status,
            transport_complete=False,
            transport_error=transport_error,
        )
        is complete
    )


@pytest.mark.parametrize(
    "encrypted,state", [(None, "absent"), ("", "empty"), (42, "malformed")]
)
def test_encrypted_field_states_are_distinct(tmp_path, encrypted, state):
    with upstream(response_body(encrypted)) as (url, _):
        with ProviderCapture(ProviderCaptureConfig(url, "custom"), tmp_path) as capture:
            httpx.post(capture.base_url + "/responses", json={})
    assert rows(tmp_path / "reasoning.jsonl")[0]["encrypted_content_state"] == state
    assert capture.summary["nonempty_encrypted_items"] == 0


def test_concurrent_requests_have_distinct_records_and_no_cross_request_dedup(tmp_path):
    with upstream(response_body()) as (url, _):
        with ProviderCapture(ProviderCaptureConfig(url, "custom"), tmp_path) as capture:
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(
                    pool.map(
                        lambda _: httpx.post(capture.base_url + "/responses", json={}),
                        range(4),
                    )
                )
            assert all(result.is_success for result in results)
    assert len(list(tmp_path.glob("*.response.bin"))) == 4
    assert len(rows(tmp_path / "reasoning.jsonl")) == 4
    assert capture.summary["requests"] == 4


@pytest.mark.parametrize(
    "path",
    ["/other", "/v1/other?redirect=http://example.com", "/v1/responses/unknown"],
)
def test_proxy_rejects_unconfigured_routes(tmp_path, path):
    with upstream(response_body()) as (url, requests):
        with ProviderCapture(ProviderCaptureConfig(url, "custom"), tmp_path) as capture:
            origin = capture.base_url.removesuffix("/v1")
            assert httpx.post(origin + path, json={}).status_code == 404
    assert requests == []


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/api",
        "http://a:b@host/v1",
        "http://host/v1?q=a",
        "http://host/v1#fragment",
    ],
)
def test_config_rejects_unsafe_target_urls(url):
    with pytest.raises(ValueError):
        ProviderCaptureConfig.from_dict(
            {"upstream_base_url": url, "model_provider": "p"}
        )


def test_http_error_and_incomplete_sse_are_not_complete(tmp_path):
    body = b'data: {"type":"response.created","response":{"id":"r"}}\n\n'
    with upstream(body, content_type="text/event-stream", status=429) as (url, _):
        with ProviderCapture(ProviderCaptureConfig(url, "custom"), tmp_path) as capture:
            assert (
                httpx.post(capture.base_url + "/responses", json={}).status_code == 429
            )
    assert capture.summary["complete_responses"] == 0
    assert capture.summary["http_errors"] == 1
    assert capture.summary["incomplete_responses"] == 1


def test_upstream_truncation_preserves_partial_bytes_and_failure(tmp_path):
    body = b'data: {"type":"response.output_item.done","item":{"type":"reasoning","id":"r","encrypted_content":"opaque"}}\n\n'
    with upstream(
        body, content_type="text/event-stream", claimed_length=len(body) + 50
    ) as (url, _):
        with ProviderCapture(ProviderCaptureConfig(url, "custom"), tmp_path) as capture:
            with pytest.raises(httpx.RemoteProtocolError):
                httpx.post(capture.base_url + "/responses", json={})
    assert (tmp_path / "000001.response.bin").read_bytes() == body
    assert rows(tmp_path / "reasoning.jsonl")[0]["complete"] is False
    assert capture.summary["records"][0]["transport_error"] == "RemoteProtocolError"
    assert capture.summary["incomplete_responses"] == 1


@pytest.mark.parametrize("stop_first", [False, True])
def test_streams_before_upstream_completion_and_cancellation_is_bounded(
    tmp_path, stop_first
):
    first = b'data: {"type":"response.created","response":{"id":"r"}}\n\n'
    released = threading.Event()

    def parts():
        yield first
        assert released.wait(10)

    with upstream(
        first,
        content_type="text/event-stream",
        parts=parts(),
        claimed_length=len(first) + 100,
    ) as (url, _):
        capture = ProviderCapture(ProviderCaptureConfig(url, "custom"), tmp_path)
        capture.start()
        try:
            with httpx.stream(
                "POST", capture.base_url + "/responses", json={}, timeout=3
            ) as response:
                assert next(response.iter_raw()) == first
                # This byte reached the caller while the upstream is still waiting.
                assert not released.is_set()
                started = time.monotonic()
                if stop_first:
                    capture.stop(timeout_s=2)
            capture.stop(timeout_s=2)
            assert time.monotonic() - started < 2.5
        finally:
            released.set()
            capture.stop()
    assert capture.summary["incomplete_responses"] == 1
    assert (
        capture.summary["records"][0]["transport_error"] == "cancelled_or_disconnected"
    )


def test_no_created_event_still_deduplicates_completed_item():
    response = json.loads(response_body())
    events = [
        {
            "type": "response.output_item.done",
            "item": response["output"][0],
            "output_index": 0,
        },
        {"type": "response.completed", "response": response},
    ]
    body = b"".join(
        b"data: " + json.dumps(event).encode() + b"\n\n" for event in events
    )
    parsed = parse_response_body(body, "text/event-stream")
    assert len(parsed["reasoning_items"]) == 1
    assert parsed["reasoning_items"][0]["response_id"] == "resp_1"


def test_malformed_include_is_forwarded_unchanged_without_mutation(tmp_path):
    body = b'{"include":42}'
    with upstream(b'{"error":"invalid include"}', status=400) as (url, requests):
        with ProviderCapture(ProviderCaptureConfig(url, "custom"), tmp_path) as capture:
            assert (
                httpx.post(capture.base_url + "/responses", content=body).status_code
                == 400
            )
    assert requests[0][2] == body


def test_malformed_output_does_not_count_as_complete():
    parsed = parse_response_body(
        b'{"status":"completed","output":{}}', "application/json"
    )
    assert parsed["parse_status"] == "malformed"


def test_done_item_encryption_is_retained_when_final_output_omits_field():
    response = json.loads(response_body())
    final = {**response, "output": [{"id": "rs_1", "type": "reasoning"}]}
    events = [
        {"type": "response.output_item.done", "item": response["output"][0]},
        {"type": "response.completed", "response": final},
    ]
    body = b"".join(
        b"data: " + json.dumps(event).encode() + b"\n\n" for event in events
    )
    parsed = parse_response_body(body, "text/event-stream")
    assert len(parsed["reasoning_items"]) == 1
    assert parsed["reasoning_items"][0]["item"]["encrypted_content"] == "opaque+/=世界"


def test_request_credential_fields_are_redacted_only_in_artifacts(tmp_path):
    request = {"model": "original", "metadata": {"API-Key": "BODY-SECRET"}}
    with upstream(response_body()) as (url, requests):
        with ProviderCapture(ProviderCaptureConfig(url, "custom"), tmp_path) as capture:
            assert httpx.post(capture.base_url + "/responses", json=request).is_success
    assert json.loads(requests[0][2]) == request
    saved = json.loads((tmp_path / "000001.request.json").read_text())
    assert saved["original"]["metadata"]["API-Key"] == "[REDACTED]"
    assert "BODY-SECRET" not in (tmp_path / "000001.request.json").read_text()


def test_refused_output_directory_is_not_changed_during_cleanup(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text('{"existing":"experiment"}')
    capture = ProviderCapture(
        ProviderCaptureConfig("http://localhost/v1", "p"), tmp_path
    )
    with pytest.raises(ValueError, match="empty"):
        capture.start()
    capture.stop()
    assert summary.read_text() == '{"existing":"experiment"}'


@pytest.mark.parametrize("request_encrypted_content", [False, True])
@pytest.mark.parametrize("tagged_response", [False, True])
def test_compaction_forwards_unchanged_and_preserves_ciphertext(
    tmp_path, request_encrypted_content, tagged_response
):
    request = b'{ "model": "original", "input": [], "instructions": "compact" }'
    payload = {
        "output": [
            {
                "id": "cmp_1",
                "type": "compaction",
                "encrypted_content": "opaque-compaction+/=世界",
            }
        ],
    }
    if tagged_response:
        payload.update(id="resp_compact_1", object="response.compaction")
    body = json.dumps(payload, ensure_ascii=False).encode()
    with upstream(body) as (url, requests):
        config = ProviderCaptureConfig(url, "custom", request_encrypted_content)
        with ProviderCapture(config, tmp_path) as capture:
            response = httpx.post(
                capture.base_url + "/responses/compact", content=request
            )
            assert response.status_code == 200
            assert response.content == body
    assert requests[0][0] == "/v1/responses/compact"
    assert requests[0][2] == request
    assert (tmp_path / "000001.response.bin").read_bytes() == body
    saved = json.loads((tmp_path / "000001.request.json").read_text())
    assert saved["endpoint"] == "/responses/compact"
    assert saved["mutated"] is False
    assert saved["original"] == saved["forwarded"] == json.loads(request)
    assert saved["requested_encrypted_content"] is False
    assert capture.summary["records"][0]["endpoint"] == "/responses/compact"
    assert capture.summary["records"][0]["response_kind"] == "compaction"
    assert capture.summary["complete_responses"] == 1
    assert capture.summary["reasoning_items"] == 0
    assert capture.summary["nonempty_encrypted_items"] == 0
    assert rows(tmp_path / "reasoning.jsonl") == []


@pytest.mark.parametrize(
    "body",
    [b'{"object":"response.compaction","output":{}}', b'{"error":"invalid"}'],
)
def test_invalid_compaction_response_is_not_complete(tmp_path, body):
    with upstream(body) as (url, _):
        with ProviderCapture(ProviderCaptureConfig(url, "custom"), tmp_path) as capture:
            assert (
                httpx.post(capture.base_url + "/responses/compact", json={}).status_code
                == 200
            )
    assert capture.summary["complete_responses"] == 0
    assert capture.summary["records"][0]["parse_status"] == "malformed"


@pytest.mark.parametrize("endpoint", ["/responses", "/responses/compact"])
def test_provider_query_bytes_reach_only_fixed_upstream_and_are_not_dumped(
    tmp_path, endpoint
):
    query = b"api-version=2026-01-01&scope=a%2fb%20c&token=QUERY-SECRET&redirect=http%3A%2F%2Fexample.com"
    with upstream(response_body()) as (url, requests):
        with ProviderCapture(ProviderCaptureConfig(url, "custom"), tmp_path) as capture:
            target = httpx.URL(capture.base_url + endpoint).copy_with(query=query)
            assert httpx.post(target, json={}).status_code == 200
    assert requests[0][0] == "/v1" + endpoint + "?" + query.decode()
    saved = json.loads((tmp_path / "000001.request.json").read_text())
    assert saved["upstream_url"] == url + endpoint
    assert saved["query_present"] is True
    assert saved["query_sha256"] == hashlib.sha256(query).hexdigest()
    for path in tmp_path.iterdir():
        assert b"QUERY-SECRET" not in path.read_bytes()


def test_full_json_response_retains_all_provider_items_only_when_requested():
    response = json.loads(response_body())
    response["output"] += [
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "answer"}],
        },
        {
            "id": "fn_1",
            "type": "function_call",
            "call_id": "call_1",
            "name": "exec_command",
            "arguments": '{"cmd":"pwd"}',
        },
        {
            "id": "custom_1",
            "type": "custom_tool_call",
            "call_id": "call_2",
            "name": "apply_patch",
            "input": "raw patch\n",
        },
        {
            "id": "search_1",
            "type": "web_search_call",
            "status": "completed",
            "action": {"type": "search", "query": "example"},
        },
    ]
    body = json.dumps(response, ensure_ascii=False).encode()
    assert "response" not in parse_response_body(body, "application/json")
    parsed = parse_response_body(body, "application/json", include_response=True)
    assert parsed["response"] == response
    assert parsed["response_completed"] is True


def test_full_sse_response_recovers_ciphertext_and_preserves_output_order():
    reason = json.loads(response_body())["output"][0]
    function = {
        "type": "function_call",
        "id": "fn_1",
        "call_id": "call_1",
        "name": "mcp__roborsi__observe",
        "arguments": "{}",
    }
    message = {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "next"}],
    }
    final = {
        "id": "resp_1",
        "status": "completed",
        "output": [{"type": "reasoning", "id": "rs_1"}, function, message],
        "usage": {"input_tokens": 8},
    }
    events = [
        {"type": "response.output_item.done", "output_index": 1, "item": function},
        {"type": "response.output_item.done", "output_index": 0, "item": reason},
        {"type": "response.output_item.done", "output_index": 0, "item": reason},
        {"type": "response.completed", "response": final},
    ]
    body = b"".join(
        b"data: " + json.dumps(event).encode() + b"\n\n" for event in events
    )
    parsed = parse_response_body(body, "text/event-stream", include_response=True)
    assert parsed["parse_status"] == "parsed"
    assert parsed["response"] == {**final, "output": [reason, function, message]}


def test_full_sse_response_exposes_partial_items_without_claiming_completion():
    item = {
        "type": "custom_tool_call",
        "id": "tool_1",
        "name": "apply_patch",
        "call_id": "call_1",
        "input": "opaque raw input",
    }
    body = (
        b"data: "
        + json.dumps(
            {"type": "response.output_item.done", "output_index": 0, "item": item}
        ).encode()
        + b"\n\ndata: {"
    )
    parsed = parse_response_body(body, "text/event-stream", include_response=True)
    assert parsed["response"]["output"] == [item]
    assert parsed["response_completed"] is False
    assert parsed["parse_status"] == "malformed"


def test_partial_stream_retains_sparse_completed_items_and_reports_gaps():
    item = {
        "id": "fn_4",
        "type": "function_call",
        "call_id": "c4",
        "name": "run",
        "arguments": "{}",
    }
    body = (
        b"data: "
        + json.dumps(
            {"type": "response.output_item.done", "output_index": 3, "item": item}
        ).encode()
        + b"\n\n"
    )
    parsed = parse_response_body(body, "text/event-stream", include_response=True)
    assert parsed["response"]["output"] == [None, None, None, item]
    assert parsed["response_completed"] is False
    assert parsed["response_conflicts"]


@pytest.mark.parametrize("conflict", ["cipher", "id", "index"])
def test_conflicting_sse_items_are_not_silently_merged(conflict):
    reason = json.loads(response_body())["output"][0]
    final_item = {**reason}
    done_index = 0
    if conflict == "cipher":
        final_item["encrypted_content"] = "different-cipher"
    elif conflict == "id":
        final_item["id"] = "different-item"
    else:
        done_index = 1
    events = [
        {
            "type": "response.output_item.done",
            "output_index": done_index,
            "item": reason,
        },
        {
            "type": "response.completed",
            "response": {"id": "resp_1", "status": "completed", "output": [final_item]},
        },
    ]
    body = b"".join(
        b"data: " + json.dumps(event).encode() + b"\n\n" for event in events
    )
    parsed = parse_response_body(body, "text/event-stream", include_response=True)
    assert parsed["parse_status"] == "malformed"
    assert parsed["response"]["output"][0] == final_item


def test_full_sse_response_rejects_mixed_response_identifiers():
    events = [
        {"type": "response.created", "response": {"id": "response_A"}},
        {
            "type": "response.output_item.done",
            "response_id": "response_A",
            "output_index": 0,
            "item": {
                "id": "rs_A",
                "type": "reasoning",
                "encrypted_content": "cipher_A",
            },
        },
        {
            "type": "response.completed",
            "response": {"id": "response_B", "status": "completed", "output": []},
        },
    ]
    body = b"".join(
        b"data: " + json.dumps(event).encode() + b"\n\n" for event in events
    )
    parsed = parse_response_body(body, "text/event-stream", include_response=True)
    assert parsed["parse_status"] == "malformed"
    assert "Conflicting response identifiers" in parsed["response_conflicts"]
    assert parsed["response"]["output"] == []


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize(
    "invalid",
    [
        b'"encrypted_content":"cipher_A","encrypted_content":"cipher_B"',
        b'"encrypted_content":"cipher_A","metadata":NaN',
        b'"encrypted_content":"cipher_A","metadata":Infinity',
        b'"encrypted_content":"cipher_A","metadata":1e999',
    ],
)
def test_full_response_rejects_ambiguous_or_nonfinite_json(streamed, invalid):
    response = (
        b'{"id":"resp_1","status":"completed","output":[{"id":"rs_1","type":"reasoning",'
        + invalid
        + b"}]}"
    )
    body = (
        b'data: {"type":"response.completed","response":' + response + b"}\n\n"
        if streamed
        else response
    )
    content_type = "text/event-stream" if streamed else "application/json"
    parsed = parse_response_body(body, content_type, include_response=True)
    assert parsed["parse_status"] == "malformed"
    assert parsed["response_conflicts"]
    # Existing capture indexing remains compatible; strictness is opt-in for
    # conversion, where accepting only a last duplicate value loses provenance.
    assert "response_conflicts" not in parse_response_body(body, content_type)


def test_full_compaction_response_retains_opaque_state_outside_reasoning_index():
    response = {
        "output": [
            {
                "type": "compaction",
                "id": "c_1",
                "encrypted_content": "opaque-compaction",
            }
        ]
    }
    parsed = parse_response_body(
        json.dumps(response).encode(),
        "application/json",
        endpoint="/responses/compact",
        include_response=True,
    )
    assert parsed["response"] == response
    assert parsed["reasoning_items"] == []
    assert parsed["response_completed"] is True


@pytest.mark.parametrize(
    "request_body",
    [
        {"model": "original", "input": [{"role": "user", "content": "task"}]},
        {"model": "original", "metadata": {"API-Key": "secret"}},
        {"model": "original", "metadata": {"API-Key": "[REDACTED]"}},
    ],
)
def test_request_fidelity_metadata_records_only_actual_redactions(
    tmp_path, request_body
):
    with upstream(response_body()) as (url, _):
        with ProviderCapture(ProviderCaptureConfig(url, "custom"), tmp_path) as capture:
            assert httpx.post(
                capture.base_url + "/responses", json=request_body
            ).is_success
    saved = json.loads((tmp_path / "000001.request.json").read_text())
    redacted = request_body.get("metadata", {}).get("API-Key") == "secret"
    assert saved["redacted_paths"] == (["/metadata/API-Key"] if redacted else [])
    assert saved["request_capture_complete"] is (not redacted)
    assert saved["request_redacted"] is redacted
    canonical = json.dumps(
        saved["forwarded"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    assert saved["forwarded_record_sha256"] == hashlib.sha256(canonical).hexdigest()


@pytest.mark.parametrize("body", [b"invalid json", b"[]", b'{"model":NaN}'])
def test_unusable_request_context_is_explicit_in_fidelity_metadata(tmp_path, body):
    with upstream(response_body()) as (url, requests):
        with ProviderCapture(ProviderCaptureConfig(url, "custom"), tmp_path) as capture:
            assert httpx.post(capture.base_url + "/responses", content=body).is_success
    saved = json.loads((tmp_path / "000001.request.json").read_text())
    assert saved["request_capture_complete"] is False
    assert requests[0][2] == body
