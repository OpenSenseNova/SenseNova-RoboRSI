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

"""Cancellation must release the connection before a delayed server replies."""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from roborsi.evaluation.deadline import CaseCancelled, CaseDeadline, CaseTimeout
from roborsi.utils.rpc.http_abort import interrupt_http_calls
from roborsi.utils.rpc.http_rpc import HttpRpcClient


@pytest.mark.parametrize("response_started", [False, True])
@pytest.mark.parametrize("cancelled", [False, True])
def test_abort_waiting_for_headers_or_body_without_request_thread(
    response_started: bool, cancelled: bool
) -> None:
    received, release, disconnected, cancellation = (
        threading.Event() for _ in range(4)
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            if response_started:
                self.send_response(200)
                self.send_header("Content-Length", "100")
                self.end_headers()
                self.wfile.write(b'{"ok":')
                self.wfile.flush()
            received.set()
            self.connection.settimeout(3)
            # A client that only abandons a worker thread would leave this open.
            if self.rfile.read(1) == b"":
                disconnected.set()
            release.wait(3)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    deadline = CaseDeadline(5 if cancelled else 0.3, cancel_event=cancellation)

    def cancel():
        if received.wait(3):
            cancellation.set()

    trigger = threading.Thread(target=cancel, daemon=True) if cancelled else None
    if trigger is not None:
        trigger.start()
    client = HttpRpcClient(f"http://127.0.0.1:{server.server_port}")
    started = time.monotonic()
    try:
        with interrupt_http_calls(deadline.check):
            with pytest.raises(CaseCancelled if cancelled else CaseTimeout):
                client.call("step", timeout_s=20)
        assert time.monotonic() - started < 2
        assert received.is_set() and disconnected.wait(1)
        assert not any(t.name == "rpc-http-abort" for t in threading.enumerate())
    finally:
        release.set()
        client.close()
        server.shutdown()
        server.server_close()
        serving.join(3)
        if trigger is not None:
            trigger.join(3)


def test_cancellation_scope_restores_normal_calls_and_does_not_send_after_abort():
    from tests.unit_tests.roborsi.utils.rpc.test_transport_loopback import (
        _running_facade,
    )

    cancellation = threading.Event()
    deadline = CaseDeadline(30, cancel_event=cancellation)
    with _running_facade("http") as running:
        with interrupt_http_calls(deadline.check):
            assert running.client.call("healthz") == {"status": "ok"}
            cancellation.set()
            with pytest.raises(CaseCancelled):
                running.client.call("delay", args=(10,))
        assert not running.facade.delay_finished.is_set()
        assert running.client.call("healthz") == {"status": "ok"}


def test_cancellable_http_keeps_environment_proxy(monkeypatch):
    from tests.unit_tests.roborsi.utils.rpc.test_transport_loopback import (
        _running_http_proxy,
    )

    with _running_http_proxy() as (port, requests):
        monkeypatch.setenv("http_proxy", f"http://127.0.0.1:{port}")
        monkeypatch.setenv("no_proxy", "")
        with interrupt_http_calls(lambda: None):
            result = HttpRpcClient("http://service.example.invalid").call("healthz")
        assert result == {"status": "proxied"}
        assert requests == ["http://service.example.invalid/call"]


def test_cancelling_one_thread_does_not_cancel_another_client():
    from tests.unit_tests.roborsi.utils.rpc.test_transport_loopback import (
        _running_facade,
    )

    ready, cancellation = threading.Event(), threading.Event()
    deadline = CaseDeadline(10, cancel_event=cancellation)
    outcomes = []
    with _running_facade("http") as running:

        def cancelled_call():
            try:
                with interrupt_http_calls(deadline.check):
                    ready.set()
                    running.client.call("delay", args=(0.3,))
            except CaseCancelled:
                outcomes.append("cancelled")

        worker = threading.Thread(target=cancelled_call)
        worker.start()
        try:
            assert ready.wait(1)
            cancellation.set()
            assert running.client.call("healthz") == {"status": "ok"}
        finally:
            worker.join(2)
        assert not worker.is_alive() and outcomes == ["cancelled"]
