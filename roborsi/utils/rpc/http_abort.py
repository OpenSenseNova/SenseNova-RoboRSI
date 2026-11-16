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

"""Interrupt connected HTTP requests without abandoning a request thread."""

from __future__ import annotations

import http.client
import socket
import threading
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_abort_check: ContextVar[Callable[[], None] | None] = ContextVar(
    "http_rpc_abort_check", default=None
)


@contextmanager
def interrupt_http_calls(check: Callable[[], None] | None) -> Iterator[None]:
    """Check cancellation during HTTP RPCs in the current execution context.

    ``check`` must return promptly or raise the caller's cancellation/timeout
    exception. Connected reads and writes are interrupted within one polling
    interval (50 ms, plus scheduling). DNS uses the system resolver; connect
    and TLS setup still use the transport timeout. Remote execution is not rolled back: the caller
    must close or terminate the service it owns after interruption.
    """
    token = _abort_check.set(check)
    try:
        yield
    finally:
        _abort_check.reset(token)


class HttpCallGuard:
    """Watch one request, shutting down its sockets when its check raises."""

    def __init__(self) -> None:
        self.check = _abort_check.get()
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._sockets: list[socket.socket] = []
        self._error: Exception | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> HttpCallGuard:
        if self.check is not None:
            self.check()
            self._thread = threading.Thread(
                target=self._watch, name="rpc-http-abort", daemon=True
            )
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._done.set()
        if self._thread is not None:
            self._thread.join()
        for connection in self._sockets:
            connection.close()
        if self._error is not None:
            raise self._error
        if self.check is not None:
            self.check()

    def _watch(self) -> None:
        assert self.check is not None
        while not self._done.wait(0.05):
            try:
                self.check()
            except Exception as exc:
                with self._lock:
                    self._error = exc
                    for connection in self._sockets:
                        try:
                            connection.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                return

    def _track(self, connection: socket.socket) -> None:
        with self._lock:
            if self._error is not None:
                connection.close()
                raise self._error
            self._sockets.append(connection)

    def opener(self, *, direct: bool) -> urllib.request.OpenerDirector:
        """Keep urllib proxy/TLS behavior while tracking each connected socket."""
        guard = self

        class HTTPConnection(http.client.HTTPConnection):
            def connect(self) -> None:
                super().connect()
                guard._track(self.sock)

        class HTTPSConnection(http.client.HTTPSConnection):
            def connect(self) -> None:
                super().connect()
                guard._track(self.sock)

        class HTTPHandler(urllib.request.HTTPHandler):
            def http_open(self, request: Any) -> Any:
                return self.do_open(HTTPConnection, request)

        class HTTPSHandler(urllib.request.HTTPSHandler):
            def https_open(self, request: Any) -> Any:
                return self.do_open(HTTPSConnection, request, context=self._context)

        handlers: list[Any] = [HTTPHandler(), HTTPSHandler()]
        if direct:
            handlers.append(urllib.request.ProxyHandler({}))
        return urllib.request.build_opener(*handlers)
