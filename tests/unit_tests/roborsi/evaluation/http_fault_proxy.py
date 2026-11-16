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

"""Lose HTTP replies after execution in both offline and real Benchmark tests."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections import Counter
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@contextmanager
def lost_responses(
    endpoint,
    *,
    mode="truncate",
    paths=("/v1/episodes", "/actions", "/finish"),
    failures_per_request=1,
    token=None,
):
    attempts, requests, lock = Counter(), [], threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def forward(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", "0"))) or None
            request = urllib.request.Request(
                endpoint + self.path,
                data=body,
                method=self.command,
                headers={
                    "Content-Type": "application/json",
                    **({"Authorization": f"Bearer {token}"} if token else {}),
                },
            )
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            try:
                response = opener.open(request, timeout=120)
            except urllib.error.HTTPError as exc:
                response = exc
            with response:
                content, status, headers = (
                    response.read(),
                    response.status,
                    dict(response.headers),
                )
            key = (self.command, self.path, body)
            with lock:
                attempts[key] += 1
                drop = (
                    self.path.endswith(paths)
                    and attempts[key] <= failures_per_request
                    and status < 300
                )
                requests.append(
                    {
                        "method": self.command,
                        "path": self.path,
                        "body": json.loads(body) if body else None,
                        "dropped": drop,
                    }
                )
            self.close_connection = True
            if drop and mode == "disconnect":
                return
            if drop and mode == "bad_status":
                self.connection.sendall(b"broken status line\r\n\r\n")
                return
            self.send_response(status)
            for name, value in headers.items():
                if name.lower() not in {
                    "content-length",
                    "connection",
                    "transfer-encoding",
                    "server",
                    "date",
                }:
                    self.send_header(name, value)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content[: len(content) // 2] if drop else content)
            self.wfile.flush()

        do_GET = do_POST = do_DELETE = forward

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
