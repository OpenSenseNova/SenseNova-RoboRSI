# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bounded JSON-line bridge to older, separately installed Habitat interpreters."""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import threading
import time
from pathlib import Path


class NativeProcess:
    """Own one simulator process; timeout and close always release its resources."""

    def __init__(self, python: str, directory: Path, *, timeout: float = 600) -> None:
        self.timeout = timeout
        self.lock = threading.Lock()
        self.buffer = b""
        self.log = (directory / "simulator.log").open("wb")
        try:
            self.process = subprocess.Popen(
                [python, "-u", str(Path(__file__).with_name("native_worker.py"))],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                bufsize=0,
            )
        except BaseException:
            self.log.close()
            raise

    def call(self, command: str, **arguments: object) -> dict:
        """Wait for one bounded reply; never reuse a failed stream."""
        with self.lock:
            process = self.process
            if process.poll() is not None:
                raise RuntimeError(f"native process exited: {process.returncode}")
            try:
                process.stdin.write(
                    (json.dumps({"command": command, **arguments}) + "\n").encode()
                )
                deadline = time.monotonic() + self.timeout
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    while b"\n" not in self.buffer:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0 or not selector.select(remaining):
                            raise TimeoutError(
                                f"native {command} exceeded {self.timeout}s"
                            )
                        data = os.read(process.stdout.fileno(), 65536)
                        if not data:
                            raise RuntimeError(
                                "native process closed its output; see simulator.log"
                            )
                        self.buffer += data
                        if len(self.buffer) > 32 * 1024 * 1024:
                            raise ValueError("native reply exceeds limit")
                line, self.buffer = self.buffer.split(b"\n", 1)
                reply = json.loads(line)
                if "error" in reply:
                    raise RuntimeError(reply["error"])
                return reply["result"]
            except BaseException:
                self.close()
                raise

    def close(self) -> None:
        """Close pipes, then terminate and reap a stuck simulator within five seconds."""
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        for stream in (self.process.stdin, self.process.stdout, self.log):
            stream.close()
