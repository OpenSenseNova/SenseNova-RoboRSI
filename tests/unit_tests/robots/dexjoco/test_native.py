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

"""Native process cleanup remains retryable after an interrupted join."""

import pytest

from robots.dexjoco.native import ProcessNative


def test_interrupted_close_retries_reaping_worker(monkeypatch, tmp_path):
    class Connection:
        closed = False

        def close(self):
            self.closed = True

        def poll(self, timeout):
            return True

        def recv(self):
            return True, None

        def send(self, message):
            if self.closed:
                raise OSError("connection closed")

    class WorkerProcess:
        alive = True
        interrupted = False

        def start(self):
            pass

        def is_alive(self):
            return self.alive

        def join(self, timeout):
            if not self.interrupted:
                self.interrupted = True
                raise OSError("join interrupted")
            self.alive = False

    class Context:
        process = WorkerProcess()

        def Pipe(self):
            return Connection(), Connection()

        def Process(self, **kwargs):
            return self.process

    context = Context()
    monkeypatch.setattr(
        "robots.dexjoco.native.multiprocessing.get_context", lambda mode: context
    )
    native = ProcessNative(tmp_path, "water_plant", 100000, 10)
    with pytest.raises(OSError, match="join interrupted"):
        native.close()
    assert context.process.is_alive()
    native.close()
    native.close()
    assert not context.process.is_alive()
    assert native.connection.closed
