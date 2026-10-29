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

"""Live pipe output and cleanup checks for navigation progress."""

import json
import logging
import selectors
import subprocess
import sys
import threading

import pytest

from robots.vln_shared.execution import run_process
from robots.vln_shared.progress import Progress
from robots.vln_shared.storage import read_json


@pytest.mark.timeout(15)
def test_stderr_is_live_before_process_exit_and_stdout_stays_json():
    script = """
import json, sys
from pathlib import Path
from robots.vln_shared.progress import Progress
case = {"id": "rxr-198", "dataset": "rxr", "episode_id": "198"}
with Progress(0.05) as progress:
    progress.phase("event=explore dataset=rxr")
    sys.stdin.readline()
    progress.start(case, 500, Path("attempt-0001"), 0, 1)
    progress.advance(case, 500, 42)
    sys.stdin.readline()
    progress.finish(case, 500, {"status": "success", "native": {"steps": 42}})
print(json.dumps({"complete": True}))
"""
    with (
        subprocess.Popen(
            [sys.executable, "-c", script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as process,
        selectors.DefaultSelector() as selector,
    ):
        selector.register(process.stderr, selectors.EVENT_READ)
        pending = b""

        def read_until(marker):
            nonlocal pending
            while marker not in pending:
                assert selector.select(timeout=5), "progress was buffered until exit"
                chunk = process.stderr.read1(4096)
                assert chunk, "child exited before producing progress"
                pending += chunk

        try:
            read_until(b"event=waiting phase=(event=explore dataset=rxr)")
            assert process.poll() is None
            process.stdin.write(b"start\n")
            process.stdin.flush()
            read_until(b"event=heartbeat steps=42/500")
            assert process.poll() is None
            process.stdin.write(b"finish\n")
            process.stdin.flush()
            stdout, stderr = process.communicate(timeout=5)
            assert process.returncode == 0
            assert json.loads(stdout) == {"complete": True}
            assert b"status=success" in pending + stderr
            assert b"dataset=rxr episode=198" in pending
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


@pytest.mark.parametrize("failure", ["cancel", "timeout"])
def test_progress_and_child_cleanup_on_interruption(tmp_path, failure):
    logger = logging.getLogger("roborsi.vln.progress")
    previous = logger.level, logger.propagate, logger.handlers[:]
    cancel = threading.Event()
    timer = threading.Timer(0.1, cancel.set)
    if failure == "cancel":
        timer.start()
    try:
        with pytest.raises(InterruptedError if failure == "cancel" else TimeoutError):
            with Progress(0.01) as progress:
                progress.start(
                    {"id": "a", "dataset": "r2r", "episode_id": "7"},
                    500,
                    tmp_path,
                    0,
                    1,
                )
                run_process(
                    [sys.executable, "-c", "import time; time.sleep(60)"],
                    tmp_path,
                    cancel,
                    10 if failure == "cancel" else 0.1,
                )
        assert not progress.thread.is_alive()
        assert (logger.level, logger.propagate, logger.handlers) == previous
        from robots.vln_shared.execution import _birth

        assert _birth(read_json(tmp_path / "process.json")["pid"]) is None
    finally:
        timer.cancel()
