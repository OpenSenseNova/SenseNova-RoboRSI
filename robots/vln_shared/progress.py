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

"""Thread-safe CLI progress using native physical steps and explicit attempt states."""

from __future__ import annotations

import logging
import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from roborsi.utils.logging import get_logger


@dataclass
class _Attempt:
    label: str
    budget: int
    started: float
    updated: float
    steps: int | None = None


class Progress:
    """Own a stderr logger and heartbeat thread for one CLI invocation.

    Observations update physical action counts; heartbeats repeat the last known
    count without treating model silence as a task failure or a timeout.
    """

    def __init__(self, interval: float = 30) -> None:
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("--progress-interval must be positive and finite")
        self.interval = interval
        self.logger = get_logger("vln.progress")
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.active: dict[tuple[str, int], _Attempt] = {}
        self.phase_name = "starting"
        self.thread = threading.Thread(target=self._heartbeat, name="vln-progress")

    def __enter__(self) -> Progress:
        self.previous = self.logger.level, self.logger.propagate
        self.handler = logging.StreamHandler(sys.stderr)
        self.handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S")
        )
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stopped.set()
        self.thread.join()
        self.logger.removeHandler(self.handler)
        self.handler.close()
        self.logger.setLevel(self.previous[0])
        self.logger.propagate = self.previous[1]

    def phase(self, name: str) -> None:
        """Display preparation, exploration, stage transitions or final counts."""
        with self.lock:
            self.phase_name = name
            self.logger.info("%s", name)

    def start(
        self, case: dict, budget: int, directory: Path, retry: int, retries: int
    ) -> None:
        """Announce a fresh attempt, resetting its count before native reset."""
        now = time.monotonic()
        state = _Attempt(
            label=(
                f"dataset={case['dataset']} episode={case['episode_id']} "
                f"attempt={directory.name} retry={retry}/{retries}"
            ),
            budget=budget,
            started=now,
            updated=now,
        )
        with self.lock:
            self.active[(case["id"], budget)] = state
            self._emit(state, "retry" if retry else "start", f"artifacts={directory}")

    def advance(self, case: dict, budget: int, steps: int) -> None:
        """Publish each observed batch's physical count, including native recovery."""
        with self.lock:
            state = self.active.get((case["id"], budget))
            if state is not None and steps != state.steps:
                state.steps = steps
                state.updated = time.monotonic()
                self._emit(state, "step")

    def finish(self, case: dict, budget: int, result: dict) -> None:
        """Display the committed status and stop heartbeats for this attempt."""
        with self.lock:
            state = self.active.pop((case["id"], budget))
            if (
                isinstance(result.get("native"), dict)
                and type(result["native"].get("steps")) is int
            ):
                state.steps = result["native"]["steps"]
            self._emit(state, "attempt_end", f"status={result['status']}")

    def reuse(self, case: dict, budget: int, result: dict) -> None:
        """Distinguish reuse of an already scored result from a new execution."""
        self.phase(
            f"event=reuse dataset={case['dataset']} episode={case['episode_id']} "
            f"steps={result['native']['steps']}/{budget} status={result['status']} "
            f"artifacts={result['attempt']}"
        )

    def _emit(self, state: _Attempt, event: str, detail: str = "") -> None:
        now = time.monotonic()
        steps = "?" if state.steps is None else str(state.steps)
        self.logger.info(
            "%s event=%s steps=%s/%d elapsed=%.1fs idle=%.1fs %s",
            state.label,
            event,
            steps,
            state.budget,
            now - state.started,
            now - state.updated,
            detail,
        )

    def _heartbeat(self) -> None:
        while not self.stopped.wait(self.interval):
            with self.lock:
                if not self.active:
                    self.logger.info("event=waiting phase=(%s)", self.phase_name)
                for state in self.active.values():
                    self._emit(state, "heartbeat")
