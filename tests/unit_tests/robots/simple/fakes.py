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

"""Scripted physics boundary; the real Adapter, process, RPC and HTTP stay active."""

from pathlib import Path


class FakeRuntime:
    """Return only deterministic feedback while exposing forbidden fields to tests."""

    def __init__(
        self,
        directory: Path | None = None,
        *,
        success: bool = True,
        end: int | None = 3,
    ) -> None:
        self.calls = self.closed = 0
        self.success, self.end = success, end
        self.submissions = []

    def values(self) -> dict:
        import numpy as np

        return {
            "state": [0.0] * 31 + [0.75],
            "rgb": np.full((360, 640, 3), self.calls % 256, dtype=np.uint8),
            "object_pose": [123, 456, 789],
            "reward": 0.99,
        }

    def start(self, task):
        return self.values()

    def submit(self, rows):
        self.submissions.append(rows)

    def step(self):
        self.calls += 1
        done = self.end is not None and self.calls >= self.end
        return self.values(), done and self.success, done

    def close(self):
        self.closed += 1


def make_runtime(directory: Path) -> FakeRuntime:
    return FakeRuntime(directory)


def make_nonterminal_runtime(directory: Path) -> FakeRuntime:
    return FakeRuntime(directory, success=False, end=None)
