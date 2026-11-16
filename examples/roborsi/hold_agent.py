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

"""Read LIBERO observations and keep the gripper open without a model."""

from __future__ import annotations

from pathlib import Path

from roborsi.evaluation.media import CaseMediaChannel
from roborsi.evaluation.messages import Action, Observation, Task
from roborsi.utils.config import require_project_env


class HoldAgent:
    """Read every camera file and hold each Benchmark's native control target."""

    def __init__(self) -> None:
        self._media = CaseMediaChannel(require_project_env("ROBORSI_MEDIA_DIR"))

    def reset(self, task: Task) -> None:
        if task.benchmark not in {
            "libero",
            "libero_pro",
            "libero_long_plus_plus",
            "libero_ultra_long",
        }:
            raise ValueError("this example supports the LIBERO benchmark family")

    def act(self, observation: Observation) -> Action:
        for reference in observation.media:
            self._media.read(reference, observation.episode_id, observation.step_index)
        return Action(
            benchmark=observation.benchmark,
            episode_id=observation.episode_id,
            step_index=observation.step_index,
            data={"vector": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]},
        )

    def close(self) -> None:
        pass


def make_agent(case_dir: Path) -> HoldAgent:
    """Create one session for PythonAgentLauncher."""
    return HoldAgent()
