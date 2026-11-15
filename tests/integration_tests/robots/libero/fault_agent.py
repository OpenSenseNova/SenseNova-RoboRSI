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

"""Controlled Agent faults for the opt-in real LIBERO lifecycle checks."""

import os
import time
from pathlib import Path

from examples.roborsi.hold_agent import HoldAgent
from roborsi.evaluation.messages import Action, Observation
from roborsi.utils.config import require_project_env


class FaultAgent(HoldAgent):
    def __init__(self, case_dir: Path) -> None:
        super().__init__()
        self.directory = case_dir

    def act(self, observation: Observation) -> Action:
        action = super().act(observation)
        (self.directory / "act.started").write_text("ready")
        mode = require_project_env("ROBORSI_TEST_FAULT")
        if mode == "agent_error":
            raise RuntimeError("injected Agent failure after reading LIBERO images")
        if mode == "agent_crash":
            os._exit(9)
        if mode in {"agent_cancel", "agent_timeout"}:
            time.sleep(120)
        if mode == "invalid_action":
            return action.model_copy(update={"data": {"vector": [0.0] * 6}})
        return action


def make_agent(case_dir: Path) -> FaultAgent:
    return FaultAgent(case_dir)
