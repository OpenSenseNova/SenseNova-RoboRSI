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

"""The Agent launcher owns one independent process and closes it on errors."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from roborsi.evaluation.agent_process import PythonAgentLauncher
from roborsi.evaluation.messages import Observation, Task
from roborsi.utils.daemon import ProcessDaemon

_FACTORY = "tests.unit_tests.roborsi.evaluation.service_child:make_demo_agent"


def test_python_agent_launcher_runs_service_in_child_process(tmp_path: Path) -> None:
    launcher = PythonAgentLauncher(_FACTORY, startup_timeout_s=10.0)
    task = Task(
        benchmark="demo", task_id="place", seed=7, instruction="Place", max_steps=2
    )
    with launcher.start(tmp_path) as process:
        assert process.daemon.poll() is None
        process.client.reset(task, "case-1", timeout_s=3.0)
        action = process.client.act(
            Observation(benchmark="demo", episode_id="case-1", step_index=0),
            timeout_s=3.0,
        )
        assert action.data == {"native": {"__ndarray__": "literal-field"}}
        process.client.close()
    deadline = time.monotonic() + 3.0
    while process.daemon.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert process.daemon.poll() is not None
    assert (tmp_path / "agent-service.log").is_file()


def test_python_agent_launcher_cleans_up_failed_factory(tmp_path: Path) -> None:
    launcher = PythonAgentLauncher(
        "missing_roborsi_agent:make_agent", startup_timeout_s=3.0
    )
    with pytest.raises(RuntimeError, match="exited with code"):
        launcher.start(tmp_path)
    assert "No module named" in (tmp_path / "agent-service.log").read_text()


def test_process_daemon_can_leave_rlinf_out_of_agent_pythonpath(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROBORSI_RLINF_ROOT", "/unlikely-rlinf-checkout")
    monkeypatch.setenv("PYTHONPATH", "/agent-dependencies")
    daemon = ProcessDaemon(
        "roborsi-agent",
        [sys.executable, "-c", "pass"],
        prepend_rlinf_path=False,
    )
    assert daemon.subprocess_env["PYTHONPATH"] == os.environ["PYTHONPATH"]
