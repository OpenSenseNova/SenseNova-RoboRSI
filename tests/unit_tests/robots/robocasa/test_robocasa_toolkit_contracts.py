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

"""Benchmark toolkit state, dispatch and native success contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from roborsi.dashboard.events import NullDashboardEventSink
from roborsi.tools.toolkit import Toolkit, _is_readonly
from roborsi.utils import templates
from robots.robocasa import toolkit

COMMON_TOOLS = {"finish"}
EXPECTED_TOOLS = COMMON_TOOLS | {
    "move_to",
    "move_delta",
    "rotate_pitch",
    "set_gripper",
    "release",
    "scripted_grasp",
    "rldx_skill",
    "rldx_arm",
    "navigate_to",
    "move_base",
    "reset",
    "view_env_state",
    "back_project_batch",
    "query_world_map",
}


def _record(step_idx: int = 0) -> SimpleNamespace:
    return SimpleNamespace(step_idx=step_idx, terminated=False, extras={})


def _tool_names(robot_toolkit: Toolkit) -> set[str]:
    return {spec["name"] for spec in robot_toolkit.get_tools_spec()}


def _readonly_names(robot_toolkit: Toolkit) -> set[str]:
    return {
        name
        for name, (_, handler) in robot_toolkit._tools.items()
        if _is_readonly(handler)
    }


def test_toolkit_constructs_and_classifies_tools_with_a_fake(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_single_arm_primitives: type[Any],
) -> None:
    monkeypatch.delenv("RLDX_MAX_CHUNKS", raising=False)
    monkeypatch.delenv("RLDX_SETTLE_PATIENCE", raising=False)
    import robots.robocasa.primitives as primitives_module

    dumped: list[Any] = []
    monkeypatch.setattr(
        templates, "default_variables", lambda: {"output_dir": "/offline/output"}
    )
    monkeypatch.setattr(
        primitives_module,
        "RoboCasaPrimitives",
        fake_single_arm_primitives,
    )
    monkeypatch.setattr(toolkit, "get_output_dir", lambda: tmp_path)
    monkeypatch.setattr(
        toolkit.robocasa_tools,
        "dump_state",
        lambda primitives, state, log: dumped.append(primitives) or _record(),
    )

    robot_toolkit = toolkit.RoboCasaToolkit(
        runtime_kwargs={"env_client": object(), "vla_client": object()},
        dashboard_events=NullDashboardEventSink(),
    )

    assert _tool_names(robot_toolkit) == EXPECTED_TOOLS
    assert _readonly_names(robot_toolkit) == COMMON_TOOLS | {
        "view_env_state",
        "back_project_batch",
        "query_world_map",
    }
    assert dumped == fake_single_arm_primitives.instances
    primitive = fake_single_arm_primitives.instances[0]
    assert primitive.reset_calls == 1
    assert primitive.recording_started is True
    assert callable(primitive.kwargs["check_cancelled"])
    assert (tmp_path / "success_criteria.md").read_text() == "offline success criteria"


def test_solved_reads_only_the_final_environment_record() -> None:
    records = [SimpleNamespace(extras={"success": True})]
    robot_toolkit = toolkit.RoboCasaToolkit.__new__(toolkit.RoboCasaToolkit)
    robot_toolkit._state = SimpleNamespace(latest_record=lambda: records[-1])

    assert robot_toolkit.solved() is True

    records.append(SimpleNamespace(extras={"success": False}))
    assert robot_toolkit.solved() is False


def test_evaluation_reset_does_not_restart_the_environment():
    from robots.robocasa.primitives import RoboCasaPrimitives

    native = RoboCasaPrimitives.__new__(RoboCasaPrimitives)
    native._allow_reset = False
    native.env = SimpleNamespace(reset=lambda: pytest.fail("evaluation reset ran"))
    assert "reset is DISABLED" in native.reset()["error"]


def test_benchmark_toolkit_rejects_exploration(tmp_path):
    with pytest.raises(ValueError, match="unsupported"):
        toolkit.RoboCasaToolkit(
            runtime_kwargs={},
            dashboard_events=NullDashboardEventSink(),
            mode="exploration",
            state_output_dir=tmp_path,
        )
