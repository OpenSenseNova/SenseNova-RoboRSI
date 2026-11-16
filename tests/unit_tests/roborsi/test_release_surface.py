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

"""Public commands expose only the supported initial release surface."""

import subprocess
import sys
from pathlib import Path

import pytest

from roborsi.benchmarks.runtime_source import benchmark_source_files
from roborsi.dashboard.events import NullDashboardEventSink
from roborsi.evaluation.cli import main
from roborsi.planner.base import build_planner


def test_evaluation_help_exposes_evaluation_without_rsi_or_cluster(capsys):
    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0
    help_text = capsys.readouterr().out
    assert "{compare,status,eval}" in help_text
    assert "    rsi " not in help_text.lower()
    with pytest.raises(SystemExit) as result:
        main(["eval", "--help"])
    assert result.value.code == 0
    assert "--sco-config" not in capsys.readouterr().out


@pytest.mark.parametrize("name", ["robotwin", "robodojo", "simple"])
def test_unreleased_benchmark_build_is_rejected(name):
    with pytest.raises(ValueError, match="unsupported Benchmark"):
        benchmark_source_files(name)


def test_removed_flash_backend_fails_before_import(tmp_path):
    with pytest.raises(ValueError, match="[Uu]nknown|[Uu]nsupported"):
        build_planner(
            "flash",
            output_dir=tmp_path,
            recipe_tag="test",
            robot_name="test",
            dashboard_events=NullDashboardEventSink(),
        )


def test_lightweight_import_and_cli_do_not_load_native_simulators():
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import roborsi; "
            "from roborsi.evaluation import cli; "
            "from robots.libero import benchmark_service, three_benchmark_service; "
            "from robots.bench2dex import benchmark_service as bench2dex_service; "
            "from robots.behavior_1k import benchmark_service as behavior_service; "
            "from robots.dexjoco import benchmark_service as dexjoco_service; "
            "from robots.robocasa import benchmark_service as robocasa_service; "
            "assert not ({'torch', 'mujoco', 'robosuite', 'libero', 'rlinf', "
            "'lilo_vla', 'isaacsim', 'isaaclab', 'gymnasium', "
            "'omnigibson', 'bddl', 'omni', 'rldx', 'sam3'} & set(sys.modules))",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
