# Copyright 2026 The SenseNova-RoboRSI Authors.
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

"""Target50 scoring over bench eval result directories."""

import json

from benchmarks.robocasa365.score import collect, score


def write(run, index, task, seed, status, message=None):
    case = run / f"case-{index}"
    case.mkdir(parents=True)
    (case / "result.json").write_text(
        json.dumps(
            {
                "task": {"benchmark": "robocasa365", "task_id": task, "seed": seed},
                "status": status,
                "error": {"message": message} if message else None,
            }
        )
    )


def test_seed1_mean_of_task_rates_counts_missing_cells_as_failures(tmp_path):
    first, second = tmp_path / "run_0", tmp_path / "run_1"
    write(first, 0, "OpenDrawer", 1, "success")
    write(first, 1, "ArrangeTea", 1, "task_failure")
    write(first, 2, "CloseFridge", 1, "error")
    write(second, 0, "CloseFridge", 1, "success")  # valid re-run replaces it
    report = score(collect([first, second]), "seed1")
    assert report["cells_expected"] == 50 and report["cells_valid"] == 3
    assert report["splits"]["atomic"]["solved"] == 2
    assert report["overall"] == 4.0
    assert "composite_unseen/ArrangeTea_s1" not in report["invalid"]
    assert "atomic/OpenStandMixerHead_s1" in report["invalid"]


def test_full_protocol_weights_each_task_equally(tmp_path):
    run = tmp_path / "run"
    for seed in range(1, 11):
        write(run, seed, "OpenDrawer", seed, "success" if seed <= 5 else "task_failure")
    report = score(collect([run]))
    assert report["cells_expected"] == 340
    assert report["splits"]["atomic"]["success_rate"] == round(100 * 0.5 / 18, 2)
    assert report["overall"] == 1.0


def test_planner_timeout_is_a_valid_failure_and_other_errors_are_listed(tmp_path):
    run = tmp_path / "run"
    write(
        run,
        0,
        "ArrangeTea",
        1,
        "agent_error",
        "act: Embodied Agent failed: API planner timed out after 3600s",
    )
    write(run, 1, "OpenDrawer", 1, "agent_error", "connection reset")
    report = score(collect([run]), "seed1")
    assert report["planner_timeouts"] == ["ArrangeTea_s1"]
    assert "composite_unseen/ArrangeTea_s1" not in report["invalid"]
    assert "atomic/OpenDrawer_s1" in report["invalid"]
