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

"""Exercise CLI selection and scheduling with offline model/native outcomes."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from robots.vln_shared import cli
from robots.vln_shared.benchmark import NavigationBenchmark
from robots.vln_shared.execution import BenchExecutor
from robots.vln_shared.storage import (
    file_digest,
    freeze_snapshot,
    inventory,
    read_json,
    write_json,
)


@pytest.fixture
def launch(tmp_path, monkeypatch):
    datasets = {}
    workspace = tmp_path / "history/workspace"
    workspace.mkdir(parents=True)
    (workspace / "EXPLORATION.md").write_text("Frozen training memory.")
    write_json(
        workspace.parent / "manifest.json",
        {"files": [{"path": k, "sha256": v} for k, v in inventory(workspace).items()]},
    )
    for name in ("lhpr_vln", "r2r", "rxr"):
        sections = {}
        annotation = tmp_path / f"{name}.json"
        annotation.write_text("offline input")
        for split in ("train", "test"):
            sections[split] = {
                "spec": {
                    "dataset": name,
                    "python": sys.executable,
                    "seed": 42,
                    "split": split,
                    "native_commit": "a" * 40,
                    "inputs": {str(annotation): file_digest(annotation)},
                },
                "episodes": [
                    {
                        "id": f"{name}-{split}-{index}",
                        "dataset": name,
                        "episode_id": str(index + 10),
                        "episode_index": index,
                        "scene": split,
                        "instruction": "Navigate to the door.",
                    }
                    for index in (197, 209, 333)
                ],
            }
        datasets[name] = {**sections, "import_snapshot": str(workspace.parent)}
    config = tmp_path / "campaign.json"
    write_json(
        config,
        {
            "schema_version": 1,
            "datasets": datasets,
            "agent": {"executable": "codex", "expected_version": "codex-cli 0.161.0"},
        },
    )
    state = SimpleNamespace(
        config=config, output=tmp_path / "run", calls=[], preflights=[], explorations=[]
    )

    class Executor(BenchExecutor):
        def __call__(self, case, budget, snapshot, directory):
            task = NavigationBenchmark(
                self.datasets[case["dataset"]]["test"]["spec"], [case], budget
            ).tasks()[0]
            assert task.seed == 42
            assert task.native["episode_index"] == case["episode_index"]
            assert task.max_steps == budget
            assert inventory(snapshot / "workspace") == inventory(workspace)
            state.calls.append((case["dataset"], case["episode_index"], budget))
            success = case["dataset"] == "r2r" or budget == 5000
            return {
                "status": "success" if success else "task_failure",
                "success": success,
                "score": float(success),
                "native": {"success": success, "steps": 12, "terminated": True},
            }

    def preflight(*args, **kwargs):
        state.preflights.append(True)
        return {"version": "offline-cli"}

    def explore(command, directory, cancel, timeout):
        dataset = command[command.index("--dataset") + 1]
        config = read_json(Path(command[command.index("--config") + 1]))
        state.explorations.append(config["datasets"][dataset]["train"])
        freeze_snapshot(workspace, Path(command[command.index("--snapshot") + 1]))
        return 0

    monkeypatch.setattr(cli, "BenchExecutor", Executor)
    monkeypatch.setattr(cli, "preflight_codex_cli", preflight)
    monkeypatch.setattr(cli, "run_process", explore)
    return state


def invoke(launch, *options):
    return cli.main(
        [
            "run",
            "--config",
            str(launch.config),
            "--output",
            str(launch.output),
            "--concurrency",
            "1",
            *options,
        ]
    )


def test_three_datasets_one_episode_only_500_and_resume(launch, capsys):
    before = launch.config.read_bytes()
    options = [
        "--datasets",
        "lhvln",
        "r2r",
        "rxr",
        "--episodes-per-dataset",
        "1",
        "--only-500",
    ]
    assert invoke(launch, *options) == 0
    assert launch.calls == [(name, 197, 500) for name in ("lhpr_vln", "r2r", "rxr")]
    report = read_json(launch.output / "evaluation/summary.json")
    assert report["complete"] and report["budgets"] == [500]
    for row in report["cases"]:
        assert row["selected"] == row["stage_500"]
        assert row["stage_5000"] is None
        assert row["policy_steps"] == 12
    assert report["datasets"]["lhpr_vln"]["success_rate"] == 0
    assert report["datasets"]["r2r"]["success_rate"] == 1
    assert read_json(launch.output / "evaluation/stage-5000.json") == {
        "enabled": False,
        "cases": [],
    }
    assert not list(launch.output.glob("evaluation/cases/*/5000"))
    captured = capsys.readouterr()
    assert json.loads(captured.out)["complete"]
    assert "event=run_start" in captured.err
    assert "event=import_exploration dataset=lhpr_vln" in captured.err
    assert "dataset=lhpr_vln episode=207" in captured.err
    assert "steps=12/500" in captured.err
    assert "status=task_failure" in captured.err
    assert (
        "event=dataset_end dataset=r2r scored=1/1 successful=1 incomplete=0"
        in captured.err
    )
    assert "event=run_end status=complete" in captured.err
    options[1] = "lhpr_vln"  # Alias and canonical name select identical inputs.
    assert invoke(launch, *options, "--resume", "--progress-interval", "1") == 0
    assert "event=reuse dataset=lhpr_vln" in capsys.readouterr().err
    assert len(launch.calls) == 3
    assert launch.config.read_bytes() == before


def test_default_runs_all_cases_and_reruns_only_scored_failures(launch):
    assert invoke(launch) == 0
    assert len(launch.calls) == 15
    assert all(budget == 500 for _, _, budget in launch.calls[:9])
    assert {name for name, _, budget in launch.calls if budget == 5000} == {
        "lhpr_vln",
        "rxr",
    }
    report = read_json(launch.output / "evaluation/summary.json")
    assert report["budgets"] == [500, 5000]
    assert all(group["requested"] == 3 for group in report["datasets"].values())
    assert all(group["success_rate"] == 1 for group in report["datasets"].values())


@pytest.mark.parametrize("only_500", [False, True])
def test_subset_skips_unselected_inputs_and_keeps_training(launch, only_500):
    original = read_json(launch.config)
    original["datasets"]["r2r"].pop("import_snapshot")
    write_json(launch.config, original)
    (launch.config.parent / "rxr.json").unlink()
    options = ["--datasets", "r2r", "--episodes-per-dataset", "2"]
    if only_500:
        options.append("--only-500")
    assert invoke(launch, *options) == 0
    assert launch.calls == [("r2r", 197, 500), ("r2r", 209, 500)]
    assert launch.explorations == [original["datasets"]["r2r"]["train"]]
    assert len(launch.explorations[0]["episodes"]) == 3
    assert sorted(p.name for p in (launch.output / "snapshots").iterdir()) == ["r2r"]


@pytest.mark.parametrize("only_500", [False, True])
def test_dry_run_reports_selection_without_model_or_exploration(
    launch, capsys, only_500
):
    options = ["--datasets", "rxr", "--episodes-per-dataset", "2", "--dry-run"]
    if only_500:
        options.append("--only-500")
    assert invoke(launch, *options) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["datasets"] == {"rxr": 2}
    assert report["budgets"] == ([500] if only_500 else [500, 5000])
    assert report["stage_5000"] == (
        "disabled" if only_500 else "only new scored failures"
    )
    assert not launch.calls and not launch.preflights and not launch.explorations


@pytest.mark.parametrize(
    "options",
    [
        ["--datasets", "unknown"],
        ["--datasets", "lhvln", "lhpr_vln"],
        ["--episodes-per-dataset", "0"],
        ["--episodes-per-dataset", "-1"],
        ["--episodes-per-dataset", "1.5"],
    ],
)
def test_invalid_cli_arguments_stop_before_creating_output(launch, options):
    with pytest.raises(SystemExit) as error:
        invoke(launch, *options)
    assert error.value.code == 2
    assert not launch.output.exists()
    assert not launch.preflights


@pytest.mark.parametrize("problem", ["missing", "too_many", "empty"])
def test_invalid_selection_stops_before_model_calls(launch, problem):
    config = read_json(launch.config)
    if problem == "missing":
        config["datasets"].pop("rxr")
    elif problem == "empty":
        config["datasets"]["rxr"]["test"]["episodes"] = []
    write_json(launch.config, config)
    with pytest.raises(ValueError, match="absent|episodes"):
        invoke(launch, "--datasets", "rxr", "--episodes-per-dataset", "4")
    assert not launch.preflights and not launch.calls


@pytest.mark.parametrize(
    "changed",
    [
        ["--datasets", "r2r", "--episodes-per-dataset", "1"],
        ["--datasets", "r2r", "--episodes-per-dataset", "2", "--only-500"],
        ["--datasets", "rxr", "--episodes-per-dataset", "1", "--only-500"],
    ],
)
def test_resume_rejects_changed_mode_or_selection(launch, changed):
    assert (
        invoke(launch, "--datasets", "r2r", "--episodes-per-dataset", "1", "--only-500")
        == 0
    )
    with pytest.raises(ValueError, match="unchanged"):
        invoke(launch, *changed, "--resume")
    assert launch.calls == [("r2r", 197, 500)]


def test_only_500_technical_error_remains_incomplete(launch, monkeypatch, capsys):
    def unavailable(self, case, budget, snapshot, directory):
        launch.calls.append(budget)
        raise TimeoutError("offline simulator error")

    monkeypatch.setattr(cli.BenchExecutor, "__call__", unavailable)
    assert (
        invoke(launch, "--datasets", "rxr", "--episodes-per-dataset", "1", "--only-500")
        == 2
    )
    assert launch.calls == [500, 500]  # One technical retry, still at 500.
    report = read_json(launch.output / "evaluation/summary.json")
    assert report["datasets"]["rxr"]["incomplete"] == 1
    assert report["datasets"]["rxr"]["success_rate"] is None
    assert report["cases"][0]["selected"] is None
    output = capsys.readouterr()
    assert json.loads(output.out)["complete"] is False
    assert "event=retry steps=?/500" in output.err
    assert "retry=1/1" in output.err
    assert "status=infrastructure_error" in output.err
    assert "event=run_end status=incomplete" in output.err


@pytest.mark.parametrize("interval", ["0", "-1", "nan", "inf"])
def test_invalid_progress_interval_stops_before_output(launch, interval):
    with pytest.raises(ValueError, match="progress-interval"):
        invoke(launch, "--progress-interval", interval)
    assert not launch.output.exists()
    assert not launch.preflights


def test_prepare_and_run_require_only_selected_dataset_files(launch, monkeypatch):
    preparation = importlib.import_module("robots.vln_shared.prepare")
    original = read_json(launch.config)
    source, history = launch.config.parent / "source", launch.config.parent / "old"
    write_json(
        history / "configs/r2r.json",
        {
            "benchmark_options": {
                "simulator_python": sys.executable,
                "seed": 42,
                "scenes_root": "unused",
                "data_root": "unused",
            },
            "train_episodes": [{"episode_id": "train"}],
        },
    )
    index = history / "episode_rollouts_500_then_5000/source_episode_index.csv"
    index.parent.mkdir()
    index.write_text("csv_file,episode_id\nr2rce.csv,test\n")
    catalog_calls = []

    def catalog(spec, ids):
        catalog_calls.append((spec["dataset"], spec["split"], ids))
        return original["datasets"]["r2r"][
            "train" if spec["split"] == "train" else "test"
        ]

    monkeypatch.setattr(preparation, "catalog", catalog)
    assert (
        cli.main(
            [
                "run",
                "--source-project",
                str(source),
                "--history",
                str(history),
                "--output",
                str(launch.output),
                "--datasets",
                "r2r",
                "--episodes-per-dataset",
                "1",
                "--only-500",
            ]
        )
        == 0
    )
    assert catalog_calls == [("r2r", "train", ["train"]), ("r2r", "rand100", ["test"])]
    assert launch.calls == [("r2r", 197, 500)]
    config = read_json(launch.output / "inputs/campaign.json")
    assert list(config["datasets"]) == ["r2r"]
    assert len(config["datasets"]["r2r"]["test"]["episodes"]) == 3
