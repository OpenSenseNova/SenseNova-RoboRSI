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

"""Agent matrices preserve independent results and resume all stages safely."""

import json

import pytest

from roborsi.evaluation import cli
from roborsi.evaluation.batch import BatchEvaluator, expand_cases
from roborsi.evaluation.matrix import MatrixAgent, matrix_status, run_matrix
from roborsi.evaluation.store import ResultStore
from tests.unit_tests.roborsi.agents.test_config import source
from tests.unit_tests.roborsi.evaluation.test_batch import (
    _PublishedCases,
    _registry,
)


def test_cli_expands_presets_cross_benchmarks_and_global_overrides(
    tmp_path, monkeypatch, capsys
):
    source(tmp_path / "src")
    matrix = tmp_path / "matrix.json"
    matrix.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "agents": {
                    "baseline": {"config": "src/preset.json"},
                    "candidate": {
                        "config": "src/preset.json",
                        "overrides": ["parameters.distance=0.75"],
                    },
                },
            }
        )
    )
    _registry(tmp_path / "registry")
    launchers = []

    def runner(registry, store, launcher, executor):
        launchers.append(launcher)
        return _PublishedCases(registry, store)

    monkeypatch.setattr(cli, "CaseOrchestrator", runner)
    arguments = [
        "eval",
        "--registry-root",
        str(tmp_path / "registry"),
        "--results-root",
        str(tmp_path / "results"),
        "--run-id",
        "matrix",
        "--benchmark",
        "demo,other",
        "--seed",
        "0",
        "--agent-matrix",
        str(matrix),
        "--agent-set",
        "agent.max_turns=6",
    ]
    assert cli.main([*arguments, "--dry-run"]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["total_cases"] == 6
    assert not (tmp_path / "results").exists()
    assert cli.main(arguments) == 0
    result = json.loads(capsys.readouterr().out)
    assert set(result["agents"]) == {"baseline", "candidate"}
    assert result["status"] == "completed"
    for row in result["agents"].values():
        assert len(row["cases"]) == 3
        assert set(row["benchmarks"]) == {"demo", "other"}
    assert (
        result["agents"]["baseline"]["agent_version"]
        != result["agents"]["candidate"]["agent_version"]
    )
    assert all(
        launcher._preset.document["agent"]["max_turns"] == 6 for launcher in launchers
    )
    assert (
        cli.main(
            [
                "status",
                "--results-root",
                str(tmp_path / "results"),
                "--run-id",
                "matrix",
            ]
        )
        == 0
    )
    status = json.loads(capsys.readouterr().out)
    assert status["status"] == "completed"
    assert len(status["agents"]["candidate"]["cases"]) == 3


def agents(registry, store, *, fail_second=False, versions=("one", "two")):
    cases = expand_cases(registry, ["demo"], seeds=[0])

    class BeforeStart(BatchEvaluator):
        def run(self, *args, **kwargs):
            if fail_second:
                raise RuntimeError("interrupted before second Agent")
            return super().run(*args, **kwargs)

    return [
        MatrixAgent(
            name,
            version,
            evaluator(registry, store, _PublishedCases(registry, store)),
            cases,
        )
        for name, version, evaluator in [
            ("first", versions[0], BatchEvaluator),
            ("second", versions[1], BeforeStart),
        ]
    ]


def test_matrix_resume_keeps_completed_cases_and_handles_unstarted_agents(tmp_path):
    registry = _registry(tmp_path / "registry")
    store = ResultStore(tmp_path / "results")
    with pytest.raises(RuntimeError, match="second Agent"):
        run_matrix(store, "original", agents(registry, store, fail_second=True))
    assert matrix_status(store, "original")["status"] == "interrupted"
    assert not (store.root / "original--second").exists()
    # An interrupted resume must still trace back to the first completed Agent.
    with pytest.raises(RuntimeError):
        run_matrix(
            store,
            "partial",
            agents(registry, store, fail_second=True),
            resume_from="original",
        )
    report = run_matrix(
        store, "resumed", agents(registry, store), resume_from="partial"
    )
    assert report["agents"]["first"]["cases"][0]["source_run_id"] == "original--first"
    assert report["agents"]["second"]["cases"][0]["source_run_id"] == "resumed--second"
    with pytest.raises(ValueError, match="inputs differ"):
        run_matrix(
            store,
            "changed",
            agents(registry, store, versions=("one", "changed")),
            resume_from="resumed",
        )
    assert not (store.root / "changed").exists()


def test_matrix_refuses_resume_while_parent_owns_execution(tmp_path):
    registry = _registry(tmp_path / "registry")
    store = ResultStore(tmp_path / "results")

    class TryResume(_PublishedCases):
        def run(self, *args, **kwargs):
            assert matrix_status(store, "active")["status"] == "running"
            with pytest.raises(ValueError, match="active worker"):
                run_matrix(
                    store, "unsafe", agents(registry, store), resume_from="active"
                )
            return super().run(*args, **kwargs)

    entries = agents(registry, store)
    entries[0].evaluator.orchestrator = TryResume(registry, store)
    run_matrix(store, "active", entries)
    assert not (store.root / "unsafe").exists()


def test_later_agent_preflight_failure_starts_nothing(tmp_path):
    registry = _registry(tmp_path / "registry")
    store = ResultStore(tmp_path / "results")
    entries = agents(registry, store)
    entries[1] = MatrixAgent("second", "two", entries[1].evaluator, ())
    with pytest.raises(ValueError, match="non-empty"):
        run_matrix(store, "invalid", entries)
    assert not store.root.exists()
