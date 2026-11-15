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

"""Batch evaluation preserves case results and benchmark-specific denominators."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock

import pytest

from roborsi.benchmarks.registry import BenchmarkRegistry
from roborsi.evaluation.batch import (
    BatchEvaluator,
    CaseRecord,
    expand_cases,
    summarize_cases,
)
from roborsi.evaluation.messages import EvaluationResult, Provenance, ServiceError
from roborsi.evaluation.store import ResultStore
from tests.unit_tests.roborsi.benchmarks.test_registry_contracts import _manifest


def _registry(root: Path) -> BenchmarkRegistry:
    first, _ = _manifest(root)
    demo = json.loads(first.read_text(encoding="utf-8"))
    demo["tasks"][0]["seeds"] = [0, 1, 2, 3]
    first.write_text(json.dumps(demo), encoding="utf-8")

    other = json.loads(first.read_text(encoding="utf-8"))
    other["benchmark"] = "other"
    other["evaluation"]["aggregation"] = "task_weighted_success_rate"
    other["evaluation"]["error_denominator"] = "valid"
    other["tasks"] = [
        {**other["tasks"][0], "task_id": "pick", "seeds": [0, 1, 2]},
        {**other["tasks"][0], "task_id": "place", "seeds": [0]},
    ]
    target = root / "other"
    target.mkdir()
    (target / "assets").mkdir()
    (target / "assets" / "tasks.bin").write_bytes(b"task fixture")
    (target / "manifest.json").write_text(json.dumps(other), encoding="utf-8")
    return BenchmarkRegistry([root])


class _PublishedCases:
    def __init__(self, registry: BenchmarkRegistry, store: ResultStore) -> None:
        self.registry = registry
        self.store = store

    def run(
        self,
        run_id,
        case_id,
        episode_id,
        benchmark,
        task_id,
        seed,
        agent_version,
        **kwargs,
    ):
        lock = self.registry.lock(benchmark)
        task = lock.task_message(task_id, seed)
        provenance = Provenance.from_dict(lock.provenance(agent_version).as_dict())
        self.store.claim(run_id, case_id, task, provenance)
        if (benchmark == "demo" and seed in {2, 3}) or (
            benchmark == "other" and task_id == "pick" and seed == 2
        ):
            infrastructure = seed == 3 or benchmark == "other"
            result = EvaluationResult(
                task=task,
                episode_id=episode_id,
                status="infrastructure_error" if infrastructure else "agent_error",
                success=None,
                score=None,
                provenance=provenance,
                error=ServiceError(
                    code="infrastructure_error" if infrastructure else "agent_error",
                    message="service failed",
                ),
            )
        else:
            success = seed == 0
            result = EvaluationResult(
                task=task,
                episode_id=episode_id,
                status="success" if success else "task_failure",
                success=success,
                score=float(success),
                provenance=provenance,
            )
        self.store.publish(run_id, case_id, result)
        return result


def test_batch_keeps_per_case_results_and_separate_benchmark_aggregates(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path / "registry")
    store = ResultStore(tmp_path / "results")
    cases = expand_cases(registry, ["demo", "other"])

    report = BatchEvaluator(registry, store, _PublishedCases(registry, store)).run(
        "run1", cases, "agent-v1", max_concurrent=2
    )

    assert len(report.cases) == 8
    assert len({record.case.case_id for record in report.cases}) == 8
    assert [record.case for record in report.cases] == list(cases)
    assert all(
        store.load("run1", record.case.case_id) == record.result
        for record in report.cases
    )
    demo = report.summaries["demo"]
    assert (
        demo.planned,
        demo.scored,
        demo.task_failures,
        demo.agent_errors,
        demo.infrastructure_errors,
    ) == (
        4,
        2,
        1,
        1,
        1,
    )
    assert demo.success_rate == pytest.approx(1 / 4)
    assert demo.aggregate_score == pytest.approx(1 / 4)
    other = report.summaries["other"]
    assert (
        other.planned,
        other.scored,
        other.successes,
        other.infrastructure_errors,
    ) == (4, 3, 2, 1)
    assert other.success_rate == pytest.approx(2 / 3)
    assert other.aggregate_score == pytest.approx(0.75)
    assert "aggregate_score" not in report.as_dict()
    with pytest.raises(FileExistsError, match="run already exists"):
        BatchEvaluator(registry, store, _PublishedCases(registry, store)).run(
            "run1", cases, "agent-v1"
        )


def test_matrix_rejects_undeclared_task_seed_before_launch(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry")
    with pytest.raises(KeyError, match="undeclared"):
        expand_cases(registry, ["demo"], seeds=[99])
    with pytest.raises(ValueError, match="unknown task"):
        expand_cases(registry, ["demo"], task_ids={"demo": ["missing"]})
    with pytest.raises(ValueError, match="unselected benchmark"):
        expand_cases(registry, ["demo"], task_ids={"other": ["pick"]})


def test_task_weighted_mean_aggregates_native_scores_per_task(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path / "registry")
    evaluation = {
        "aggregation": "task_weighted_mean",
        "error_denominator": "valid",
    }
    records = []
    lock = registry.lock("other")
    for index, (task_id, seed, score) in enumerate(
        (("pick", 0, 1.0), ("pick", 1, 0.5), ("place", 0, 0.25))
    ):
        case = expand_cases(
            registry, ["other"], task_ids={"other": [task_id]}, seeds=[seed]
        )[0]
        task = lock.task_message(task_id, seed)
        result = EvaluationResult(
            task=task,
            episode_id=f"episode-{index}",
            status="success" if score == 1.0 else "task_failure",
            success=score == 1.0,
            score=score,
            provenance=Provenance(),
        )
        records.append(CaseRecord(case, result, "run"))

    summary = summarize_cases(records, evaluation)
    assert summary.mean_native_score == pytest.approx((1.0 + 0.5 + 0.25) / 3)
    assert summary.aggregate_score == pytest.approx(((1.0 + 0.5) / 2 + 0.25) / 2)


def test_batch_starts_independent_cases_up_to_concurrency_limit(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry")
    store = ResultStore(tmp_path / "results")
    barrier = Barrier(2, timeout=3.0)
    lock = Lock()
    active = 0
    peak = 0

    class ConcurrentCases(_PublishedCases):
        def run(self, *args, **kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                barrier.wait()
                return super().run(*args, **kwargs)
            finally:
                with lock:
                    active -= 1

    cells = expand_cases(registry, ["demo"], seeds=[0, 1])
    BatchEvaluator(registry, store, ConcurrentCases(registry, store)).run(
        "run1", cells, "agent-v1", max_concurrent=2
    )
    assert peak == 2


def test_resume_reuses_completed_cases_and_retries_only_retryable_or_claimed(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path / "registry")
    store = ResultStore(tmp_path / "results")
    cases = expand_cases(registry, ["demo"], seeds=[0, 1, 2, 3])
    calls: list[tuple[str, int]] = []

    class InterruptedCases(_PublishedCases):
        def run(
            self,
            run_id,
            case_id,
            episode_id,
            benchmark,
            task_id,
            seed,
            agent_version,
            **kwargs,
        ):
            calls.append((run_id, seed))
            if run_id == "run1" and seed in {1, 3}:
                lock = registry.lock(benchmark)
                task = lock.task_message(task_id, seed)
                provenance = Provenance.from_dict(
                    lock.provenance(agent_version).as_dict()
                )
                store.claim(run_id, case_id, task, provenance)
                if seed == 3:
                    raise RuntimeError("worker interrupted before publication")
                result = EvaluationResult(
                    task=task,
                    episode_id=episode_id,
                    status="infrastructure_error",
                    success=None,
                    score=None,
                    provenance=provenance,
                    error=ServiceError(
                        code="infrastructure_error",
                        message="temporary failure",
                        retryable=True,
                    ),
                )
                store.publish(run_id, case_id, result)
                return result
            return super().run(
                run_id,
                case_id,
                episode_id,
                benchmark,
                task_id,
                seed,
                agent_version,
                **kwargs,
            )

    evaluator = BatchEvaluator(registry, store, InterruptedCases(registry, store))
    with pytest.raises(RuntimeError, match="worker interrupted"):
        evaluator.run("run1", cases, "agent-v1", max_concurrent=3)
    assert store.list_cases("run1") == {
        cases[0].case_id: "published",
        cases[1].case_id: "published",
        cases[2].case_id: "published",
        cases[3].case_id: "claimed",
    }
    first_result = store.load("run1", cases[0].case_id)
    report = evaluator.run("run2", cases, "agent-v1", resume_from="run1")
    assert calls.count(("run2", 0)) == 0
    assert calls.count(("run2", 2)) == 0
    assert {(run, seed) for run, seed in calls if run == "run2"} == {
        ("run2", 1),
        ("run2", 3),
    }
    assert [record.source_run_id for record in report.cases] == [
        "run1",
        "run2",
        "run1",
        "run2",
    ]
    assert store.load("run1", cases[0].case_id) == first_result
    assert store.list_cases("run2") == {
        cases[1].case_id: "published",
        cases[3].case_id: "published",
    }

    chained = evaluator.run("run3", cases, "agent-v1", resume_from="run2")
    assert [record.source_run_id for record in chained.cases] == [
        "run1",
        "run2",
        "run1",
        "run2",
    ]
    assert store.list_cases("run3") == {}
    assert chained.summaries["demo"] == report.summaries["demo"]


def test_resume_rejects_changed_inputs_before_starting_cases(tmp_path: Path) -> None:
    root = tmp_path / "registry"
    registry = _registry(root)
    store = ResultStore(tmp_path / "results")
    cells = expand_cases(registry, ["demo"], seeds=[0, 1])
    evaluator = BatchEvaluator(registry, store, _PublishedCases(registry, store))
    evaluator.run("run1", cells, "agent-v1")
    with pytest.raises(ValueError, match="batch inputs differ"):
        evaluator.run("run2", cells, "agent-v2", resume_from="run1")
    with pytest.raises(ValueError, match="batch inputs differ"):
        evaluator.run("run2", cells[:1], "agent-v1", resume_from="run1")
    with pytest.raises(ValueError, match="batch inputs differ"):
        evaluator.run("run2", cells, "agent-v1", timeout_s=10, resume_from="run1")

    manifest = root / "demo" / "manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["evaluation"]["error_denominator"] = "valid"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    changed_registry = BenchmarkRegistry([root])
    with pytest.raises(ValueError, match="batch inputs differ"):
        BatchEvaluator(
            changed_registry, store, _PublishedCases(changed_registry, store)
        ).run("run2", cells, "agent-v1", resume_from="run1")
    assert not (store.root / "run2").exists()


def test_batch_run_id_reservation_is_exclusive(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry")
    store = ResultStore(tmp_path / "results")
    cases = expand_cases(registry, ["demo"], seeds=[0])
    evaluator = BatchEvaluator(registry, store, _PublishedCases(registry, store))
    barrier = Barrier(2)

    def launch():
        barrier.wait()
        try:
            return evaluator.run("run1", cases, "agent-v1")
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        reports = list(executor.map(lambda _: launch(), range(2)))
    assert sum(report is not None for report in reports) == 1
    assert store.list_cases("run1") == {cases[0].case_id: "published"}


def test_cli_selects_matrix_and_emits_queryable_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    from roborsi.evaluation import cli

    root = tmp_path / "registry"
    _registry(root)
    monkeypatch.setattr(
        cli,
        "CaseOrchestrator",
        lambda registry, store, launcher, docker: _PublishedCases(registry, store),
    )
    result_root = tmp_path / "results"
    assert (
        cli.main(
            [
                "eval",
                "--registry-root",
                str(root),
                "--results-root",
                str(result_root),
                "--run-id",
                "run1",
                "--agent-factory",
                "example.agent:make_agent",
                "--agent-version",
                "v1",
                "--benchmark",
                "demo,other",
                "--task",
                "other:pick",
                "--seed",
                "0",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert len(report["cases"]) == 2
    assert set(report["benchmarks"]) == {"demo", "other"}
    store = ResultStore(result_root)
    assert all(
        store.load("run1", record["case_id"]).as_dict() == record["result"]
        for record in report["cases"]
    )
    assert (
        cli.main(
            [
                "eval",
                "--registry-root",
                str(root),
                "--results-root",
                str(result_root),
                "--run-id",
                "run2",
                "--resume-from",
                "run1",
                "--agent-factory",
                "example.agent:make_agent",
                "--agent-version",
                "v1",
                "--benchmark",
                "demo,other",
                "--task",
                "other:pick",
                "--seed",
                "0",
            ]
        )
        == 0
    )
    resumed = json.loads(capsys.readouterr().out)
    assert resumed["resume_from"] == "run1"
    assert {case["source_run_id"] for case in resumed["cases"]} == {"run1"}
    assert store.list_cases("run2") == {}


def test_cli_accepts_an_external_http_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    from roborsi.evaluation import cli
    from roborsi.evaluation.http_agent import HttpAgentLauncher

    root = tmp_path / "registry"
    _registry(root)
    launchers = []

    def fake_orchestrator(registry, store, launcher, docker):
        launchers.append(launcher)
        return _PublishedCases(registry, store)

    monkeypatch.setattr(cli, "CaseOrchestrator", fake_orchestrator)
    assert (
        cli.main(
            [
                "eval",
                "--registry-root",
                str(root),
                "--results-root",
                str(tmp_path / "results"),
                "--run-id",
                "run1",
                "--agent-url",
                "https://agent.example.org",
                "--agent-version",
                "v1",
                "--benchmark",
                "demo",
                "--seed",
                "0",
            ]
        )
        == 0
    )
    assert isinstance(launchers[0], HttpAgentLauncher)
    assert launchers[0].endpoint == "https://agent.example.org"
    assert len(json.loads(capsys.readouterr().out)["cases"]) == 1
