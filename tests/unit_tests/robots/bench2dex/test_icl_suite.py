# Copyright 2026 The RoboRSI Contributors.
# Licensed under the Apache License, Version 2.0.

"""Original frozen package and complete-cohort evidence contracts."""

import hashlib
import json
import subprocess
import sys

import pytest

from benchmarks.bench2dex import run_suite
from roborsi.evaluation.messages import EvaluationResult, Provenance, ServiceError
from roborsi.evaluation.store import ResultStore
from robots.bench2dex import contracts, icl


def test_package_mutation_is_rejected(package, task):
    icl.verify_package(package, task)
    (package / "payload/TASK.md").write_text("changed")
    with pytest.raises(ValueError, match="changed after freezing"):
        icl.verify_package(package, task)


def _publish_suite_case(work, task_id, sample, outcome, case_id=None):
    if outcome == "missing":
        return
    store = ResultStore(work / "results")
    task = contracts.task_for(task_id, sample)
    run_id = f"task_{int(task_id):02d}"
    case_id = case_id or f"episode-{sample}"
    provenance = Provenance()
    directory = store.claim(run_id, case_id, task, provenance)
    if outcome == "unpublished":
        return
    operational = outcome in {"infrastructure_error", "agent_error", "cancelled"}
    status = outcome if operational or outcome == "task_failure" else "success"
    trace = directory / "trace/manifest.json"
    trace.parent.mkdir()
    trace.write_text(
        json.dumps({"status": "complete", "complete": outcome != "incomplete_trace"})
    )
    native = directory / "native/result.json"
    native.parent.mkdir()
    native.write_text(json.dumps({"success": status == "success"}))
    has_native = not operational and outcome != "missing_native"
    result = EvaluationResult(
        task=task,
        episode_id=case_id,
        status=status,
        success=None if operational else status == "success",
        score=None if operational else float(status == "success"),
        native_score_source="native/result.json" if has_native else None,
        native_score_sha256=hashlib.sha256(native.read_bytes()).hexdigest()
        if has_native
        else None,
        trace_manifest="trace/manifest.json",
        provenance=provenance,
        error=ServiceError(code=status, message="offline failure")
        if operational
        else None,
    )
    store.publish(run_id, case_id, result)
    if outcome == "tampered_native":
        native.write_text("changed evidence")
    elif outcome == "missing_native_file":
        native.unlink()
    elif outcome == "malformed_result":
        (directory / "result.json").write_text("{")
    elif outcome == "duplicate":
        _publish_suite_case(work, task_id, sample, "success", case_id="duplicate")


@pytest.fixture
def suite_invocation(tmp_path, monkeypatch):
    plan = {
        "schema_version": 1,
        "condition": "standard_1x",
        "tasks": ["3", "6"],
        "samples": [5, 6],
        "expected_cases": 4,
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    work = tmp_path / "evaluation"
    calls = []
    monkeypatch.setattr(
        run_suite, "prepare", lambda task, output, **kwargs: output / "agent.yaml"
    )

    def configure(outcome="success", returncode=0, dry_run=False):
        arguments = [
            "run_suite",
            "--work",
            str(work),
            "--plan",
            str(plan_path),
            "--benchmark-url",
            "http://127.0.0.1:8000",
            "--icl-source",
            str(tmp_path / "icl"),
        ]
        if dry_run:
            arguments.append("--dry-run")
        monkeypatch.setattr(sys, "argv", arguments)

        def run(command, **kwargs):
            task_id = command[command.index("--task") + 1].split(":")[1]
            calls.append(task_id)
            if "--dry-run" not in command:
                for sample in plan["samples"]:
                    selected = "success"
                    if task_id == "3":
                        selected = "task_failure" if sample == 5 else outcome
                    _publish_suite_case(work, task_id, sample, selected)
            return subprocess.CompletedProcess(command, returncode)

        monkeypatch.setattr(run_suite.subprocess, "run", run)
        return work, calls

    return configure


@pytest.mark.parametrize("outcome", ["success", "task_failure"])
def test_suite_continues_after_valid_outcomes(suite_invocation, outcome):
    work, calls = suite_invocation(outcome)
    run_suite.main()
    report = json.loads((work / "evaluation-summary.json").read_text())
    assert calls == ["3", "6"]
    assert report["complete"] and report["valid"] == report["expected"] == 4
    assert report["success_rate"] == (0.75 if outcome == "success" else 0.5)
    assert not (work / "execution-failure.json").exists()


@pytest.mark.parametrize(
    "outcome",
    [
        "infrastructure_error",
        "agent_error",
        "cancelled",
        "missing",
        "unpublished",
        "incomplete_trace",
        "missing_native",
        "tampered_native",
        "missing_native_file",
        "malformed_result",
        "duplicate",
    ],
)
def test_suite_stops_on_invalid_evidence_despite_zero_exit(
    suite_invocation, outcome, capsys, monkeypatch
):
    work, calls = suite_invocation(outcome)
    with pytest.raises(SystemExit) as error:
        run_suite.main()
    assert error.value.code == 1
    assert calls == ["3"]
    assert "execution-failure.json" in capsys.readouterr().err
    failure = json.loads((work / "execution-failure.json").read_text())
    assert failure["task"] == "3"
    assert failure["exit_code"] == 1 and failure["subprocess_exit_code"] == 0
    assert failure["missing"] or failure["issues"]
    report = json.loads((work / "evaluation-summary.json").read_text())
    assert not report["complete"] and report["success_rate"] is None
    assert report["expected"] == 4
    assert {"task": "6", "sample": 5} in report["missing"]
    preserved = work / "results/task_03/episode-5/result.json"
    original = preserved.read_bytes()
    assert json.loads(original)["status"] == "task_failure"
    monkeypatch.setattr(sys, "argv", ["run_suite", "--work", str(work), "--summarize"])
    run_suite.main()
    assert calls == ["3"] and preserved.read_bytes() == original
    assert json.loads((work / "evaluation-summary.json").read_text()) == report


@pytest.mark.parametrize("dry_run", [False, True])
def test_suite_preserves_nonzero_command_exit(suite_invocation, dry_run):
    work, calls = suite_invocation("missing", returncode=7, dry_run=dry_run)
    with pytest.raises(SystemExit) as error:
        run_suite.main()
    assert error.value.code == 7 and calls == ["3"]
    failure = json.loads((work / "execution-failure.json").read_text())
    assert failure["exit_code"] == failure["subprocess_exit_code"] == 7
    assert (work / "evaluation-summary.json").exists() is not dry_run


def test_suite_dry_run_does_not_require_results(suite_invocation):
    work, calls = suite_invocation(dry_run=True)
    run_suite.main()
    assert calls == ["3", "6"]
    assert not (work / "execution-failure.json").exists()
    assert not (work / "evaluation-summary.json").exists()
