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

"""Observe running, interrupted and failed cases independently of output files."""

from __future__ import annotations

import errno
import json
import select
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path

import pytest

from roborsi.evaluation import cli
from roborsi.evaluation import store as store_module
from roborsi.evaluation.messages import EvaluationResult, Provenance
from roborsi.evaluation.runner import CaseRunner
from roborsi.evaluation.services import AdapterClient, AgentClient
from roborsi.evaluation.store import ResultStore
from tests.unit_tests.roborsi.evaluation.test_runner import _task
from tests.unit_tests.roborsi.evaluation.test_services import _service


def test_live_worker_crash_and_read_only_queries(tmp_path, capsys):
    root, journal = tmp_path / "results", tmp_path / "state"
    store = ResultStore(root, status_root=journal)
    store.claim("run", "case", _task(), Provenance())
    assert store.status("run", "case").state == "pending"
    script = """
import sys, time
from roborsi.evaluation.store import ResultStore
store = ResultStore(sys.argv[1], status_root=sys.argv[2])
with store.execution('run', 'case', 'episode'):
    print('ready', flush=True)
    time.sleep(30)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(root), str(journal)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert select.select([process.stdout], [], [], 10)[0]
        assert process.stdout.readline().strip() == "ready"
        reader = ResultStore(root, status_root=journal)
        assert reader.status("run", "case").state == "running"
        with pytest.raises(ValueError, match="active worker"):
            with reader.execution("run", "case", "other"):
                pytest.fail("concurrent worker entered")
        assert (
            cli.main(
                [
                    "status",
                    "--results-root",
                    str(root),
                    "--status-root",
                    str(journal),
                    "--run-id",
                    "run",
                    "--case-id",
                    "case",
                ]
            )
            == 0
        )
        assert json.loads(capsys.readouterr().out)["state"] == "running"
        process.kill()
        process.wait(timeout=5)
        record = journal / "run/case.json"
        before = record.read_bytes()
        state = reader.status("run", "case")
        assert state.state == "interrupted" and state.error.retryable
        assert state.error.details["stage"] == "worker.exit"
        assert record.read_bytes() == before
        with pytest.raises(ValueError, match="already started"):
            with reader.execution("run", "case", "again"):
                pytest.fail("interrupted case was reused")
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        process.stdout.close()


def test_result_write_failure_remains_queryable_and_cannot_be_replaced(
    tmp_path, monkeypatch, capsys
):
    root, journal = tmp_path / "results", tmp_path / "state"
    store = ResultStore(root, status_root=journal)
    original = store_module.write_json_atomic

    def disk_full(path, record):
        if Path(path).name.startswith(".result-"):
            raise OSError(errno.ENOSPC, "result volume full")
        return original(path, record)

    monkeypatch.setattr(store_module, "write_json_atomic", disk_full)
    with ExitStack() as services:
        agent = AgentClient(services.enter_context(_service("agent")))
        adapter = AdapterClient(services.enter_context(_service("adapter")))
        result = CaseRunner(store).run(
            "run", "case", "episode", _task(), Provenance(), agent, adapter, timeout_s=3
        )
    assert result.status == "infrastructure_error" and result.score is None
    assert result.error.details["stage"] == "store.publish"
    assert not (root / "run/case/result.json").exists()
    reader = ResultStore(root, status_root=journal)
    assert reader.load("run", "case") == result
    state = reader.status("run", "case")
    assert state.state == "failed" and not state.result_published
    assert state.result_status == "infrastructure_error"
    assert reader.list_cases("run") == {"case": "failed"}
    assert (
        cli.main(
            [
                "status",
                "--results-root",
                str(root),
                "--status-root",
                str(journal),
                "--run-id",
                "run",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["case"]["error"]["details"]["stage"] == "store.publish"
    monkeypatch.setattr(store_module, "write_json_atomic", original)
    with pytest.raises(FileExistsError, match="final storage failure"):
        reader.publish("run", "case", result)
    assert reader.load("run", "case") == result


def test_plan_write_failure_and_unhandled_exception_leave_error_states(
    tmp_path, monkeypatch
):
    store = ResultStore(tmp_path / "results", status_root=tmp_path / "state")
    original = store_module.write_json_atomic

    def fail_plan(path, record):
        raise PermissionError("output directory is read-only")

    monkeypatch.setattr(store_module, "write_json_atomic", fail_plan)
    with pytest.raises(PermissionError):
        store.claim("run", "plan", _task(), Provenance(), episode_id="plan-episode")
    reader = ResultStore(store.root, status_root=tmp_path / "state")
    assert reader.status("run", "plan").error.details["stage"] == "store.claim"
    assert reader.load("run", "plan").score is None
    monkeypatch.setattr(store_module, "write_json_atomic", original)
    store.claim("run", "crash", _task(), Provenance())
    with pytest.raises(RuntimeError, match="unexpected"):
        with store.execution("run", "crash", "crash-episode"):
            raise RuntimeError("unexpected worker failure")
    assert reader.status("run", "crash").state == "failed"
    assert reader.load("run", "crash").error.message == "unexpected worker failure"


def test_status_checks_published_evidence_and_old_cases(tmp_path):
    store = ResultStore(tmp_path)
    store.claim("run", "case", _task(), Provenance())
    result = EvaluationResult(
        task=_task(),
        episode_id="episode",
        provenance=Provenance(),
        status="task_failure",
        success=False,
        score=0,
    )
    store.publish("run", "case", result)
    state = store.status("run", "case")
    assert state.state == "completed" and state.result_status == "task_failure"
    assert state.result_published
    (tmp_path / "run/case/result.json").write_text('{"score":NaN}')
    assert store.status("run", "case").state == "failed"
    assert store.status("run", "case").error.details["stage"] == "store.load"
    old = tmp_path / "run/legacy"
    old.mkdir()
    assert store.status("run", "legacy").state == "unknown"
    with pytest.raises(FileNotFoundError):
        store.status("run", "missing")
    with pytest.raises(ValueError):
        store.status("../escape", "case")


def test_committed_result_survives_status_update_failure(tmp_path, monkeypatch):
    store = ResultStore(tmp_path)
    store.claim("run", "case", _task(), Provenance())
    result = EvaluationResult(
        task=_task(),
        episode_id="episode",
        provenance=Provenance(),
        status="task_failure",
        success=False,
        score=0,
    )

    def fail_status(*args):
        raise OSError(errno.ENOSPC, "status volume full")

    with store.execution("run", "case", "episode"):
        monkeypatch.setattr(store.journal, "completed", fail_status)
        store.publish("run", "case", result)
    assert store.load("run", "case") == result
    assert store.status("run", "case").state == "completed"
    assert store.status("run", "case").result_published


def test_batch_rejects_resume_while_a_previous_worker_is_alive(tmp_path):
    from roborsi.evaluation.batch import BatchEvaluator, expand_cases
    from tests.unit_tests.roborsi.evaluation.test_batch import (
        _PublishedCases,
        _registry,
    )

    registry = _registry(tmp_path / "registry")
    store = ResultStore(tmp_path / "results")
    cells = expand_cases(registry, ["demo"], seeds=[0])
    batch = BatchEvaluator(registry, store, _PublishedCases(registry, store))
    original = store.publish
    contexts = []

    def running_case(run_id, case_id, result):
        context = store.execution(run_id, case_id, result.episode_id)
        context.__enter__()
        contexts.append(context)
        raise RuntimeError("worker is still running")

    store.publish = running_case
    try:
        with pytest.raises(RuntimeError):
            batch.run("first", cells, "agent-v1")
        with pytest.raises(ValueError, match="active worker"):
            batch.run("second", cells, "agent-v1", resume_from="first")
        assert not (store.root / "second").exists()
    finally:
        store.publish = original
        for context in contexts:
            context.__exit__(None, None, None)
    report = batch.run("recovered", cells, "agent-v1", resume_from="first")
    assert report.cases[0].result.status == "success"
    assert report.cases[0].source_run_id == "recovered"


def test_batch_counts_storage_failure_and_retries_in_a_new_run(tmp_path, monkeypatch):
    from roborsi.evaluation.batch import BatchEvaluator, expand_cases
    from roborsi.evaluation.case_status import CasePublicationError
    from tests.unit_tests.roborsi.evaluation.test_batch import (
        _PublishedCases,
        _registry,
    )

    registry = _registry(tmp_path / "registry")
    store = ResultStore(tmp_path / "results", status_root=tmp_path / "state")
    cells = expand_cases(registry, ["demo"], seeds=[0])

    class Publisher(_PublishedCases):
        def run(self, *args, **kwargs):
            try:
                return super().run(*args, **kwargs)
            except CasePublicationError as exc:
                return exc.result

    batch = BatchEvaluator(registry, store, Publisher(registry, store))
    original = store_module.write_json_atomic

    def no_space(path, value):
        if Path(path).name.startswith(".result-"):
            raise OSError(errno.ENOSPC, "result volume full")
        return original(path, value)

    monkeypatch.setattr(store_module, "write_json_atomic", no_space)
    first = batch.run("failed", cells, "agent-v1")
    assert first.summaries["demo"].planned == 1
    assert first.summaries["demo"].scored == 0
    assert first.summaries["demo"].infrastructure_errors == 1
    assert first.cases[0].result.error.retryable
    monkeypatch.setattr(store_module, "write_json_atomic", original)
    second = batch.run("retry", cells, "agent-v1", resume_from="failed")
    assert second.cases[0].source_run_id == "retry"
    assert second.summaries["demo"].scored == 1
    assert store.load("failed", cells[0].case_id) == first.cases[0].result


def test_failure_of_both_storage_locations_does_not_look_running(tmp_path, monkeypatch):
    store = ResultStore(tmp_path / "results", status_root=tmp_path / "state")
    store.claim("run", "case", _task(), Provenance())
    result = EvaluationResult(
        task=_task(),
        episode_id="episode",
        provenance=Provenance(),
        status="task_failure",
        success=False,
        score=0,
    )

    def no_space(*args, **kwargs):
        raise OSError(errno.ENOSPC, "storage unavailable")

    with pytest.raises(OSError, match="storage unavailable"):
        with store.execution("run", "case", "episode"):
            monkeypatch.setattr(store_module, "write_json_atomic", no_space)
            monkeypatch.setattr(store.journal, "_write", no_space)
            store.publish("run", "case", result)
    assert store.status("run", "case").state == "interrupted"
    assert not store.status("run", "case").result_published
    assert not (store.root / "run/case/result.json").exists()


@pytest.mark.parametrize("change", ["episode", "trace", "published", "empty"])
def test_damaged_failure_record_cannot_bypass_result_validation(tmp_path, change):
    store = ResultStore(tmp_path / "results", status_root=tmp_path / "state")
    store.claim("run", "case", _task(), Provenance())
    with pytest.raises(OSError, match="write failed"):
        with store.execution("run", "case", "episode"):
            raise OSError("write failed")
    path = tmp_path / "state/run/case.json"
    record = json.loads(path.read_text())
    if change == "episode":
        record["fallback_result"]["episode_id"] = "another-episode"
    elif change == "trace":
        record["fallback_result"]["trace_manifest"] = "unverified/manifest.json"
    elif change == "published":
        record["status"]["result_published"] = True
    else:
        record = {}
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="invalid case"):
        store.load("run", "case")
