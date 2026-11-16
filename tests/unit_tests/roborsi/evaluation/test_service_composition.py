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

"""Replace result persistence while using real, independent HTTP services."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from roborsi.evaluation.agent_artifact import AgentArtifactStore
from roborsi.evaluation.batch import BatchEvaluator, expand_cases
from roborsi.evaluation.case_status import CaseStatus, timestamp
from roborsi.evaluation.http_benchmark import (
    HttpBenchmarkExecutor,
    HttpBenchmarkRegistry,
)
from roborsi.evaluation.messages import EvaluationResult
from roborsi.evaluation.orchestration import CaseOrchestrator
from roborsi.evaluation.store import ResultStore
from tests.unit_tests.roborsi.evaluation.test_agent_artifact import (
    _COMPONENTS,
    _FACTORY,
    _source,
)
from tests.unit_tests.roborsi.evaluation.test_services import _service


class IndexedResults:
    """Test storage: SQLite result records plus the standard artifact verifier.

    No inheritance or private ResultStore access. Only workspace reservations,
    execution leases and artifact checks are delegated to the file store.
    """

    def __init__(self, root: Path, *, fail_after_commit=False):
        root.mkdir(exist_ok=True)
        self.database = root / "results.sqlite"
        self.files = ResultStore(root / "workspaces")
        self.fail_after_commit = fail_after_commit
        self.publications = 0
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS results "
                "(run TEXT, id TEXT, body TEXT, PRIMARY KEY (run, id))"
            )
        self.claim = self.files.claim
        self.verify_claim = self.files.verify_claim
        self.execution = self.files.execution
        self.start_trace = self.files.start_trace
        self.verify_artifacts = self.files.verify_artifacts
        self.reserve_batch_run = self.files.reserve_batch_run
        self.load_batch_plan = self.files.load_batch_plan

    def publish(self, run_id, case_id, result):
        self.publications += 1
        self.verify_claim(run_id, case_id, result.task, result.provenance)
        self.verify_artifacts(run_id, case_id, result)
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                "INSERT INTO results VALUES (?, ?, ?)",
                (run_id, case_id, json.dumps(result.as_dict(), allow_nan=False)),
            )
        if self.fail_after_commit:
            raise OSError("result notification failed after database commit")

    def load(self, run_id, case_id):
        with sqlite3.connect(self.database) as connection:
            row = connection.execute(
                "SELECT body FROM results WHERE run=? AND id=?", (run_id, case_id)
            ).fetchone()
        if row is None:
            raise FileNotFoundError((run_id, case_id))
        result = EvaluationResult.from_dict(json.loads(row[0]))
        self.verify_claim(run_id, case_id, result.task, result.provenance)
        self.verify_artifacts(run_id, case_id, result)
        return result

    def status(self, run_id, case_id):
        try:
            result = self.load(run_id, case_id)
        except FileNotFoundError:
            return self.files.status(run_id, case_id)
        return CaseStatus(
            run_id=run_id,
            case_id=case_id,
            episode_id=result.episode_id,
            state="cancelled"
            if result.status == "cancelled"
            else "failed"
            if result.error
            else "completed",
            updated_at=timestamp(),
            result_published=True,
            result_status=result.status,
            error=result.error,
        )

    def list_cases(self, run_id):
        return {
            case_id: "published"
            if self.status(run_id, case_id).result_published
            else state
            for case_id, state in self.files.list_cases(run_id).items()
        }


def setup_runner(tmp_path, results, endpoint):
    artifacts = AgentArtifactStore(tmp_path / "agents")
    baseline = artifacts.create(
        _source(tmp_path / "source"), _FACTORY, components=_COMPONENTS
    )
    registry = HttpBenchmarkRegistry({"demo": endpoint})
    cases = expand_cases(registry, ["demo"])

    def runner(artifact):
        orchestrator = CaseOrchestrator(
            registry,
            results,
            artifact.launcher(startup_timeout_s=10),
            HttpBenchmarkExecutor(),
        )
        return SimpleNamespace(run=BatchEvaluator(registry, results, orchestrator).run)

    return artifacts, baseline, cases, runner


def test_committed_database_result_is_not_republished_after_notification_error(
    tmp_path,
):
    results = IndexedResults(tmp_path / "results", fail_after_commit=True)
    with _service("benchmark", factory="make_prompt_benchmark") as service:
        _, baseline, cases, runner = setup_runner(tmp_path, results, service.endpoint)
        with pytest.raises(OSError, match="notification failed after database commit"):
            runner(baseline).run("run1", cases, baseline.version_id)
        assert results.publications == 1
        saved = results.load("run1", cases[0].case_id)
        assert saved.status == "task_failure" and saved.score == 0
        assert results.status("run1", cases[0].case_id).result_published
        assert service.call("healthz", timeout_s=3)["busy"] is False
    assert not list((tmp_path / "results").rglob("result.json"))
