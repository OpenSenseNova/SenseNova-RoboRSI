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

"""Resume real Docker cases after a worker dies before result publication."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

import pytest

from roborsi.benchmarks.registry import BenchmarkRegistry, directory_sha256
from roborsi.evaluation.batch import BatchEvaluator
from roborsi.evaluation.store import ResultStore
from roborsi.utils.config import get_project_env, require_project_env
from tests.integration_tests.robots.libero.batch_worker import (
    AGENT_VERSION,
    CASE_TIMEOUT_S,
    TIMEOUT_S,
    matrix,
)

pytestmark = [
    pytest.mark.skipif(
        get_project_env("ROBORSI_RUN_LIBERO_BENCHMARK") != "1",
        reason="requires the three-step LIBERO HTTP image and assets",
    ),
    pytest.mark.timeout(600),
]


def resources(root, run_id, kind):
    return {
        path.name.removesuffix(f"-{kind}.json"): json.loads(path.read_text())
        for path in (root / "resources" / run_id).glob(f"*-{kind}.json")
    }


def assert_clean(root, run_id, expected):
    agents = resources(root, run_id, "agent")
    containers = resources(root, run_id, "container")
    assert agents.keys() == containers.keys() == set(expected)
    for item in agents.values():
        assert item["exit_code"] is not None
    for item in containers.values():
        assert "closed" in item
        assert not subprocess.check_output(
            ["docker", "ps", "-aq", "--filter", f"id={item['container_id']}"],
            text=True,
            timeout=10,
        ).strip()


def start_worker(root, run_id, *extra):
    command = [
        sys.executable,
        "-m",
        "tests.integration_tests.robots.libero.batch_worker",
        "--registry",
        require_project_env("ROBORSI_LIBERO_REGISTRY"),
        "--output",
        str(root),
        "--run-id",
        run_id,
        *extra,
    ]
    with (root / f"{run_id}.log").open("w") as log:
        return subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )


def test_native_batch_resume_preserves_old_results_and_only_retries_eligible_cases(
    tmp_path,
):
    registry = BenchmarkRegistry([require_project_env("ROBORSI_LIBERO_REGISTRY")])
    cells = matrix(registry)
    ids = [cell.case_id for cell in cells]
    store = ResultStore(tmp_path / "results")
    workers = []
    try:
        first = start_worker(tmp_path, "first", "--initial")
        workers.append(first)
        end = time.monotonic() + 300
        while not (tmp_path / "ready.json").exists():
            assert first.poll() is None, (tmp_path / "first.log").read_text()
            assert time.monotonic() < end, "initial batch did not reach publication"
            time.sleep(0.1)
        assert first.poll() is None
        assert store.status("first", ids[3]).state == "running"
        assert_clean(tmp_path, "first", ids)
        assert store.load("first", ids[0]).status in {"success", "task_failure"}
        transient = store.load("first", ids[1])
        permanent = store.load("first", ids[2])
        assert transient.status == "infrastructure_error" and transient.error.retryable
        assert permanent.status == "agent_error" and not permanent.error.retryable
        with pytest.raises(FileNotFoundError):
            store.load("first", ids[3])

        class MustNotLaunch:
            def run(self, *args, **kwargs):
                pytest.fail("resume launched work before checking the previous worker")

        resume = BatchEvaluator(registry, store, MustNotLaunch())
        with pytest.raises(ValueError, match="active worker"):
            resume.run(
                "blocked",
                cells,
                AGENT_VERSION,
                resume_from="first",
                timeout_s=TIMEOUT_S,
                case_timeout_s=CASE_TIMEOUT_S,
            )
        assert not (store.root / "blocked").exists()
        original = directory_sha256(store.root / "first")
        journal = directory_sha256(store.root / ".case-status" / "first")
        first.kill()
        assert first.wait(timeout=10) == -signal.SIGKILL
        assert store.status("first", ids[3]).state == "interrupted"
        with pytest.raises(ValueError, match="batch inputs differ"):
            resume.run(
                "changed",
                cells,
                AGENT_VERSION,
                resume_from="first",
                timeout_s=TIMEOUT_S + 1,
                case_timeout_s=CASE_TIMEOUT_S,
            )
        assert not (store.root / "changed").exists()

        second = start_worker(tmp_path, "second", "--resume-from", "first")
        workers.append(second)
        assert second.wait(timeout=240) == 0, (tmp_path / "second.log").read_text()
        assert_clean(tmp_path, "second", [ids[1], ids[3]])
        report = json.loads((tmp_path / "second.json").read_text())
        assert [entry["source_run_id"] for entry in report["cases"]] == [
            "first",
            "second",
            "first",
            "second",
        ]
        summary = report["benchmarks"]["libero"]
        successes = sum(entry["result"]["success"] is True for entry in report["cases"])
        assert (
            summary["planned"],
            summary["scored"],
            summary["task_failures"],
            summary["agent_errors"],
            summary["infrastructure_errors"],
        ) == (4, 3, 3 - successes, 1, 0)
        assert summary["aggregate_score"] == summary["success_rate"] == successes / 4
        spans = list(resources(tmp_path, "second", "agent").values())
        assert max(item["started"] for item in spans) < min(
            item["closed"] for item in spans
        )
        for entry in report["cases"]:
            result = store.load(entry["source_run_id"], entry["case_id"])
            assert result.as_dict() == entry["result"]
            assert store.trace_state(entry["source_run_id"], entry["case_id"]).complete
            if result.score is not None:
                directory = store.root / entry["source_run_id"] / entry["case_id"]
                native = json.loads(
                    (directory / result.native_score_source).read_text()
                )
                assert (
                    1 <= native["steps"] <= 3
                    and float(native["native_terminated"]) == result.score
                )
                assert len(list((directory / "media").rglob("*.bin"))) == 2 * (
                    native["steps"] + 1
                )
                events = store.trace_events(entry["source_run_id"], entry["case_id"])
                actions = [
                    e
                    for e in events
                    if e["type"] == "roborsi_message"
                    and e["payload"]["message"]["kind"] == "action"
                ]
                assert len(actions) == native["steps"]
        assert directory_sha256(store.root / "first") == original
        assert directory_sha256(store.root / ".case-status" / "first") == journal

        third = start_worker(tmp_path, "third", "--resume-from", "second")
        workers.append(third)
        assert third.wait(timeout=30) == 0, (tmp_path / "third.log").read_text()
        assert_clean(tmp_path, "third", [])
        again = json.loads((tmp_path / "third.json").read_text())
        assert again["cases"] == report["cases"]
        assert again["benchmarks"] == report["benchmarks"]
        assert store.list_cases("third") == {}
        assert directory_sha256(store.root / "first") == original
    finally:
        # Cleanup for an assertion/timeout failure must not make the assertions pass.
        for worker in workers:
            if worker.poll() is None:
                os.killpg(worker.pid, signal.SIGKILL)
                worker.wait(timeout=10)
        for path in (tmp_path / "resources").rglob("*-container.json"):
            container_id = json.loads(path.read_text())["container_id"]
            subprocess.run(
                ["docker", "rm", "-f", container_id], capture_output=True, timeout=15
            )
