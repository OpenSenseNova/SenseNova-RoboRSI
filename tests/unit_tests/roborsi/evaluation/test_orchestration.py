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

"""One case is claimed before launch and remains queryable after startup errors."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from roborsi.benchmarks.registry import BenchmarkRegistry
from roborsi.evaluation.agent_process import PythonAgentLauncher
from roborsi.evaluation.benchmark_client import BenchmarkClient
from roborsi.evaluation.deadline import CaseDeadline
from roborsi.evaluation.orchestration import CaseOrchestrator
from roborsi.evaluation.services import AdapterClient
from roborsi.evaluation.store import ResultStore
from tests.unit_tests.roborsi.benchmarks.test_registry_contracts import _manifest
from tests.unit_tests.roborsi.evaluation.test_services import _service

_FACTORY = "tests.unit_tests.roborsi.evaluation.service_child:make_demo_agent"


class _AdapterExecutor:
    def __init__(
        self, *, fail: bool = False, factory: str = "make_demo_adapter"
    ) -> None:
        self.fail = fail
        self.factory = factory
        self.saw_claim = False
        self.closed = False

    @contextmanager
    def start_adapter(self, lock, case_dir, *, deadline=None):
        assert lock.benchmark == "demo"
        assert isinstance(deadline, CaseDeadline)
        self.saw_claim = (case_dir / "plan.json").is_file()
        if self.fail:
            raise RuntimeError("container startup failed")
        try:
            if self.factory == "make_media_demo_adapter":
                with _service("benchmark", factory="make_media_demo_benchmark") as rpc:
                    yield SimpleNamespace(client=BenchmarkClient(rpc.endpoint))
            else:
                with _service("adapter", factory=self.factory) as rpc:
                    yield SimpleNamespace(client=AdapterClient(rpc))
        finally:
            self.closed = True


def _orchestrator(tmp_path: Path, executor, launcher):
    _manifest(tmp_path / "benchmarks")
    store = ResultStore(tmp_path / "results")
    registry = BenchmarkRegistry([tmp_path / "benchmarks"])
    return CaseOrchestrator(registry, store, launcher, executor), store


def _run(orchestrator: CaseOrchestrator):
    return orchestrator.run(
        "run1",
        "case1",
        "case-1",
        "demo",
        "place_block",
        0,
        "agent-v1",
        timeout_s=3.0,
    )


def test_orchestrator_drives_two_services_from_a_registry_lock(tmp_path: Path) -> None:
    executor = _AdapterExecutor()
    launcher = PythonAgentLauncher(_FACTORY, startup_timeout_s=10.0)
    orchestrator, store = _orchestrator(tmp_path, executor, launcher)

    result = _run(orchestrator)

    assert executor.saw_claim and executor.closed
    assert result.status == "success" and result.score == 0.75
    assert result.provenance.agent_version == "agent-v1"
    assert result.provenance.docker_image_digest == "sha256:" + "b" * 64
    assert store.load("run1", "case1") == result
    assert store.list_cases("run1") == {"case1": "published"}
    with pytest.raises(FileExistsError):
        _run(orchestrator)


def test_runtime_cleanup_error_is_published_before_trace_finishes(
    tmp_path: Path,
) -> None:
    class BadCleanup(_AdapterExecutor):
        @contextmanager
        def start_adapter(self, lock, case_dir, *, deadline=None):
            with super().start_adapter(lock, case_dir, deadline=deadline) as adapter:
                yield adapter
            raise RuntimeError("container cleanup failed")

    executor = BadCleanup()
    orchestrator, store = _orchestrator(
        tmp_path, executor, PythonAgentLauncher(_FACTORY, startup_timeout_s=10.0)
    )
    result = _run(orchestrator)
    assert executor.closed
    assert result.status == "infrastructure_error" and result.score is None
    assert "container cleanup failed" in str(result.error.details)
    assert store.load("run1", "case1") == result
    assert store.trace_state("run1", "case1").complete
    assert (
        store.trace_events("run1", "case1")[-1]["payload"]["status"]
        == "infrastructure_error"
    )
    assert store.list_cases("run1") == {"case1": "published"}
    with pytest.raises(FileExistsError):
        _run(orchestrator)


def test_orchestrator_records_adapter_startup_failure(tmp_path: Path) -> None:
    executor = _AdapterExecutor(fail=True)
    orchestrator, store = _orchestrator(
        tmp_path, executor, PythonAgentLauncher(_FACTORY)
    )

    result = _run(orchestrator)

    assert executor.saw_claim
    assert result.status == "infrastructure_error"
    assert (
        result.error is not None and result.error.details["stage"] == "adapter.startup"
    )
    assert store.load("run1", "case1") == result
    events = store.trace_events("run1", "case1")
    assert events[-1]["payload"]["status"] == "infrastructure_error"
    errors = [
        event["payload"]["message"]["payload"]
        for event in events
        if event["type"] == "roborsi_message"
    ]
    assert errors[0]["details"]["stage"] == "adapter.startup"


def test_orchestrator_records_agent_startup_failure_and_closes_adapter(
    tmp_path: Path,
) -> None:
    class FailedLauncher:
        def start(self, case_dir, *, deadline=None, media_dir=None):
            assert case_dir.is_dir()
            assert isinstance(deadline, CaseDeadline)
            assert media_dir.is_dir()
            raise RuntimeError("agent cannot start")

    executor = _AdapterExecutor()
    orchestrator, store = _orchestrator(tmp_path, executor, FailedLauncher())

    result = _run(orchestrator)

    assert executor.closed
    assert result.status == "agent_error"
    assert result.error is not None and result.error.details["stage"] == "agent.startup"
    assert store.load("run1", "case1") == result


def test_orchestrator_records_agent_interaction_failure(tmp_path: Path) -> None:
    executor = _AdapterExecutor()
    launcher = PythonAgentLauncher(
        "tests.unit_tests.roborsi.evaluation.service_child:make_failing_agent",
        startup_timeout_s=10.0,
    )
    orchestrator, store = _orchestrator(tmp_path, executor, launcher)

    result = _run(orchestrator)

    assert executor.closed
    assert result.status == "agent_error"
    assert result.error is not None and result.error.details["stage"] == "agent.act"
    assert store.load("run1", "case1") == result


def test_orchestrator_publishes_cancellation_during_agent_startup(
    tmp_path: Path,
) -> None:
    cancelled = Event()

    class CancellingLauncher:
        def start(self, case_dir, *, deadline, media_dir):
            assert media_dir.is_dir()
            cancelled.set()
            deadline.check()

    executor = _AdapterExecutor()
    orchestrator, store = _orchestrator(tmp_path, executor, CancellingLauncher())

    result = orchestrator.run(
        "run1",
        "case1",
        "case-1",
        "demo",
        "place_block",
        0,
        "agent-v1",
        cancel_event=cancelled,
    )

    assert executor.closed
    assert result.status == "cancelled"
    assert result.error is not None and result.error.code == "cancelled"
    assert store.load("run1", "case1") == result


def test_orchestrator_transfers_two_camera_images_by_reference(
    tmp_path: Path,
) -> None:
    executor = _AdapterExecutor(factory="make_media_demo_adapter")
    launcher = PythonAgentLauncher(
        "tests.unit_tests.roborsi.evaluation.service_child:make_media_demo_agent",
        startup_timeout_s=10.0,
    )
    orchestrator, store = _orchestrator(tmp_path, executor, launcher)

    result = _run(orchestrator)

    assert result.status == "success" and result.score == 0.75
    assert executor.closed
    directory = tmp_path / "results" / "run1" / "case1"
    assert len(list((directory / "media").rglob("*.bin"))) == 2
    assert store.load("run1", "case1") == result
    from roborsi.evaluation.trajectory import TrajectoryReader

    reader = TrajectoryReader(directory, trace_manifest=result.trace_manifest)
    index = reader.index()
    assert index["summary"]["status"] == "success"
    assert index["summary"]["n_turns"] == 1
    assert len(index["media"]) == 2
    assert all(media["available"] for media in index["media"])
    assert (
        reader.turn("step-0")["events"][0]["payload"]["observation"]["step_index"] == 0
    )
