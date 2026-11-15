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

"""Subprocess driver for real LIBERO batch interruption and resume checks."""

from __future__ import annotations

import argparse
import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from roborsi.benchmarks.registry import BenchmarkRegistry
from roborsi.evaluation.agent_process import PythonAgentLauncher
from roborsi.evaluation.batch import BatchEvaluator, expand_cases
from roborsi.evaluation.docker_runtime import DockerExecutor
from roborsi.evaluation.orchestration import CaseOrchestrator
from roborsi.evaluation.result import write_json_atomic
from roborsi.evaluation.store import ResultStore

FACTORY = "tests.integration_tests.robots.libero.fault_agent:make_agent"
AGENT_VERSION = "libero-resume-fault-agent-v1"
TIMEOUT_S = 120.0
CASE_TIMEOUT_S = 300.0


def matrix(registry):
    cells = expand_cases(
        registry, ["libero"], task_ids={"libero": ["0", "1", "2", "3"]}, seeds=[0]
    )
    for cell in cells:
        assert (
            registry.lock("libero").task_message(cell.task_id, cell.seed).max_steps == 3
        )
    return cells


class Resources:
    def __init__(self, root: Path, run_id: str):
        self.root = root / "resources" / run_id
        self.root.mkdir(parents=True)

    def record(self, case_id: str, kind: str, **values):
        write_json_atomic(self.root / f"{case_id}-{kind}.json", values)


class TrackedDocker:
    def __init__(self, resources: Resources):
        self.resources = resources
        self.executor = DockerExecutor()

    @contextmanager
    def start_adapter(self, lock, case_dir, *, deadline):
        container = self.executor.start_adapter(lock, case_dir, deadline=deadline)
        record = {"container_id": container.container_id, "started": time.monotonic()}
        try:
            self.resources.record(case_dir.name, "container", **record)
            yield container
        finally:
            container.close()
            self.resources.record(
                case_dir.name, "container", **record, closed=time.monotonic()
            )


class TrackedAgent:
    def __init__(self, resources: Resources, fault_modes: dict, *, concurrent: bool):
        self.resources, self.fault_modes = resources, fault_modes
        self.barrier = threading.Barrier(2) if concurrent else None

    @contextmanager
    def start(self, case_dir, *, deadline, media_dir):
        case_id = case_dir.parent.name
        launcher = PythonAgentLauncher(
            FACTORY,
            env_overrides={"ROBORSI_TEST_FAULT": self.fault_modes.get(case_id, "none")},
        )
        process = launcher.start(case_dir, deadline=deadline, media_dir=media_dir)
        record = {"started": time.monotonic()}
        try:
            self.resources.record(case_id, "agent", **record)
            if self.barrier is not None:
                # Both resumed cases must own a live Agent before either acts.
                self.barrier.wait(timeout=120)
            yield process
        finally:
            process.close()
            self.resources.record(
                case_id,
                "agent",
                **record,
                closed=time.monotonic(),
                exit_code=process.daemon.poll(),
            )


class PausingStore(ResultStore):
    def __init__(self, root: Path, pause_case: str | None, ready: Path):
        super().__init__(root)
        self.pause_case, self.ready = pause_case, ready

    def publish(self, run_id, case_id, result):
        if case_id == self.pause_case:
            # CaseRunner has already downloaded evidence and closed both runtimes.
            # Keep the execution lease while the parent tests the live-worker gate.
            assert result.score is not None, result.as_dict()
            write_json_atomic(self.ready, {"run_id": run_id, "case_id": case_id})
            threading.Event().wait()
        return super().publish(run_id, case_id, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume-from")
    parser.add_argument("--initial", action="store_true")
    args = parser.parse_args()
    registry = BenchmarkRegistry([args.registry])
    cells = matrix(registry)
    resources = Resources(args.output, args.run_id)
    modes = (
        {cells[1].case_id: "agent_crash", cells[2].case_id: "agent_error"}
        if args.initial
        else {}
    )
    store = PausingStore(
        args.output / "results",
        cells[3].case_id if args.initial else None,
        args.output / "ready.json",
    )
    evaluator = BatchEvaluator(
        registry,
        store,
        CaseOrchestrator(
            registry,
            store,
            TrackedAgent(resources, modes, concurrent=not args.initial),
            TrackedDocker(resources),
        ),
    )
    report = evaluator.run(
        args.run_id,
        cells,
        AGENT_VERSION,
        max_concurrent=1 if args.initial else 2,
        timeout_s=TIMEOUT_S,
        case_timeout_s=CASE_TIMEOUT_S,
        resume_from=args.resume_from,
    )
    write_json_atomic(args.output / f"{args.run_id}.json", report.as_dict())
    print(json.dumps({"run_id": args.run_id, "cases": len(report.cases)}), flush=True)


if __name__ == "__main__":
    main()
