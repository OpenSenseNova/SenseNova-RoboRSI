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

"""Use the public Agent preset and bench eval entry points for navigation runs."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from roborsi.evaluation.messages import EvaluationResult
from robots.vln_shared.benchmark import NavigationBenchmark, serve
from robots.vln_shared.progress import Progress
from robots.vln_shared.storage import file_digest, read_json, write_json


def run_process(
    command: list[str], directory: Path, cancel: threading.Event, timeout: float
) -> int:
    """Capture a framework subprocess and reap its process group on interruption."""
    with (
        (directory / "stdout.log").open("wb") as stdout,
        (directory / "stderr.log").open("wb") as stderr,
    ):
        process = subprocess.Popen(
            command, stdout=stdout, stderr=stderr, start_new_session=True
        )
        write_json(
            directory / "process.json",
            {"pid": process.pid, "birth": _birth(process.pid)},
        )
        deadline = time.monotonic() + timeout
        try:
            while process.poll() is None:
                if cancel.wait(0.2):
                    raise InterruptedError("campaign cancelled")
                if time.monotonic() >= deadline:
                    raise TimeoutError("framework process exceeded case timeout")
            return process.returncode
        finally:
            # The group can contain a native CLI even when its launcher has exited.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def _birth(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def stop_abandoned(directory: Path) -> None:
    """Reap a previous launch only when its recorded PID still has the same birth."""
    path = directory / "process.json"
    if not path.exists():
        return
    record = read_json(path)
    pid, birth = record["pid"], record["birth"]
    if birth is None or _birth(pid) != birth:
        return
    try:
        os.killpg(pid, signal.SIGTERM)
        end = time.monotonic() + 5
        while _birth(pid) == birth and time.monotonic() < end:
            time.sleep(0.05)
        if _birth(pid) == birth:
            os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


class BenchExecutor:
    """Run a single task through common HTTP, Agent lifecycle, scoring and artifacts."""

    def __init__(
        self,
        datasets: dict,
        presets: dict[str, Path],
        cancel: threading.Event,
        *,
        timeout: float = 7200,
        progress: Progress | None = None,
    ) -> None:
        self.datasets, self.presets, self.cancel, self.timeout = (
            datasets,
            presets,
            cancel,
            timeout,
        )
        self.progress = progress

    def __call__(
        self, case: dict, budget: int, snapshot: Path, directory: Path
    ) -> dict:
        """Launch a fresh framework run; parse native evidence rather than exit status."""
        spec = self.datasets[case["dataset"]]["test"]["spec"]
        provider = NavigationBenchmark(
            spec,
            [case],
            budget,
            on_progress=(lambda steps: self.progress.advance(case, budget, steps))
            if self.progress
            else None,
        )
        task = provider.tasks()[0]
        results = directory / "bench"
        with serve(provider, directory / "service") as endpoint:
            command = [
                sys.executable,
                "-m",
                "roborsi.evaluation.cli",
                "eval",
                "--benchmark-url",
                f"{provider.name}={endpoint}",
                "--benchmark",
                provider.name,
                "--task",
                f"{provider.name}:{task.task_id}",
                "--seed",
                str(task.seed),
                "--agent-config",
                str(self.presets[case["dataset"]]),
                "--results-root",
                str(results),
                "--run-id",
                "evaluation",
                "--max-concurrent",
                "1",
                "--timeout-s",
                "600",
                "--case-timeout-s",
                str(self.timeout),
            ]
            code = run_process(command, directory, self.cancel, self.timeout + 120)
        if code != 0:
            raise RuntimeError(f"bench eval exited {code}; see stderr.log")
        report = json.loads((directory / "stdout.log").read_text())
        if len(report["cases"]) != 1:
            raise ValueError("single-case report required")
        result = EvaluationResult.from_dict(report["cases"][0]["result"])
        return self._result(case, budget, directory, result)

    def recover(self, case: dict, budget: int, directory: Path) -> dict | None:
        """Recover a committed framework result after a scheduler crash."""
        stop_abandoned(directory)
        for path in (directory / "bench").glob("evaluation/case-*/result.json"):
            document = read_json(path)
            if "provenance" in document and "status" in document and "task" in document:
                return {
                    **self._result(
                        case, budget, directory, EvaluationResult.from_dict(document)
                    ),
                    "completed_at": path.stat().st_mtime,
                }
        return None

    def _result(
        self, case: dict, budget: int, directory: Path, result: EvaluationResult
    ) -> dict:
        spec = self.datasets[case["dataset"]]["test"]["spec"]
        task = NavigationBenchmark(spec, [case], budget).tasks()[0]
        results = directory / "bench"
        if result.task != task:
            raise ValueError("bench returned a different task")
        outcome = {
            "status": result.status,
            "success": result.success,
            "score": result.score,
            "evaluation": result.as_dict(),
        }
        stats = [
            read_json(p)["stats"]
            for p in results.glob(
                "evaluation/case-*/agent/codex-cli/call-*/result.json"
            )
            if "stats" in read_json(p)
        ]
        complete = bool(stats) and all(
            s.get("usage_status") == "complete" for s in stats
        )
        outcome["usage"] = {
            "status": "complete" if complete else "unavailable",
            "input_tokens": sum(s["total_input_tokens"] for s in stats)
            if complete
            else None,
            "output_tokens": sum(s["total_output_tokens"] for s in stats)
            if complete
            else None,
        }
        if result.status in {"success", "task_failure"}:
            if not result.native_score_source:
                raise ValueError(
                    "scored evaluation is missing its original native result"
                )
            matches = list(results.glob("evaluation/case-*/result.json"))
            candidates = [
                p.parent / result.native_score_source
                for p in matches
                if (p.parent / result.native_score_source).is_file()
            ]
            if (
                len(candidates) != 1
                or file_digest(candidates[0]) != result.native_score_sha256
            ):
                raise ValueError(
                    "native score evidence is absent, ambiguous or changed"
                )
            outcome["native"] = read_json(candidates[0])
        return outcome
