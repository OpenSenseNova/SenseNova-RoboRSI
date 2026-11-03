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

"""Restartable 500-step evaluation with optional fresh 5000-step failure reruns."""

from __future__ import annotations

import fcntl
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from robots.vln_shared.progress import Progress
from robots.vln_shared.storage import (
    digest,
    file_digest,
    read_json,
    verify_snapshot,
    write_json,
)

SCORED = {"success", "task_failure"}


def campaign_source_identity() -> str:
    """Track all navigation modules, including Agent orchestration, for resume."""
    root = Path(__file__).resolve().parents[1]
    return digest(
        {
            str(p.relative_to(root)): file_digest(p)
            for name in ("vln_shared", "vlnce", "lhpr_vln")
            for p in sorted((root / name).glob("*.py"))
        }
    )


def validate_outcome(result: dict) -> dict:
    """Require native evidence before an attempt can control the next stage."""
    status = result.get("status")
    if status in SCORED:
        score = result.get("score")
        native = result.get("native")
        if (
            result.get("success") is not (status == "success")
            or type(score) not in (int, float)
            or not math.isfinite(score)
            or not isinstance(native, dict)
            or native.get("success") is not result["success"]
            or not (native.get("terminated") or native.get("truncated"))
            or type(native.get("steps")) is not int
            or native["steps"] < 0
        ):
            raise ValueError("scored attempt needs consistent terminal native evidence")
    elif status not in {"agent_error", "infrastructure_error", "cancelled"}:
        raise ValueError("unknown attempt status")
    elif result.get("success") is not None or result.get("score") is not None:
        raise ValueError("technical errors must remain unscored")
    return result


class Campaign:
    """Own the output lock, attempt journal, stage barriers and selected outcomes.

    The backend receives a new attempt directory and the same frozen snapshot
    for every attempt. Completed outcomes are committed before queue bookkeeping;
    resume recovers that commit without executing the case again.
    """

    def __init__(
        self,
        root: Path,
        identity: dict,
        cases: list[dict],
        snapshots: dict[str, Path],
        execute: Callable[[dict, int, Path, Path], dict],
        *,
        concurrency: int = 4,
        retries: int = 1,
        cancel: threading.Event | None = None,
        only_500: bool = False,
        progress: Progress | None = None,
    ) -> None:
        if concurrency < 1 or retries < 0 or not cases:
            raise ValueError(
                "positive concurrency, nonnegative retries and cases required"
            )
        keys = [case["id"] for case in cases]
        if len(set(keys)) != len(keys) or any(
            not key
            or any(
                c
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                for c in key
            )
            for key in keys
        ):
            raise ValueError("case IDs must be unique safe basenames")
        self.root, self.cases, self.snapshots = root, cases, snapshots
        self.execute, self.concurrency, self.retries = execute, concurrency, retries
        self.cancel = cancel or threading.Event()
        self.only_500 = only_500
        self.progress = progress
        self.identity = {
            "schema_version": 1,
            "inputs": identity,
            "cases": cases,
            "budgets": [500] if only_500 else [500, 5000],
            "snapshots": {k: verify_snapshot(v) for k, v in snapshots.items()},
        }
        self.identity_digest = digest(self.identity)

    def _outcome(self, case: dict, budget: int) -> dict | None:
        directory = self.root / "cases" / case["id"] / str(budget)
        for attempt in sorted(directory.glob("attempt-*")):
            path = attempt / "outcome.json"
            if not path.exists():
                recover = getattr(self.execute, "recover", None)
                request = read_json(attempt / "request.json")
                if request["identity"] != self._attempt_identity(case, budget):
                    raise ValueError("unfinished attempt identity changed")
                result = recover(case, budget, attempt) if recover else None
                if result is None:
                    continue
                validate_outcome(result)
                result["wall_seconds"] = max(
                    0, result.pop("completed_at") - request["started_at"]
                )
                result["recovered"] = True
                write_json(
                    path,
                    {
                        "identity": request["identity"],
                        "result": result,
                        "sha256": digest(result),
                    },
                )
            record = read_json(path)
            expected = self._attempt_identity(case, budget)
            if record["identity"] != expected or record["sha256"] != digest(
                record["result"]
            ):
                raise ValueError(f"attempt identity or result changed: {path}")
            result = validate_outcome(record["result"])
            if result["status"] in SCORED:
                return {**result, "attempt": str(path.parent.relative_to(self.root))}
        return None

    def _attempt_identity(self, case: dict, budget: int) -> dict:
        return {
            "campaign": self.identity_digest,
            "case": case["id"],
            "budget": budget,
            "snapshot": self.identity["snapshots"][case["dataset"]],
        }

    def _run_case(self, case: dict, budget: int) -> None:
        if previous := self._outcome(case, budget):
            if self.progress:
                self.progress.reuse(case, budget, previous)
            return
        parent = self.root / "cases" / case["id"] / str(budget)
        parent.mkdir(parents=True, exist_ok=True)
        for retry in range(self.retries + 1):
            if self.cancel.is_set():
                return
            numbers = [
                int(p.name.removeprefix("attempt-")) for p in parent.glob("attempt-*")
            ]
            directory = parent / f"attempt-{max(numbers, default=0) + 1:04d}"
            directory.mkdir()
            snapshot = self.snapshots[case["dataset"]]
            if verify_snapshot(snapshot) != self.identity["snapshots"][case["dataset"]]:
                raise ValueError("exploration changed before attempt")
            identity = self._attempt_identity(case, budget)
            write_json(
                directory / "request.json",
                {"identity": identity, "case": case, "started_at": time.time()},
            )
            started = time.monotonic()
            if self.progress:
                self.progress.start(case, budget, directory, retry, self.retries)
            try:
                result = validate_outcome(
                    self.execute(case, budget, snapshot, directory)
                )
                if verify_snapshot(snapshot) != identity["snapshot"]:
                    raise ValueError("exploration changed during attempt")
            except Exception as exc:
                result = {
                    "status": "infrastructure_error",
                    "success": None,
                    "score": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            result = {**result, "wall_seconds": time.monotonic() - started}
            write_json(
                directory / "outcome.json",
                {
                    "identity": identity,
                    "result": result,
                    "sha256": digest(result),
                },
            )
            if self.progress:
                self.progress.finish(case, budget, result)
            if result["status"] in SCORED:
                return

    def run(self, *, resume: bool = False) -> dict:
        """Execute enabled stages under an exclusive, nonblocking campaign lock."""
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "campaign.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = self.root / "identity.json"
            if path.exists():
                if not resume:
                    raise ValueError("output already initialized; use --resume")
                if read_json(path) != self.identity:
                    raise ValueError("resume inputs, code, catalog or snapshot changed")
            else:
                write_json(path, self.identity)
            self._stage(self.cases, 500)
            failed = [
                c
                for c in self.cases
                if not self.only_500
                and (self._outcome(c, 500) or {}).get("status") == "task_failure"
            ]
            write_json(
                self.root / "stage-5000.json",
                {"enabled": not self.only_500, "cases": [c["id"] for c in failed]},
            )
            if not self.only_500 and not self.cancel.is_set():
                self._stage(failed, 5000)
            report = self.summary()
            write_json(self.root / "summary.json", report)
            return report

    def _stage(self, cases: list[dict], budget: int) -> None:
        if self.progress:
            self.progress.phase(f"event=stage_start budget={budget} cases={len(cases)}")
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = [pool.submit(self._run_case, case, budget) for case in cases]
            for count, future in enumerate(as_completed(futures), 1):
                future.result()
                if self.progress:
                    self.progress.phase(
                        f"event=stage_progress budget={budget} processed={count}/{len(cases)}"
                    )

    def summary(self) -> dict:
        """Select a whole scored run, retaining both stages and technical costs."""
        rows = []
        for case in self.cases:
            first = self._outcome(case, 500)
            second = None if self.only_500 else self._outcome(case, 5000)
            selected = (
                first
                if first and (self.only_500 or first["success"])
                else second
                if first
                else None
            )
            attempts = [
                read_json(p)["result"]
                for p in (self.root / "cases" / case["id"]).glob(
                    "*/attempt-*/outcome.json"
                )
            ]
            valid = [r for r in (first, second) if r]
            rows.append(
                {
                    "id": case["id"],
                    "dataset": case["dataset"],
                    "stage_500": first,
                    "stage_5000": second,
                    "selected": selected,
                    "policy_steps": sum(r["native"]["steps"] for r in valid),
                    "policy_tokens": _tokens(valid),
                    "technical_tokens": _tokens(
                        [r for r in attempts if r["status"] not in SCORED]
                    ),
                    "policy_wall_seconds": sum(r["wall_seconds"] for r in valid),
                    "technical_wall_seconds": sum(
                        r["wall_seconds"] for r in attempts if r["status"] not in SCORED
                    ),
                }
            )
        groups = {}
        for dataset in sorted({row["dataset"] for row in rows}):
            subset = [r for r in rows if r["dataset"] == dataset]
            scored = [r["selected"] for r in subset if r["selected"]]
            successful = sum(r["success"] for r in scored)
            groups[dataset] = {
                "requested": len(subset),
                "scored": len(scored),
                "incomplete": len(subset) - len(scored),
                "successful": successful,
                "success_rate_scored": successful / len(scored) if scored else None,
                "success_rate": successful / len(subset)
                if len(scored) == len(subset)
                else None,
                "stage_500": _stage_counts([r["stage_500"] for r in subset]),
                "stage_5000": _stage_counts([r["stage_5000"] for r in subset]),
            }
        return {
            "complete": all(r["selected"] is not None for r in rows),
            "budgets": self.identity["budgets"],
            "datasets": groups,
            "cases": rows,
        }


def _tokens(results: list[dict]) -> dict | None:
    if not all(r.get("usage", {}).get("status") == "complete" for r in results):
        return None
    return {
        key: sum(r["usage"][key] for r in results)
        for key in ("input_tokens", "output_tokens")
    }


def _stage_counts(results: list[dict | None]) -> dict:
    scored = [r for r in results if r]
    successes = sum(r["success"] for r in scored)
    return {
        "scored": len(scored),
        "successful": successes,
        "success_rate_scored": successes / len(scored) if scored else None,
    }
