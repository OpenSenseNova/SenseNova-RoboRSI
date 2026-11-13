# Copyright 2026 The RoboRSI Contributors.
# Licensed under the Apache License, Version 2.0.

"""Run the common RoboRSI CLI with each task's matching frozen Agent resource."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from benchmarks.bench2dex.agent_bundle import prepare
from roborsi.evaluation.store import ResultStore
from robots.bench2dex.contracts import TASKS, validate_task
from robots.bench2dex.icl import SETTING
from robots.bench2dex.reference.core import write_json


def summarize(work: Path, plan: dict) -> dict:
    """Withhold the evaluation mean until every planned case has valid evidence."""
    store = ResultStore(work / "results")
    expected = {(task, sample) for task in plan["tasks"] for sample in plan["samples"]}
    valid, issues, records = {}, [], []
    for task in plan["tasks"]:
        run_id = f"task_{int(task):02d}"
        for case, status in store.list_cases(run_id).items():
            if status != "published":
                issues.append({"case": case, "reason": "unpublished"})
                continue
            try:
                result = store.load(run_id, case)
                validate_task(result.task)
                identity = (result.task.task_id, result.task.seed)
                trace = store.trace_state(run_id, case)
                if (
                    identity not in expected
                    or identity in valid
                    or result.task.native["smoke_only"]
                    or result.status not in {"success", "task_failure"}
                    or not result.native_score_source
                    or trace is None
                    or not trace.complete
                ):
                    raise ValueError(
                        "case is diagnostic, incomplete, duplicate or lacks native evidence"
                    )
                valid[identity] = result.score
                records.append(
                    {
                        "task": identity[0],
                        "sample": identity[1],
                        "status": result.status,
                        "success": result.success,
                        "score": result.score,
                    }
                )
            except (ValueError, OSError) as exc:
                issues.append({"case": case, "reason": str(exc)})
    missing = sorted(expected - set(valid))
    complete = not missing and not issues
    return {
        "setting": SETTING,
        "complete": complete,
        "expected": len(expected),
        "valid": len(valid),
        "missing": [{"task": t, "sample": s} for t, s in missing],
        "issues": issues,
        "success_rate": sum(valid.values()) / len(expected) if complete else None,
        "cases": records,
        "note": "RoboRSI's partial mean is operational feedback; this complete-plan mean preserves the standalone reporting rule.",
    }


def main() -> None:
    """Freeze selection before execution; valid scored failures are never retried."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument(
        "--plan", type=Path, default=Path(__file__).with_name("plan-78.json")
    )
    parser.add_argument("--benchmark-url")
    parser.add_argument("--benchmark-token-env")
    origin = parser.add_mutually_exclusive_group()
    origin.add_argument("--icl-source", type=Path)
    origin.add_argument("--icl-repo")
    parser.add_argument("--icl-revision", default="main")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    plan = json.loads(
        (args.work / "plan.json" if args.summarize else args.plan).read_text()
    )
    if (
        set(plan)
        != {"schema_version", "condition", "tasks", "samples", "expected_cases"}
        or plan["schema_version"] != 1
        or plan["condition"] != "standard_1x"
        or not plan["tasks"]
        or not plan["samples"]
        or any(t not in TASKS for t in plan["tasks"])
        or len(set(plan["tasks"])) != len(plan["tasks"])
        or len(set(plan["samples"])) != len(plan["samples"])
        or any(type(s) is not int or s < 0 for s in plan["samples"])
        or plan["expected_cases"] != len(plan["tasks"]) * len(plan["samples"])
    ):
        parser.error("invalid fixed evaluation plan")
    if args.summarize:
        report = summarize(args.work, plan)
        write_json(args.work / "evaluation-summary.json", report)
        print(json.dumps(report, indent=2))
        return
    if not args.benchmark_url or not (args.icl_source or args.icl_repo):
        parser.error("execution needs --benchmark-url and an ICL source")
    args.work.mkdir(parents=True, exist_ok=False)
    write_json(args.work / "plan.json", plan)
    for task in plan["tasks"]:
        config = prepare(
            task,
            args.work / f"agent-task-{int(task):02d}",
            source=args.icl_source,
            repo_id=args.icl_repo,
            revision=args.icl_revision,
        )
        command = [
            sys.executable,
            "-m",
            "roborsi.evaluation.cli",
            "eval",
            "--benchmark-url",
            "bench2dex=" + args.benchmark_url,
            "--benchmark",
            "bench2dex",
            "--task",
            "bench2dex:" + task,
            "--agent-config",
            str(config),
            "--results-root",
            str(args.work / "results"),
            "--run-id",
            f"task_{int(task):02d}",
            "--max-concurrent",
            "1",
            "--timeout-s",
            "86400",
            "--case-timeout-s",
            "86400",
        ]
        for sample in plan["samples"]:
            command.extend(["--seed", str(sample)])
        if args.benchmark_token_env:
            command.extend(["--benchmark-token-env", args.benchmark_token_env])
        if args.dry_run:
            command.append("--dry-run")
        with (args.work / f"task-{int(task):02d}.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        task_report = None
        if not args.dry_run:
            task_report = summarize(
                args.work,
                {**plan, "tasks": [task], "expected_cases": len(plan["samples"])},
            )
        if result.returncode or (
            task_report is not None and not task_report["complete"]
        ):
            exit_code = result.returncode or 1
            failure = {
                "task": task,
                "exit_code": exit_code,
                "subprocess_exit_code": result.returncode,
                "reason": "evaluation command failed"
                if result.returncode
                else "task has missing or invalid results",
            }
            if task_report is not None:
                failure.update(
                    missing=task_report["missing"], issues=task_report["issues"]
                )
                write_json(
                    args.work / "evaluation-summary.json", summarize(args.work, plan)
                )
            write_json(
                args.work / "execution-failure.json",
                failure,
            )
            parser.exit(
                exit_code,
                f"Task {task}: {failure['reason']}; "
                f"see {args.work / 'execution-failure.json'}\n",
            )
    if not args.dry_run:
        write_json(args.work / "evaluation-summary.json", summarize(args.work, plan))


if __name__ == "__main__":
    main()
