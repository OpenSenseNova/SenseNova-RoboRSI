# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Run a complete three-LIBERO migration subset in one deployment image."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from benchmarks.libero_three_combined.run_combined import _CASES
from roborsi.agents.config import AgentConfig
from roborsi.evaluation.result import write_json_atomic
from roborsi.evaluation.store import ResultStore

_EXPECTED = {
    "libero_long_plus_plus": 12,
    "libero_ultra_long": 9,
    "libero_pro": 80,
}
_VALID_TERMINALS = {"success", "task_failure"}
_ASSET_MANIFEST = Path("/opt/roborsi-agent-assets/asset.json")
_COMPLETION = "attempt-completion.json"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(document: dict[str, Any]) -> str:
    encoded = json.dumps(
        document, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _object(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"expected JSON object: {path}")
    return document


def _tasks(repository: Path, benchmark: str) -> tuple[str, ...]:
    path = repository / "configs/agents/libero/context_maps" / f"{benchmark}.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    packages = document.get("packages") if isinstance(document, dict) else None
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != 1
        or not isinstance(packages, dict)
    ):
        raise ValueError(f"invalid repository context map: {path}")
    tasks = tuple(sorted(packages))
    if len(tasks) != _EXPECTED[benchmark]:
        raise ValueError(f"{benchmark} context map needs {_EXPECTED[benchmark]} tasks")
    return tasks


def _validated_result(
    root: Path, *, benchmark: str, task_id: str, seed: int
) -> dict[str, Any]:
    evaluation = root / "evaluation"
    matches = list(evaluation.glob("*/*/result.json"))
    if len(matches) != 1:
        raise ValueError(f"expected one published result, found {len(matches)}")
    result_path = matches[0]
    relative = result_path.relative_to(evaluation)
    if len(relative.parts) != 3:
        raise ValueError("published result has an unexpected storage path")
    run_id, case_id, _ = relative.parts
    result = ResultStore(evaluation).load(run_id, case_id)
    if (
        result.task.benchmark != benchmark
        or result.task.task_id != task_id
        or result.task.seed != seed
    ):
        raise ValueError("published result belongs to another task or seed")
    if result.status not in _VALID_TERMINALS:
        raise ValueError(f"published result is not a native terminal: {result.status}")
    if result.native_score_source is None or result.native_score_sha256 is None:
        raise ValueError("published result lacks native score evidence")
    return {
        "status": result.status,
        "success": result.success,
        "score": result.score,
        "error": result.error.as_dict() if result.error is not None else None,
        "task": result.task.as_dict(),
        "provenance": result.provenance.as_dict(),
        "native_score_source": result.native_score_source,
        "native_score_sha256": result.native_score_sha256,
        "result_path": relative.as_posix(),
        "result_sha256": _file_sha256(result_path),
    }


def _record_result(
    root: Path, *, benchmark: str, task_id: str, seed: int
) -> dict[str, Any]:
    try:
        return _validated_result(root, benchmark=benchmark, task_id=task_id, seed=seed)
    except (OSError, ValueError) as exc:
        return {"status": "invalid", "validation_error": str(exc)}


def _conditions(
    repository: Path,
    benchmark: str,
    *,
    base_url: str,
    timeout_s: float,
    case_timeout_s: float,
) -> dict[str, Any]:
    environment = {}
    for name in (
        "ROBORSI_IMAGE_REVISION",
        "ROBORSI_AGENT_IMAGE",
        "ROBORSI_BENCHMARK_IMAGE",
    ):
        value = os.environ.get(name)
        if not value:
            raise ValueError(f"combined image is missing {name}")
        environment[name] = value
    asset_manifest = Path(
        os.environ.get("ROBORSI_AGENT_ASSET_MANIFEST", str(_ASSET_MANIFEST))
    )
    assets = _object(asset_manifest)
    if assets.get("schema_version") != 1:
        raise ValueError("Agent asset manifest has an unsupported schema")
    bundle = _CASES[benchmark][0]
    config = AgentConfig.load(bundle)
    return {
        "schema_version": 1,
        "source_revision": environment["ROBORSI_IMAGE_REVISION"],
        "agent_parent_image": environment["ROBORSI_AGENT_IMAGE"],
        "benchmark_parent_image": environment["ROBORSI_BENCHMARK_IMAGE"],
        "agent_config": bundle,
        "agent_config_sha256": config.fingerprint,
        "agent_assets": assets,
        "agent_assets_manifest_sha256": _file_sha256(asset_manifest),
        "model_endpoint_sha256": hashlib.sha256(
            base_url.rstrip("/").encode()
        ).hexdigest(),
        "run_combined_sha256": _file_sha256(
            repository / "benchmarks/libero_three_combined/run_combined.py"
        ),
        "subset_runner_sha256": _file_sha256(Path(__file__).resolve()),
        "context_map_sha256": _file_sha256(
            repository / "configs/agents/libero/context_maps" / f"{benchmark}.json"
        ),
        "profile": "migration_subset",
        "plus9_on_failure": False,
        "seed": 0,
        "timeout_s": timeout_s,
        "case_timeout_s": case_timeout_s,
    }


def _completed_attempt(
    case_root: Path, *, benchmark: str, task_id: str, condition_sha256: str
) -> tuple[Path, dict[str, Any]] | None:
    for attempt in sorted(case_root.glob("attempt-*"), reverse=True):
        try:
            completion = _object(attempt / _COMPLETION)
            if (
                completion.get("schema_version") != 1
                or completion.get("benchmark") != benchmark
                or completion.get("task_id") != task_id
                or completion.get("seed") != 0
                or completion.get("condition_sha256") != condition_sha256
                or completion.get("returncode") != 0
            ):
                continue
            record = _validated_result(
                attempt, benchmark=benchmark, task_id=task_id, seed=0
            )
            if (
                completion.get("result_path") != record["result_path"]
                or completion.get("result_sha256") != record["result_sha256"]
            ):
                continue
            return attempt, record
        except (OSError, ValueError):
            continue
    return None


def _write_completion(
    attempt: Path,
    *,
    benchmark: str,
    task_id: str,
    condition_sha256: str,
    returncode: int,
    record: dict[str, Any],
) -> None:
    attempt.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        attempt / _COMPLETION,
        {
            "schema_version": 1,
            "benchmark": benchmark,
            "task_id": task_id,
            "seed": 0,
            "condition_sha256": condition_sha256,
            "returncode": returncode,
            "result_path": record.get("result_path"),
            "result_sha256": record.get("result_sha256"),
        },
    )


def _next_attempt(case_root: Path) -> Path:
    numbers = []
    for path in case_root.glob("attempt-*"):
        try:
            numbers.append(int(path.name.removeprefix("attempt-")))
        except ValueError:
            continue
    return case_root / f"attempt-{max(numbers, default=0) + 1:04d}"


def _summary(
    benchmark: str, run_id: str, selected: tuple[str, ...], records: dict[str, Any]
) -> dict[str, Any]:
    ordered = [records[task] for task in selected]
    valid = [
        record
        for record in ordered
        if record.get("returncode") == 0 and record.get("status") in _VALID_TERMINALS
    ]
    scores = [
        float(record["score"])
        for record in valid
        if isinstance(record.get("score"), (int, float))
    ]
    successes = sum(record.get("success") is True for record in valid)
    return {
        "schema_version": 1,
        "benchmark": benchmark,
        "run_id": run_id,
        "planned": len(selected),
        "completed": len(valid),
        "failed": len(selected) - len(valid),
        "successes": successes,
        "task_failures": sum(
            record.get("status") == "task_failure" for record in valid
        ),
        "success_rate": successes / len(selected) if selected else 0.0,
        "mean_score": sum(scores) / len(scores) if scores else None,
        "cases": ordered,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=tuple(_EXPECTED), required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--max-concurrent", type=int, default=16)
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--port-base", type=int, default=18400)
    parser.add_argument("--timeout-s", type=float, default=43200)
    parser.add_argument("--case-timeout-s", type=float, default=14400)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    args = parser.parse_args()
    if args.max_concurrent < 1:
        parser.error("--max-concurrent must be positive")
    if args.gpu_count < 1:
        parser.error("--gpu-count must be positive")
    if not args.base_url:
        parser.error("--base-url or OPENAI_BASE_URL is required")

    repository = Path(__file__).resolve().parents[2]
    declared = _tasks(repository, args.benchmark)
    selected = tuple(args.task) if args.task else declared
    if len(selected) != len(set(selected)) or not set(selected) <= set(declared):
        parser.error("--task values must be unique members of the migration subset")

    root = args.results_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    conditions = _conditions(
        repository,
        args.benchmark,
        base_url=args.base_url,
        timeout_s=args.timeout_s,
        case_timeout_s=args.case_timeout_s,
    )
    condition_sha256 = _json_sha256(conditions)
    plan = {
        "schema_version": 1,
        "benchmark": args.benchmark,
        "run_id": args.run_id,
        "tasks": list(selected),
        "max_concurrent": args.max_concurrent,
        "gpu_count": args.gpu_count,
        "conditions": conditions,
        "condition_sha256": condition_sha256,
    }
    plan_path = root / "subset-plan.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing != plan:
            raise ValueError("results root belongs to a different subset plan")
    else:
        plan_path.write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def run(index_task: tuple[int, str]) -> tuple[str, dict[str, Any]]:
        index, task = index_task
        case_root = root / "cases" / task
        completed = _completed_attempt(
            case_root,
            benchmark=args.benchmark,
            task_id=task,
            condition_sha256=condition_sha256,
        )
        if completed is not None:
            attempt, record = completed
            return task, {
                "task_id": task,
                "returncode": 0,
                "attempt": attempt.name,
                "cached": True,
                **record,
            }

        attempt = _next_attempt(case_root)
        log_path = root / "logs" / f"{task}-{attempt.name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-B",
            str(repository / "benchmarks/libero_three_combined/run_combined.py"),
            "--benchmark",
            args.benchmark,
            "--task",
            task,
            "--run-id",
            f"{args.run_id}-{task}-{attempt.name}",
            "--results-root",
            str(attempt),
            "--port",
            str(args.port_base + index),
            "--timeout-s",
            str(args.timeout_s),
            "--case-timeout-s",
            str(args.case_timeout_s),
            "--api-key-file",
            str(args.api_key_file),
            "--base-url",
            args.base_url,
        ]
        with log_path.open("w", encoding="utf-8") as log:
            completed_process = subprocess.run(
                command,
                cwd=repository,
                env={
                    **os.environ,
                    "MUJOCO_EGL_DEVICE_ID": str(index % args.gpu_count),
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                },
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        record = _record_result(attempt, benchmark=args.benchmark, task_id=task, seed=0)
        _write_completion(
            attempt,
            benchmark=args.benchmark,
            task_id=task,
            condition_sha256=condition_sha256,
            returncode=completed_process.returncode,
            record=record,
        )
        return task, {
            "task_id": task,
            "returncode": completed_process.returncode,
            "attempt": attempt.name,
            "cached": False,
            **record,
        }

    records: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.max_concurrent) as executor:
        futures = {executor.submit(run, item): item[1] for item in enumerate(selected)}
        for future in as_completed(futures):
            task, record = future.result()
            records[task] = record
            print(
                "ROBORSI_SUBSET_CASE="
                + json.dumps(
                    {
                        "benchmark": args.benchmark,
                        "task_id": task,
                        "status": record.get("status"),
                        "success": record.get("success"),
                        "score": record.get("score"),
                        "returncode": record["returncode"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    summary = _summary(args.benchmark, args.run_id, selected, records)
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "ROBORSI_SUBSET_RESULT="
        + json.dumps(
            {key: summary[key] for key in summary if key != "cases"},
            sort_keys=True,
        ),
        flush=True,
    )
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
