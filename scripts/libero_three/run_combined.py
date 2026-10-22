# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Run one real-model three-LIBERO case in a combined deployment image."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_BUNDLES = {
    "libero_long_plus_plus": "bundle:libero_long_plus_plus_v13",
    "libero_ultra_long": "bundle:libero_ultra_long_v13",
    "libero_pro": "bundle:libero_pro",
}
_DEFAULT_TASKS = {
    "official": {
        "libero_long_plus_plus": "t00-r00",
        "libero_ultra_long": "t00-r00",
        "libero_pro": "libero_spatial_task-t00-i00",
    },
    "migration_subset": {
        "libero_long_plus_plus": "t00-i01",
        "libero_ultra_long": "t00-i01",
        "libero_pro": "libero_spatial_task-t00-i03",
    },
}
_VALID_TERMINALS = {"success", "task_failure"}


def _ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def _required_secret(path: Path) -> str:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValueError(f"model API key file is unavailable: {exc}") from exc
    if not value:
        raise ValueError("model API key file is empty")
    return value


def _result(root: Path) -> dict:
    matches = list(root.rglob("result.json"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one result.json, found {len(matches)}")
    return json.loads(matches[0].read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=tuple(_BUNDLES), required=True)
    parser.add_argument(
        "--profile",
        choices=("official", "migration_subset"),
        default="official",
    )
    parser.add_argument("--task")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-root", type=Path, default=Path("/tmp/results"))
    parser.add_argument("--port", type=int, default=19080)
    parser.add_argument("--timeout-s", type=float, default=43200)
    parser.add_argument("--case-timeout-s", type=float, default=14400)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("ROBORSI_MODEL"))
    args = parser.parse_args()
    if not args.base_url:
        parser.error("--base-url or OPENAI_BASE_URL is required")
    if not args.model:
        parser.error("--model or ROBORSI_MODEL is required")

    bundle = _BUNDLES[args.benchmark]
    task = args.task or _DEFAULT_TASKS[args.profile][args.benchmark]
    root = args.results_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    service_log = (root / "benchmark-service.log").open("w", encoding="utf-8")
    service_env = {
        **os.environ,
        "ROBORSI_LIBERO_BENCHMARK": args.benchmark,
        "ROBORSI_LIBERO_PROFILE": args.profile,
        "ROBORSI_LIBERO_PLUS9": "0",
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "LIBERO_CONFIG_PATH": f"/tmp/libero-config-{args.benchmark}-{args.port}",
    }
    service_env.pop("OPENAI_API_KEY", None)
    service_env.pop("OPENAI_BASE_URL", None)
    service = subprocess.Popen(
        [
            sys.executable,
            "-B",
            "-m",
            "robots.libero.three_benchmark_serve",
            "--benchmark",
            args.benchmark,
            "--profile",
            args.profile,
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
        ],
        cwd="/opt/roborsi",
        env=service_env,
        stdout=service_log,
        stderr=subprocess.STDOUT,
    )
    try:
        health = f"http://127.0.0.1:{args.port}/v1/health"
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline and service.poll() is None:
            if _ready(health):
                break
            time.sleep(1)
        if not _ready(health):
            raise RuntimeError("Benchmark service did not become healthy")

        agent_env = {
            **os.environ,
            "OPENAI_API_KEY": _required_secret(args.api_key_file),
            "OPENAI_BASE_URL": args.base_url,
        }
        command = [
            sys.executable,
            "-B",
            "-m",
            "roborsi.evaluation.cli",
            "eval",
            "--benchmark-url",
            f"{args.benchmark}=http://127.0.0.1:{args.port}",
            "--results-root",
            str(root / "evaluation"),
            "--run-id",
            args.run_id,
            "--agent-config",
            bundle,
            "--agent-set",
            f"runtime.llm.model={json.dumps(args.model)}",
            "--benchmark",
            args.benchmark,
            "--task",
            f"{args.benchmark}:{task}",
            "--seed",
            "0",
            "--max-concurrent",
            "1",
            "--timeout-s",
            str(args.timeout_s),
            "--case-timeout-s",
            str(args.case_timeout_s),
        ]
        completed = subprocess.run(
            command,
            cwd="/opt/roborsi",
            env=agent_env,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(f"evaluation CLI exited {completed.returncode}")
        result = _result(root / "evaluation")
        if result.get("status") not in _VALID_TERMINALS:
            error = result.get("error") or {}
            raise RuntimeError(
                "case ended without a native evaluation result: "
                f"{result.get('status')}: {error.get('message', error)}"
            )
        summary = {
            "benchmark": args.benchmark,
            "profile": args.profile,
            "task_id": task,
            "status": result.get("status"),
            "success": result.get("success"),
            "score": result.get("score"),
        }
        (root / "smoke-summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print("ROBORSI_SMOKE_RESULT=" + json.dumps(summary, sort_keys=True), flush=True)
        return 0
    finally:
        service.terminate()
        try:
            service.wait(timeout=20)
        except subprocess.TimeoutExpired:
            service.kill()
            service.wait()
        service_log.close()


if __name__ == "__main__":
    raise SystemExit(main())
