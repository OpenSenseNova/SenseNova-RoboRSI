# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Offline plans, streaming diagnostics and credential-free invocation artifacts."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path


def redactor(args):
    secrets = {model.api_key for model in args.models.values() if model.api_key}
    secrets.update(
        os.environ.get(variable)
        for variable in args.benchmark_tokens.values()
        if variable and os.environ.get(variable)
    )

    def redact(text):
        for secret in sorted(secrets, key=len, reverse=True):
            text = text.replace(secret, "[REDACTED]")
        return text

    return redact


def plan(args, preparation):
    return {
        **preparation,
        "network_requests": False,
        "run_id": args.run_id,
        "resume_from": args.resume_from,
        "results": str(args.results_root / args.run_id),
        "configurations": {
            name: {
                "benchmark": str(config.path),
                "model": str(config.model),
                "agent": str(config.agent),
                "agent_overrides": config.agent_overrides,
                "environment": config.environment,
            }
            for name, config in args.configurations.items()
        },
        "models": {
            name: {
                "base_url": model.base_url,
                "model_name": model.model_name,
                "openai_format": model.openai_format,
                "api_key_env": model.api_key_env,
                "sources": model.sources,
                "vla": asdict(model.vla) if model.vla else None,
            }
            for name, model in args.models.items()
        },
        "services": {
            name: {
                "endpoint": args.endpoints.get(name),
                "mode": "http"
                if name in args.endpoints
                else "registry"
                if args.registry_root
                else "managed",
                "token_env": args.benchmark_tokens.get(name),
                "agent_resources": "prepare ICL/tokenizer/VLA as required",
            }
            for name in args.benchmark
        },
        "selection": {
            name: {
                **config.evaluation,
                **({"tasks": args.tasks[name]} if args.tasks.get(name) else {}),
                **({"seeds": args.seed} if args.seed is not None else {}),
                "case_count": "resolved from catalog during execution",
            }
            for name, config in args.configurations.items()
        },
    }


def write_json(path, document, redact=lambda value: value):
    def clean(value):
        if isinstance(value, str):
            return redact(value)
        if isinstance(value, dict):
            return {redact(str(key)): clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(clean(document), indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def run_child(command, *, cwd, env, log_path, redact):
    """Stream both pipes immediately; retain the tail containing the final JSON report."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tails = {"stdout": deque(maxlen=50), "stderr": deque(maxlen=50)}
    lock = threading.Lock()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=os.name != "nt",
        )

        def forward(pipe, name, target):
            for line in pipe:
                clean = redact(line)
                with lock:
                    tails[name].append(line)
                    log.write(clean)
                    log.flush()
                    target.write(clean)
                    target.flush()
            pipe.close()

        threads = [
            threading.Thread(
                target=forward, args=(process.stdout, "stdout", sys.stdout)
            ),
            threading.Thread(
                target=forward, args=(process.stderr, "stderr", sys.stderr)
            ),
        ]
        for thread in threads:
            thread.start()
        started = time.monotonic()
        try:
            while True:
                try:
                    code = process.wait(timeout=15)
                    break
                except subprocess.TimeoutExpired:
                    heartbeat = f"Runner active: elapsed={time.monotonic() - started:.0f}s | {log_path}\n"
                    with lock:
                        log.write(heartbeat)
                        log.flush()
                        sys.stderr.write(heartbeat)
                        sys.stderr.flush()
        except BaseException:
            if os.name == "nt":
                process.terminate()
            else:
                os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    process.kill()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        finally:
            for thread in threads:
                thread.join()
    return subprocess.CompletedProcess(
        command, code, "".join(tails["stdout"]), "".join(tails["stderr"])
    )


def save_service_log(container, directory, run):
    """Best-effort diagnostics must not mask the original startup/evaluation error."""
    try:
        result = run(
            ["docker", "logs", "--timestamps", container],
            capture_output=True,
            check=False,
        )
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{container}.log").write_text(
            (result.stdout or "") + (result.stderr or ""), encoding="utf-8"
        )
    except (OSError, AttributeError):
        print(f"Could not save service log: {container}", file=sys.stderr)


def summary(reports, started):
    counts = {}
    cases = []
    errors = 0
    for report in reports:
        for record in report["cases"]:
            result = record["result"]
            status = result["status"]
            if status not in {
                "success",
                "task_failure",
                "agent_error",
                "infrastructure_error",
                "cancelled",
            }:
                raise ValueError("unrecognized evaluation status")
            counts[status] = counts.get(status, 0) + 1
            cases.append(record)
            protocol_timeout = (
                result["task"]["benchmark"] == "robocasa365"
                and status == "agent_error"
                and "planner timed out"
                in ((result.get("error") or {}).get("message") or "")
            )
            errors += (
                status in {"agent_error", "infrastructure_error", "cancelled"}
                and not protocol_timeout
            )
    total = len(cases)
    return {
        "total": total,
        "successes": counts.get("success", 0),
        "success_rate": counts.get("success", 0) / total if total else 0,
        "runtime_errors": errors,
        "statuses": counts,
        "elapsed_s": time.monotonic() - started,
        "groups": [
            {"run_id": item["run_id"], "benchmarks": item.get("benchmarks", {})}
            for item in reports
        ],
    }
