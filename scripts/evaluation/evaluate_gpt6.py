#!/usr/bin/env python3
# Copyright 2026 The SenseNova-RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Run GPT-6 Astra across the released benchmarks using their existing Agents.

Configure prepared environments in configs/evaluations/gpt6.yaml once, then run
this script without benchmark arguments. --dry-run is entirely offline.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
IS_WINDOWS = sys.platform == "win32"
DEFAULT_CONFIG = ROOT / "configs/evaluations/gpt6.yaml"
MODEL = "gpt-6-astra"
COHORTS = (
    "libero",
    "libero_pro",
    "libero_long_plus_plus",
    "libero_ultra_long",
    "robocasa365",
    "robolab",
    "bench2dex",
    "dexjoco",
    "behavior_1k",
    "lhpr_vln",
    "vlnce_r2r",
    "vlnce_rxr",
)
MANAGED = set(COHORTS[:5])
CLI_VERSIONS = {
    "robolab": "0.160.0",
    "dexjoco": "0.154.0",
    "behavior_1k": "0.162.0",
    "lhpr_vln": "0.161.0",
    "vlnce_r2r": "0.161.0",
    "vlnce_rxr": "0.161.0",
}


@dataclass
class Job:
    name: str
    benchmark: str
    runner: str
    command: list[str]
    env: dict[str, str]
    settings: dict
    report: Path | None


def path_value(base: Path, value: str) -> Path:
    """Resolve a configuration path against its declaring YAML."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("configuration paths must be nonempty strings")
    return (base / Path(value).expanduser()).resolve()


def executable(base: Path, value: str | None) -> str:
    """Keep PATH executable names while resolving explicit file paths."""
    if value is None:
        return sys.executable
    if not isinstance(value, str) or not value.strip():
        raise ValueError("python must be an executable name or path")
    return str(path_value(base, value)) if "/" in value or "\\" in value else value


def load_config(path: Path) -> dict:
    """Load the suite without importing simulators or contacting services."""
    import yaml

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        not isinstance(document, dict)
        or set(document) != {"schema_version", "model_config", "work_dir", "benchmarks"}
        or type(document["schema_version"]) is not int
        or document["schema_version"] != 1
        or not isinstance(document["benchmarks"], dict)
        or set(document["benchmarks"]) != set(COHORTS)
        or any(not isinstance(value, dict) for value in document["benchmarks"].values())
    ):
        raise ValueError("suite configuration must declare all 12 released cohorts")
    return document


def build_jobs(document: dict, config: Path, args) -> tuple[Path, list[Job]]:
    """Compose existing entry points, retaining per-benchmark policy and budgets."""
    from scripts.evaluation.evaluate_model import validate_url

    base = config.parent
    work = (
        args.work_dir.resolve()
        if args.work_dir
        else path_value(base, document["work_dir"])
    )
    results = work / "native-results"
    model_config = path_value(base, document["model_config"])
    jobs = []
    for name in args.benchmark or COHORTS:
        settings = dict(document["benchmarks"][name])
        benchmark = "vlnce" if name.startswith("vlnce_") else name
        runner = (
            "model"
            if name in MANAGED
            else "bench2dex"
            if name == "bench2dex"
            else "native"
        )
        allowed = {"python", "endpoint", "token_env"}
        if runner == "bench2dex":
            allowed |= {"codex", "icl_source", "plan"}
        else:
            allowed |= {
                "agent_config",
                "registry_root",
                "tasks",
                "seeds",
                "timeout_s",
                "case_timeout_s",
            }
            allowed |= {"benchmark_config"} if runner == "model" else {"codex"}
        if set(settings) - allowed:
            raise ValueError(
                f"{name}: unsupported configuration fields {sorted(set(settings) - allowed)}"
            )
        python = executable(base, settings.get("python"))
        env = dict(os.environ)
        endpoint = settings.get("endpoint")
        if endpoint is not None:
            endpoint = validate_url(endpoint, origin=True)
        if endpoint and settings.get("registry_root"):
            raise ValueError(
                f"{name}: endpoint and registry_root are mutually exclusive"
            )
        token = settings.get("token_env")
        if token is not None and (
            not isinstance(token, str) or not re.fullmatch(r"[A-Za-z_]\w*", token)
        ):
            raise ValueError(f"{name}: token_env must name an environment variable")
        codex = settings.get("codex")
        if codex is not None:
            codex = str(path_value(base, codex))
            settings["codex"] = codex
            env["PATH"] = str(Path(codex).parent) + os.pathsep + env.get("PATH", "")
            if name == "bench2dex":
                env["BENCH2DEX_CODEX_BIN"] = codex
            elif name == "dexjoco":
                env["ROBORSI_DEXJOCO_CODEX"] = codex
        run_id = f"{args.run_id}-{name}"
        if runner == "model":
            command = [
                python,
                str(ROOT / "scripts/evaluation/evaluate_model.py"),
                "--benchmark",
                benchmark,
                "--model-config",
                str(model_config),
                "--model-name",
                MODEL,
                "--work-dir",
                str(work / "managed"),
                "--results-root",
                str(results),
                "--run-id",
                run_id,
            ]
            if settings.get("benchmark_config"):
                command += [
                    "--benchmark-config",
                    str(path_value(base, settings["benchmark_config"])),
                ]
            if endpoint:
                command += ["--benchmark-url", f"{benchmark}={endpoint}"]
            elif settings.get("registry_root"):
                command += [
                    "--registry-root",
                    str(path_value(base, settings["registry_root"])),
                ]
            report = results / run_id / "summary.json"
        elif runner == "bench2dex":
            if (
                not endpoint
                or not settings.get("icl_source")
                or not settings.get("plan")
            ):
                raise ValueError("bench2dex requires endpoint, icl_source and plan")
            settings["icl_source"] = str(path_value(base, settings["icl_source"]))
            settings["plan"] = str(path_value(base, settings["plan"]))
            output = work / "results" / args.run_id / "bench2dex"
            command = [
                python,
                "-m",
                "benchmarks.bench2dex.run_suite",
                "--benchmark-url",
                endpoint,
                "--work",
                str(output),
                "--plan",
                settings["plan"],
                "--icl-source",
                settings["icl_source"],
            ]
            report = output / "evaluation-summary.json"
        else:
            if bool(endpoint) == bool(settings.get("registry_root")):
                raise ValueError(
                    f"{name}: select exactly one endpoint or registry_root"
                )
            command = [
                python,
                "-m",
                "roborsi.evaluation.cli",
                "eval",
                "--benchmark",
                benchmark,
                "--run-id",
                run_id,
                "--results-root",
                str(results),
                "--progress",
            ]
            command += (
                ["--benchmark-url", f"{benchmark}={endpoint}"]
                if endpoint
                else [
                    "--registry-root",
                    str(path_value(base, settings["registry_root"])),
                ]
            )
            report = None
        if runner != "bench2dex":
            agent = settings.get("agent_config")
            if not isinstance(agent, str) or not agent.strip():
                raise ValueError(
                    f"{name}: declare the existing task Agent with agent_config"
                )
            command += [
                "--agent-config",
                agent if agent.startswith("bundle:") else str(path_value(base, agent)),
            ]
            # Explicit API presets skip evaluate_model's generic API overrides.
            # CLI presets retain their own model/provider, tool and session rules.
            if codex and name not in {"dexjoco", "behavior_1k"}:
                command += [
                    "--agent-set",
                    f"agent.codex_cli.executable={json.dumps(codex)}",
                ]
            for field, default in (("timeout_s", 900), ("case_timeout_s", 7200)):
                value = settings.get(field, default)
                if type(value) not in {int, float} or not 0 < value < float("inf"):
                    raise ValueError(f"{name}: {field} must be finite and positive")
                command += ["--" + field.replace("_", "-"), str(value)]
            command += ["--max-concurrent", "1"]
            for field, option in (("tasks", "--task"), ("seeds", "--seed")):
                values = settings.get(field)
                if values is not None and (not isinstance(values, list) or not values):
                    raise ValueError(f"{name}: {field} must be a nonempty unique list")
                for value in values or ():
                    if field == "tasks" and (
                        not isinstance(value, str) or not value.strip()
                    ):
                        raise ValueError(f"{name}: task IDs must be nonempty strings")
                    if field == "seeds" and type(value) is not int:
                        raise ValueError(f"{name}: seeds must be integers")
                    command += [
                        option,
                        f"{benchmark}:{value}" if field == "tasks" else str(value),
                    ]
                if values and len(set(values)) != len(values):
                    raise ValueError(f"{name}: {field} must not contain duplicates")
        if token:
            if not endpoint:
                raise ValueError(f"{name}: token_env requires an HTTP endpoint")
            command += ["--benchmark-token-env", token]
        settings["endpoint"] = endpoint
        jobs.append(Job(name, benchmark, runner, command, env, settings, report))
    return work, jobs


def check_inputs(job: Job) -> None:
    """Check local prerequisites before admitting any model evaluations."""
    if job.name not in {"bench2dex", "dexjoco"} and not job.env.get("OPENAI_API_KEY"):
        raise ValueError(
            f"{job.name}: set OPENAI_API_KEY in the evaluation environment"
        )
    token = job.settings.get("token_env")
    if token and not job.env.get(token):
        raise ValueError(
            f"{job.name}: missing Benchmark credential environment variable {token}"
        )
    if job.name == "dexjoco":
        for variable, directory in (
            ("ROBORSI_DEXJOCO_AGENT_ASSETS", True),
            ("ROBORSI_DEXJOCO_POLICY_PYTHON", True),
            ("ROBORSI_DEXJOCO_AUTH", False),
        ):
            value = job.env.get(variable)
            if not value or not (
                Path(value).is_dir() if directory else Path(value).is_file()
            ):
                raise ValueError(
                    f"dexjoco: prepare {variable} using benchmarks/dexjoco/ICL.md"
                )
    if job.runner == "bench2dex":
        from roborsi.evaluation.http_benchmark import HttpBenchmarkRegistry

        auth = Path(
            job.env.get("BENCH2DEX_CODEX_AUTH", str(Path.home() / ".codex/auth.json"))
        )
        if not auth.is_file():
            raise ValueError("bench2dex: prepare BENCH2DEX_CODEX_AUTH")
        plan = json.loads(Path(job.settings["plan"]).read_text(encoding="utf-8"))
        catalog = json.loads(
            (ROOT / "robots/bench2dex/icl_catalog.json").read_text(encoding="utf-8")
        )
        packages = {str(row["task_id"]): row for row in catalog["packages"]}
        if (
            set(plan)
            != {"schema_version", "condition", "tasks", "samples", "expected_cases"}
            or plan["schema_version"] != 1
            or plan["condition"] != "standard_1x"
            or not isinstance(plan["tasks"], list)
            or not plan["tasks"]
            or any(
                not isinstance(task, str) or task not in packages
                for task in plan["tasks"]
            )
            or len(set(plan["tasks"])) != len(plan["tasks"])
            or not isinstance(plan["samples"], list)
            or not plan["samples"]
            or any(type(seed) is not int or seed < 0 for seed in plan["samples"])
            or len(set(plan["samples"])) != len(plan["samples"])
            or plan["expected_cases"] != len(plan["tasks"]) * len(plan["samples"])
        ):
            raise ValueError("bench2dex: invalid fixed evaluation plan")
        for task in plan["tasks"]:
            archive = Path(job.settings["icl_source"]) / packages[task]["archive"]
            if not archive.is_file():
                raise ValueError(f"bench2dex: missing frozen task archive {archive}")
        registry = HttpBenchmarkRegistry(
            {job.benchmark: job.settings["endpoint"]},
            token=job.env.get(token) if token else None,
        )
        lock = registry.lock(job.benchmark)
        for task in plan["tasks"]:
            for sample in plan["samples"]:
                message = lock.task_message(task, sample)
                if message.native.get("smoke_only") is not False:
                    raise ValueError(
                        "bench2dex: prepare the scored standard_1x catalog, not a smoke catalog"
                    )
    if job.name in CLI_VERSIONS:
        codex = (
            job.settings.get("codex")
            or (job.env.get("ROBORSI_DEXJOCO_CODEX") if job.name == "dexjoco" else None)
            or shutil.which("codex", path=job.env.get("PATH"))
        )
        if not codex:
            raise ValueError(f"{job.name}: configure its pinned codex executable")
        version = subprocess.run(
            [codex, "--version"],
            env=job.env,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
        expected = "codex-cli " + CLI_VERSIONS[job.name]
        if version != expected:
            raise ValueError(
                f"{job.name}: requires {expected}; configure this cohort's codex path"
            )


def preflight_command(job: Job) -> list[str]:
    """Use existing non-model preflights before starting the suite."""
    if job.runner == "bench2dex":
        return [job.command[0], "-m", "benchmarks.bench2dex.check_runtime"]
    return [*job.command, "--dry-run"]


def validate_preflight(job: Job, completed) -> None:
    """Reject swapped navigation catalogs before any evaluations start."""
    if job.name in {"vlnce_r2r", "vlnce_rxr"}:
        prefix = job.name.removeprefix("vlnce_") + "-"
        cases = json.loads(completed.stdout.strip().splitlines()[-1]).get("cases")
        if not cases or any(
            case.get("benchmark") != "vlnce"
            or not case.get("task_id", "").startswith(prefix)
            for case in cases
        ):
            raise ValueError(
                f"{job.name}: select its own {prefix[:-1].upper()} registry/catalog"
            )


def read_report(job: Job, completed) -> dict:
    """Retain native summaries and treat runtime errors as failed execution."""
    if job.report is not None:
        report = json.loads(job.report.read_text(encoding="utf-8"))
        if not report.get("complete") or report.get("runtime_errors", 0):
            raise ValueError(
                "evaluation is incomplete or contains runtime errors; inspect its native report"
            )
        if job.runner == "model" and not report.get("total", 0):
            raise ValueError("evaluation returned no cases")
        if job.runner == "bench2dex" and (
            not report.get("expected") or report.get("valid") != report["expected"]
        ):
            raise ValueError("Bench2Dex did not score every case in its fixed plan")
        return report
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    if not report.get("cases") or job.benchmark not in report.get("benchmarks", {}):
        raise ValueError("Runner did not return a nonempty native benchmark report")
    errors = sum(
        record["result"]["status"] not in {"success", "task_failure"}
        for record in report["cases"]
    )
    if errors:
        raise ValueError(
            f"{errors} cases have Agent/infrastructure errors; inspect the native results"
        )
    return {"run_id": report["run_id"], "benchmarks": report["benchmarks"]}


def make_redactor(jobs: list[Job]):
    """Redact inherited API and service credentials from diagnostics."""
    variables = {
        "OPENAI_API_KEY",
        "HF_TOKEN",
        *(job.settings.get("token_env") for job in jobs),
    }
    secrets = {
        os.environ[name]
        for name in variables
        if name and os.environ.get(name) and os.environ[name] != "EMPTY"
    }

    def redact(value: str) -> str:
        for secret in sorted(secrets, key=len, reverse=True):
            value = value.replace(secret, "[REDACTED]")
        return value

    return redact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--benchmark",
        action="append",
        choices=COHORTS,
        help="default: all released cohorts; repeat to select several",
    )
    parser.add_argument(
        "--work-dir", type=Path, help="override the suite cache/results directory"
    )
    parser.add_argument(
        "--run-id",
        default=f"gpt6-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print an offline plan; no downloads, subprocesses, files or model calls",
    )
    args = parser.parse_args(argv)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.evaluation import reporting

    jobs = []
    owned = False
    summary = None
    started = time.monotonic()
    try:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", args.run_id):
            raise ValueError(
                "run-id must use letters, digits, '_' or '-' and be at most 101 characters"
            )
        if args.benchmark and len(set(args.benchmark)) != len(args.benchmark):
            raise ValueError("--benchmark must not repeat the same cohort")
        work, jobs = build_jobs(
            load_config(args.config.resolve()), args.config.resolve(), args
        )
        redact = make_redactor(jobs)
        plan = {
            "model": MODEL,
            "run_id": args.run_id,
            "network_requests": False,
            "results": str(work / "results" / args.run_id),
            "jobs": [
                {"name": job.name, "benchmark": job.benchmark, "command": job.command}
                for job in jobs
            ],
        }
        if args.dry_run:
            print(redact(json.dumps(plan, indent=2)))
            return 0
        if IS_WINDOWS:
            raise ValueError(
                "evaluation requires Linux/WSL; native Windows supports --dry-run only"
            )
        directory = work / "results" / args.run_id
        directory.mkdir(parents=True, exist_ok=False)
        owned = True
        summary = {
            "model": MODEL,
            "run_id": args.run_id,
            "complete": False,
            "jobs": [{"name": job.name, "status": "not_run"} for job in jobs],
        }
        reporting.write_json(directory / "plan.json", plan, redact)
        failures = []
        for job, record in zip(jobs, summary["jobs"], strict=True):
            try:
                check_inputs(job)
                completed = reporting.run_child(
                    preflight_command(job),
                    cwd=ROOT,
                    env=job.env,
                    log_path=directory / "logs" / f"{job.name}-preflight.log",
                    redact=redact,
                )
                if completed.returncode:
                    raise ValueError(
                        "preflight failed; inspect this cohort's preflight log"
                    )
                validate_preflight(job, completed)
            except (
                OSError,
                ValueError,
                RuntimeError,
                TypeError,
                KeyError,
                IndexError,
                subprocess.SubprocessError,
            ) as exc:
                record.update(status="preflight_failed", error=redact(str(exc)))
                failures.append(job.name)
        if failures:
            raise ValueError(
                "no evaluations started; fix prerequisites for: " + ", ".join(failures)
            )
        for job, record in zip(jobs, summary["jobs"], strict=True):
            print(f"GPT-6: {job.name}", flush=True)
            record["status"] = "running"
            reporting.write_json(directory / "summary.json", summary, redact)
            try:
                completed = reporting.run_child(
                    job.command,
                    cwd=ROOT,
                    env=job.env,
                    log_path=directory / "logs" / f"{job.name}.log",
                    redact=redact,
                )
                record["exit_code"] = completed.returncode
                if completed.returncode:
                    raise ValueError(
                        f"evaluation command exited with {completed.returncode}"
                    )
                record.update(status="completed", report=read_report(job, completed))
            except (
                OSError,
                ValueError,
                RuntimeError,
                TypeError,
                KeyError,
                IndexError,
                subprocess.SubprocessError,
            ) as exc:
                record.update(status="failed", error=redact(str(exc)))
            reporting.write_json(directory / "summary.json", summary, redact)
        summary["complete"] = all(
            record["status"] == "completed" for record in summary["jobs"]
        )
        return 0 if summary["complete"] else 1
    except (OSError, ValueError, TypeError, KeyError) as exc:
        redact = make_redactor(jobs)
        print(redact(str(exc)), file=sys.stderr)
        if summary is not None:
            summary["error"] = redact(str(exc))
        return 2
    except KeyboardInterrupt:
        if summary is not None:
            for record in summary["jobs"]:
                if record["status"] == "running":
                    record["status"] = "interrupted"
        print(
            "Interrupted; inspect the suite summary before selecting remaining cohorts.",
            file=sys.stderr,
        )
        return 130
    finally:
        if owned:
            summary["elapsed_s"] = time.monotonic() - started
            reporting.write_json(
                directory / "summary.json", summary, make_redactor(jobs)
            )
            print(f"Suite summary: {directory / 'summary.json'}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
