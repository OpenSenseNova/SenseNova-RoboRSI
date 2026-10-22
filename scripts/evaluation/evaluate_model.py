#!/usr/bin/env python3
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

"""Evaluate an already deployed vision/tool model with RoboRSI's existing runner.

With no benchmark source, download environments and ICL, then run managed Docker
services. --prepare-only downloads and prepares without evaluation. --dry-run
prints the automatic setup plan, or preflights an explicitly supplied catalog.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
IS_WINDOWS = sys.platform == "win32"
BENCHMARKS = (
    "libero",
    "libero_pro",
    "libero_long_plus_plus",
    "libero_ultra_long",
    "robocasa365",
)
LONG_BENCHMARKS = {"libero_pro", "libero_long_plus_plus", "libero_ultra_long"}
DEFAULT_PRESET = ROOT / "configs/agents/embodied/default.yaml"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def validate_url(value: str, *, origin: bool = False) -> str:
    """Reject URL credentials and query parameters before anything is printed."""
    try:
        parsed = urlsplit(value)
        port = parsed.port
        valid = (
            parsed.scheme in {"http", "https"}
            and parsed.hostname
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
            and not any(char.isspace() or ord(char) < 32 for char in value)
            and "\\" not in value
            and (port is None or port > 0)
            and (not origin or parsed.path in {"", "/"})
        )
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(
            "URLs must be HTTP(S), without credentials, query strings or fragments"
            + ("; Benchmark URLs must be service origins" if origin else "")
        )
    return value.rstrip("/")


def named_values(
    values: list[str] | None, allowed: list[str], option: str
) -> dict[str, str]:
    result = {}
    for value in values or ():
        name, separator, selected = value.partition("=")
        if (
            not separator
            or name not in allowed
            or name in result
            or not selected.strip()
        ):
            raise ValueError(
                f"{option} must be unique selected-benchmark=VALUE entries"
            )
        result[name] = selected
    return result


def parse_args(
    argv: list[str] | None,
) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url", help="override model YAML and its named environment variable"
    )
    parser.add_argument(
        "--api-key",
        help="prefer OPENAI_API_KEY in the environment; use EMPTY for unauthenticated APIs",
    )
    parser.add_argument(
        "--model-name", help="override model YAML and its named environment variable"
    )
    parser.add_argument(
        "--openai-format",
        choices=("chat", "responses"),
        help="override model YAML / Agent API format",
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        choices=BENCHMARKS,
        required=True,
        help="benchmark to evaluate; repeat only when explicitly selecting several",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--registry-root", type=Path)
    source.add_argument("--benchmark-url", action="append", metavar="NAME=URL")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=ROOT / "work/model_eval",
        help="cache for images, assets, ICL, optional VLA and generated presets",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="prepare environments, ICL and optional VLA; no model inference or episodes",
    )
    parser.add_argument(
        "--benchmark-token-env",
        help="environment variable containing the Benchmark token",
    )
    parser.add_argument(
        "--agent-config",
        action="append",
        metavar="[NAME=]PATH",
        help="Agent YAML; automatic mode defaults to the benchmark configuration's Agent",
    )
    parser.add_argument(
        "--benchmark-config",
        action="append",
        metavar="[NAME=]PATH",
        help="benchmark YAML; may override test selection while inheriting the native environment",
    )
    parser.add_argument(
        "--model-config",
        action="append",
        metavar="[NAME=]PATH",
        help="model connection YAML",
    )
    parser.add_argument("--task", action="append", metavar="NAME:ID")
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        help="repeat to select seeds; default: each catalog's declared seeds",
    )
    parser.add_argument("--results-root", type=Path)
    parser.add_argument(
        "--run-id",
        default=f"model-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}",
    )
    parser.add_argument("--max-concurrent", type=positive_int, default=1)
    parser.add_argument("--timeout-s", type=positive_float, default=1800)
    parser.add_argument(
        "--case-timeout-s",
        type=positive_float,
        help="default: 43200 for PRO/Long++/Ultra, 4500 otherwise",
    )
    parser.add_argument(
        "--max-turns", type=positive_int, help="generic preset default: 100"
    )
    parser.add_argument(
        "--max-tokens", type=positive_int, help="generic preset default: 2048"
    )
    parser.add_argument(
        "--keep-turns", type=positive_int, help="generic preset default: 8"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        configuration = load_configuration()
        benchmark_paths = configuration_paths(
            args.benchmark_config, args.benchmark, "--benchmark-config"
        )
        model_paths = configuration_paths(
            args.model_config, args.benchmark, "--model-config"
        )
        agent_paths = configuration_paths(
            args.agent_config, args.benchmark, "--agent-config"
        )
        args.configurations = {
            name: configuration.load_benchmark(
                name,
                **({"path": benchmark_paths[name]} if name in benchmark_paths else {}),
            )
            for name in args.benchmark
        }
        for name, config in args.configurations.items():
            args.configurations[name] = replace(
                config,
                model=model_paths.get(name, config.model),
                agent=agent_paths.get(name, config.agent),
                agent_overrides={} if name in agent_paths else config.agent_overrides,
            )
        args.models = {}
        for name, config in args.configurations.items():
            model = configuration.resolve_model(
                config.model,
                base_url=args.base_url,
                model_name=args.model_name,
                api_key=args.api_key,
                openai_format=args.openai_format,
            )
            if model.base_url:
                model = replace(model, base_url=validate_url(model.base_url))
            if not args.prepare_only and (
                not model.base_url
                or not model.model_name
                or not model.model_name.strip()
            ):
                raise ValueError(
                    "set model YAML, --base-url / OPENAI_BASE_URL and --model-name / MODEL_NAME"
                )
            if (
                not args.dry_run
                and not args.prepare_only
                and (not model.api_key or not model.api_key.strip())
            ):
                raise ValueError(
                    "set the model profile's api_key_env or --api-key (use EMPTY for an unauthenticated model API)"
                )
            args.models[name] = model
        first_model = args.models[args.benchmark[0]]
        args.base_url, args.model_name, args.api_key = (
            first_model.base_url,
            first_model.model_name,
            first_model.api_key,
        )
        args.auto_prepare = not (args.registry_root or args.benchmark_url)
        if args.prepare_only and not args.auto_prepare:
            raise ValueError(
                "--prepare-only cannot be combined with an existing benchmark source"
            )
        if len(args.benchmark) != len(set(args.benchmark)):
            raise ValueError("--benchmark values must be unique")
        args.libero_suite = (
            configuration.catalog_setting(args.configurations["libero"], "suite")
            if args.auto_prepare and "libero" in args.benchmark
            else None
        )
        args.robocasa_cells = (
            configuration.catalog_setting(args.configurations["robocasa365"], "cells")
            if args.auto_prepare and "robocasa365" in args.benchmark
            else None
        )
        if args.auto_prepare and args.max_concurrent != 1:
            raise ValueError(
                "automatic HTTP Benchmark services require --max-concurrent 1"
            )
        if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
            raise ValueError(
                "--run-id must contain only letters, digits, underscores and hyphens"
            )
        if args.seed is not None and (
            len(args.seed) != len(set(args.seed)) or min(args.seed) < 0
        ):
            raise ValueError("--seed values must be unique nonnegative integers")
        args.endpoints = named_values(
            args.benchmark_url, args.benchmark, "--benchmark-url"
        )
        if args.benchmark_url:
            if set(args.endpoints) != set(args.benchmark):
                raise ValueError("provide --benchmark-url for every selected benchmark")
            if args.max_concurrent != 1:
                raise ValueError("HTTP Benchmarks require --max-concurrent 1")
            args.endpoints = {
                name: validate_url(url, origin=True)
                for name, url in args.endpoints.items()
            }
        if args.benchmark_token_env:
            if not args.benchmark_url or not os.environ.get(args.benchmark_token_env):
                raise ValueError(
                    "--benchmark-token-env requires HTTP Benchmarks and a populated environment variable"
                )
        args.presets = {
            name: str(args.configurations[name].agent)
            for name in set(agent_paths) | set(benchmark_paths)
        }
        args.tasks = defaultdict(list)
        for value in args.task or ():
            name, separator, task = value.partition(":")
            if (
                not separator
                or name not in args.benchmark
                or not task.strip()
                or task in args.tasks[name]
            ):
                raise ValueError("--task must be a unique selected-benchmark:task_id")
            args.tasks[name].append(task)
        args.work_dir = args.work_dir.expanduser().resolve()
        args.results_root = (
            (args.results_root or args.work_dir / "results").expanduser().resolve()
        )
        if args.registry_root:
            args.registry_root = args.registry_root.expanduser().resolve()
        if IS_WINDOWS and not (args.dry_run and (args.endpoints or args.auto_prepare)):
            raise ValueError(
                "preparation and evaluation require Linux/WSL; native Windows supports --help and automatic/HTTP --dry-run only"
            )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return parser, args


def configuration_paths(values, benchmarks, option):
    """Accept an unqualified path only when exactly one benchmark is selected."""
    scoped = []
    for value in values or ():
        if "=" not in value and len(benchmarks) == 1:
            value = f"{benchmarks[0]}={value}"
        scoped.append(value)
    paths = {
        name: Path(path).expanduser().resolve()
        for name, path in named_values(scoped, benchmarks, option).items()
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"{option} is missing: {path}")
    return paths


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    # The script works from any cwd; the repository supplies the installed runner.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from roborsi.benchmarks.registry import BenchmarkRegistry
    from roborsi.evaluation.batch import expand_cases
    from roborsi.evaluation.http_benchmark import HttpBenchmarkRegistry
    from scripts.evaluation.selection import select_cells

    registry = (
        HttpBenchmarkRegistry(
            args.endpoints,
            token=os.environ.get(args.benchmark_token_env)
            if args.benchmark_token_env
            else None,
            timeout_s=args.timeout_s,
        )
        if args.endpoints
        else BenchmarkRegistry([args.registry_root])
    )
    cells = []
    for name in args.benchmark:
        config = args.configurations[name]
        tasks = args.tasks.get(name) or config.evaluation.get("tasks")
        seeds = args.seed if args.seed is not None else config.evaluation.get("seeds")
        cells.extend(
            select_cells(
                config,
                expand_cases(
                    registry,
                    [name],
                    task_ids={name: tasks} if tasks else None,
                    seeds=seeds,
                ),
            )
        )
    groups = {}
    budgets = {}
    for benchmark in args.benchmark:
        lock = registry.lock(benchmark)
        for cell in (cell for cell in cells if cell.benchmark == benchmark):
            budget = None if benchmark in args.presets else 3600
            suffix = benchmark
            if benchmark == "robocasa365":
                budget = lock.task(cell.task_id, cell.seed).native.get("timeout_s")
                if isinstance(budget, bool) or budget not in {1800, 3600}:
                    raise ValueError(
                        "RoboCasa365 catalogs must declare Target50 timeout_s of 1800 or 3600 per task"
                    )
                budget = int(budget)
                suffix += "-atomic" if budget == 1800 else "-composite"
            task_key = (benchmark, cell.task_id)
            if task_key in budgets and budgets[task_key] != budget:
                raise ValueError(
                    "a task must have the same planner budget for all selected seeds"
                )
            budgets[task_key] = budget
            seeds = tuple(
                sorted(
                    c.seed
                    for c in cells
                    if c.benchmark == benchmark and c.task_id == cell.task_id
                )
            )
            tasks = groups.setdefault((benchmark, suffix, budget, seeds), [])
            if cell.task_id not in tasks:
                tasks.append(cell.task_id)

    commands = []
    for (benchmark, suffix, budget, seeds), tasks in groups.items():
        if sum(1 for key in groups if key[:3] == (benchmark, suffix, budget)) > 1:
            suffix += (
                "-seeds-" + hashlib.sha256(json.dumps(seeds).encode()).hexdigest()[:8]
            )
        custom = benchmark in args.presets
        model = args.models[benchmark]
        overrides = {
            "runtime.llm.provider": "openai",
            "runtime.llm.base_url": model.base_url,
            "runtime.llm.model": model.model_name,
        }
        for key, explicit, default in (
            ("runtime.llm.openai_format", model.openai_format, "chat"),
            ("agent.max_turns", args.max_turns, 100),
            ("agent.max_tokens", args.max_tokens, 2048),
            ("runtime.context.keep_turns", args.keep_turns, 8),
        ):
            if explicit is not None or not custom:
                overrides[key] = explicit if explicit is not None else default
        if not custom:
            overrides["agent.reasoning_effort"] = "none"
        if budget is not None:
            overrides["agent.planner_timeout_s"] = budget
        case_timeout = (
            args.case_timeout_s or args.configurations[benchmark].case_timeout_s
        )
        command = [
            sys.executable,
            "-m",
            "roborsi.evaluation.cli",
            "eval",
            "--agent-config",
            args.presets.get(benchmark, str(DEFAULT_PRESET)),
            "--benchmark",
            benchmark,
            "--run-id",
            f"{args.run_id}-{suffix}",
            "--results-root",
            str(args.results_root),
            "--max-concurrent",
            str(args.max_concurrent),
            "--timeout-s",
            str(args.timeout_s),
            "--case-timeout-s",
            str(case_timeout),
        ]
        if args.endpoints:
            command += ["--benchmark-url", f"{benchmark}={args.endpoints[benchmark]}"]
            if args.benchmark_token_env:
                command += ["--benchmark-token-env", args.benchmark_token_env]
        else:
            command += ["--registry-root", str(args.registry_root)]
        for key, value in overrides.items():
            command += ["--agent-set", f"{key}={json.dumps(value, ensure_ascii=False)}"]
        for task in tasks:
            command += ["--task", f"{benchmark}:{task}"]
        for seed in seeds:
            command += ["--seed", str(seed)]
        commands.append(command)
    return commands


def load_configuration():
    """Load the YAML resolver when evaluating, keeping --help lightweight."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.evaluation import configuration

    return configuration


def load_bootstrap():
    """Import setup lazily so ordinary help and prepared services stay lightweight."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.evaluation import bootstrap

    return bootstrap


def main(argv: list[str] | None = None) -> int:
    parser, args = parse_args(argv)
    if not args.auto_prepare:
        from scripts.evaluation import vla

        try:
            if args.dry_run:
                vla.plan(args.presets)
                return run_evaluation(parser, args)
            deployments = vla.prepare_agents(args.presets, args.work_dir)
            with vla.services(deployments, args.presets, args.work_dir) as bound:
                args.presets = {**args.presets, **bound}
                return run_evaluation(parser, args)
        except (
            ValueError,
            RuntimeError,
            OSError,
            subprocess.CalledProcessError,
        ) as exc:
            parser.error(str(exc))
        except KeyboardInterrupt:
            return 130
    bootstrap = load_bootstrap()
    if args.dry_run:
        try:
            plan = bootstrap.preparation_plan(
                args.benchmark, args.work_dir, configurations=args.configurations
            )
        except (ValueError, RuntimeError, OSError) as exc:
            parser.error(str(exc))
        print(json.dumps(plan, indent=2))
        return 0
    try:
        prepared = bootstrap.prepare_benchmarks(
            args.benchmark, args.work_dir, configurations=args.configurations
        )
        if args.prepare_only:
            print(
                json.dumps(
                    {
                        "work_dir": str(args.work_dir),
                        "images": prepared.images,
                        "agent_presets": prepared.presets,
                    },
                    indent=2,
                )
            )
            return 0
        # Preparation already selected explicit user presets and bound local assets.
        args.presets = prepared.presets
        args.prepared_env = prepared.env
        from scripts.evaluation import vla

        with (
            bootstrap.benchmark_services(
                prepared,
                libero_suite=args.libero_suite,
                robocasa_cells=args.robocasa_cells,
            ) as endpoints,
            vla.services(
                getattr(prepared, "vla", {}), args.presets, args.work_dir
            ) as bound,
        ):
            args.presets = {**args.presets, **bound}
            args.endpoints = endpoints
            args.benchmark_url = [f"{name}={url}" for name, url in endpoints.items()]
            return run_evaluation(parser, args)
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        print(
            "Interrupted; managed Benchmark services have been stopped.",
            file=sys.stderr,
        )
        return 130


def run_evaluation(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Preflight and execute already available catalogs and selected Agent presets."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from roborsi.utils.rpc import RpcError

    secrets = [
        value
        for value in (
            *(model.api_key for model in args.models.values()),
            os.environ.get(args.benchmark_token_env)
            if args.benchmark_token_env
            else None,
        )
        if value
    ]

    def redact(value: str) -> str:
        for secret in secrets:
            value = value.replace(secret, "[REDACTED]")
        return value

    try:
        commands = build_commands(args)
    except (ValueError, OSError, KeyError, RpcError) as exc:
        parser.error(redact(str(exc)))
    environment = os.environ.copy()
    for variable in {
        "OPENAI_API_KEY",
        *(model.api_key_env for model in args.models.values()),
    }:
        environment.pop(variable, None)
    environment.update(getattr(args, "prepared_env", {}))
    environment["PYTHONUTF8"] = "1"
    for name, model in args.models.items():
        print(
            redact(f"Model ({name}): {model.model_name} | API: {model.base_url}"),
            flush=True,
        )
    for command in commands:
        run_id = command[command.index("--run-id") + 1]
        print(redact(f"Results: {args.results_root / run_id}"), flush=True)

    # Validate every group before starting the first expensive evaluation.
    phases = [True] if args.dry_run else [True, False]
    failed = False
    for dry_run in phases:
        for command in commands:
            run_id = command[command.index("--run-id") + 1]
            print(f"{'Preflight' if dry_run else 'Evaluate'}: {run_id}", flush=True)
            try:
                model = args.models[command[command.index("--benchmark") + 1]]
                child_environment = environment.copy()
                if model.api_key:
                    child_environment["OPENAI_API_KEY"] = model.api_key
                else:
                    child_environment.pop("OPENAI_API_KEY", None)
                completed = subprocess.run(
                    command + (["--dry-run"] if dry_run else []),
                    cwd=ROOT,
                    env=child_environment,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    capture_output=True,
                )
            except OSError as exc:
                print(redact(str(exc)), file=sys.stderr)
                return 1
            if completed.stdout:
                print(
                    redact(completed.stdout),
                    end="" if completed.stdout.endswith("\n") else "\n",
                    flush=True,
                )
            if completed.stderr:
                print(
                    redact(completed.stderr),
                    file=sys.stderr,
                    end="" if completed.stderr.endswith("\n") else "\n",
                )
            if completed.returncode:
                return completed.returncode
            if not dry_run:
                try:
                    report = json.loads(completed.stdout.strip().splitlines()[-1])
                    results = [record["result"] for record in report["cases"]]
                    if not results:
                        raise ValueError("empty evaluation report")
                    has_errors = False
                    for result in results:
                        status = result["status"]
                        # Match benchmarks/robocasa365/score.py: protocol planner
                        # budget exhaustion is a scored task failure in Target50.
                        protocol_timeout = (
                            result["task"]["benchmark"] == "robocasa365"
                            and status == "agent_error"
                            and "planner timed out"
                            in ((result.get("error") or {}).get("message") or "")
                        )
                        if status not in {
                            "success",
                            "task_failure",
                            "agent_error",
                            "infrastructure_error",
                            "cancelled",
                        }:
                            raise ValueError("unrecognized evaluation status")
                        has_errors |= (
                            status
                            in {"agent_error", "infrastructure_error", "cancelled"}
                            and not protocol_timeout
                        )
                except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                    print(
                        "Runner did not return a valid evaluation report.",
                        file=sys.stderr,
                    )
                    return 1
                if has_errors:
                    print(
                        f"Runtime errors recorded in {run_id}; inspect the case results.",
                        file=sys.stderr,
                    )
                    failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
