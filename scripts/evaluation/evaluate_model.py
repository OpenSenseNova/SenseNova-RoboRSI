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
import time
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
        help="model YAML: main API connection and optional VLA model",
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
    parser.add_argument(
        "--resume-from", help="previous invocation run ID; requires a new --run-id"
    )
    args = parser.parse_args(argv)
    try:
        args.framework = None
        args.image_observation_tools = False
        args.eval_config = None
        args.pro_protocol = None
        args.tool_urls = {}
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
        pro = args.configurations.get("libero_pro")
        if pro is not None and pro.protocol is not None:
            if pro.environment.get("native_bundle") != "libero_pro_image_tools":
                raise ValueError(
                    "explicit PRO protocol requires environment.native_bundle: libero_pro_image_tools"
                )
            if args.benchmark != ["libero_pro"]:
                raise ValueError(
                    "an explicit PRO protocol must be evaluated separately"
                )
            from robots.libero.pro_protocol import load_pro_config

            config = args.configurations["libero_pro"]
            args.eval_config = config.protocol
            args.pro_protocol = load_pro_config(args.eval_config)
            if args.seed is not None and args.seed != [args.pro_protocol["seed"]]:
                raise ValueError("--seed conflicts with the selected PRO protocol")
            if config.evaluation.get("seeds", [0]) != [0]:
                raise ValueError(
                    "evaluation.seeds conflicts with the selected PRO protocol"
                )
            args.seed = [args.pro_protocol["seed"]]
            args.framework = configuration.pro_tool_mode(config.agent)
            from robots.libero.image_tools import configured

            args.image_observation_tools = configured(config.agent)
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
            if (
                not args.prepare_only
                and not args.dry_run
                and (
                    not model.base_url
                    or not model.model_name
                    or not model.model_name.strip()
                )
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
        if args.framework == "vla_sam3":
            if not args.models["libero_pro"].tools:
                raise ValueError("VLA/SAM3 requires an RLinf/SAM3 --model-config")
            args.tool_urls = args.models["libero_pro"].tools["urls"]
        first_model = args.models[args.benchmark[0]]
        args.base_url, args.model_name, args.api_key = (
            first_model.base_url,
            first_model.model_name,
            first_model.api_key,
        )
        configured_endpoints = {
            name: config.endpoint
            for name, config in args.configurations.items()
            if config.endpoint is not None
        }
        if not args.registry_root:
            configured_endpoints.update(
                named_values(args.benchmark_url, args.benchmark, "--benchmark-url")
            )
            args.benchmark_url = [
                f"{name}={url}" for name, url in configured_endpoints.items()
            ]
        args.auto_prepare = not (args.registry_root or args.benchmark_url)
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
        if args.resume_from and (
            not re.fullmatch(r"[A-Za-z0-9_-]+", args.resume_from)
            or args.resume_from == args.run_id
            or args.prepare_only
        ):
            raise ValueError(
                "--resume-from requires a valid previous run ID and a different --run-id; cannot combine with --prepare-only"
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
                raise ValueError(
                    "provide endpoint in benchmark YAML or --benchmark-url for every selected benchmark"
                )
            if args.max_concurrent != 1:
                raise ValueError("HTTP Benchmarks require --max-concurrent 1")
            args.endpoints = {
                name: validate_url(url, origin=True)
                for name, url in args.endpoints.items()
            }
        args.benchmark_tokens = {
            name: args.benchmark_token_env or args.configurations[name].token_env
            for name in args.endpoints
        }
        if args.benchmark_token_env and not args.endpoints:
            raise ValueError("--benchmark-token-env requires HTTP Benchmarks")
        for variable in args.benchmark_tokens.values():
            if variable and not re.fullmatch(r"[A-Za-z_]\w*", variable):
                raise ValueError(
                    "Benchmark token_env must name an environment variable"
                )
            if (
                variable
                and not args.dry_run
                and not args.prepare_only
                and not os.environ.get(variable)
            ):
                raise ValueError(
                    f"Benchmark token variable {variable} is not populated"
                )
        args.presets = {
            name: str(args.configurations[name].agent)
            for name in set(agent_paths)
            | set(benchmark_paths)
            | {
                name
                for name, config in args.configurations.items()
                if config.endpoint is not None and not args.registry_root
            }
        }
        if args.pro_protocol:
            args.presets["libero_pro"] = str(args.configurations["libero_pro"].agent)
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
        args.run_dir = args.results_root / args.run_id
        if args.registry_root:
            args.registry_root = args.registry_root.expanduser().resolve()
        if IS_WINDOWS and not args.dry_run:
            raise ValueError(
                "preparation and evaluation require Linux/WSL; native Windows supports --help and offline --dry-run only"
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
            token={
                name: os.environ.get(variable) if variable else None
                for name, variable in args.benchmark_tokens.items()
            },
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
        if args.pro_protocol:
            validate_pro_catalog(args, lock)
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
        # RoboCasa365 keeps the planner settings of its Target50 Agent; the
        # other automatic presets use the generic API-compatible defaults.
        generic = not custom and benchmark != "robocasa365"
        for key, explicit, default in (
            ("runtime.llm.openai_format", model.openai_format, "chat"),
            ("agent.max_turns", args.max_turns, 100),
            ("agent.max_tokens", args.max_tokens, 2048),
            ("runtime.context.keep_turns", args.keep_turns, 8),
        ):
            if explicit is not None:
                overrides[key] = explicit
            elif generic or (not custom and key == "runtime.llm.openai_format"):
                overrides[key] = default
        if generic:
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
            if args.benchmark_tokens.get(benchmark):
                command += ["--benchmark-token-env", args.benchmark_tokens[benchmark]]
        else:
            command += ["--registry-root", str(args.registry_root)]
        for key, value in overrides.items():
            command += ["--agent-set", f"{key}={json.dumps(value, ensure_ascii=False)}"]
        for task in tasks:
            command += ["--task", f"{benchmark}:{task}"]
        for seed in seeds:
            command += ["--seed", str(seed)]
        command += ["--progress"]
        commands.append(command)
    return commands


def validate_pro_catalog(args, lock):
    """Reject an existing service with a different initial state, budget or tools."""
    from robots.libero.image_tools import tool_action_schema
    from robots.libero.pro_protocol import PRO_SUITES, pro_cases

    expected = {
        c.task_id: c
        for c in pro_cases(
            {name: [""] * 10 for name in PRO_SUITES}, config=args.pro_protocol
        )
    }
    selected = lock.list_tasks()
    if {(c.task_id, c.seed) for c in selected} != {(name, 0) for name in expected}:
        raise ValueError(
            "Benchmark catalog does not match the selected 80-case PRO protocol"
        )
    for cell in selected:
        task = lock.task(cell.task_id, cell.seed)
        if task.native != expected[cell.task_id].native or task.max_steps != 2000:
            raise ValueError(
                "Benchmark task/budget mismatch; restart with the selected PRO protocol"
            )
        expected_schema = tool_action_schema(args.framework == "vla_sam3")
        if getattr(args, "image_observation_tools", False):
            from robots.libero.image_tools import (
                PROFILE,
                action_schema,
                locked_manifest,
            )

            expected_schema = action_schema(args.framework == "vla_sam3")
            if task.capabilities.get("image_observation_tools") != PROFILE:
                raise ValueError("Benchmark image observation tools are not enabled")
            if (
                task.capabilities.get("image_tool_source", {}).get("sha256")
                != locked_manifest()["sha256"]
            ):
                raise ValueError(
                    "Benchmark image tools differ from the pinned implementation"
                )
        if task.capabilities.get("action_schema") != expected_schema:
            raise ValueError(
                "Benchmark tool mode differs from the selected Agent config"
            )


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
    from scripts.evaluation import preflight, reporting, vla

    redact = reporting.redactor(args)
    owned_run_dir = False
    try:
        if args.dry_run:
            bootstrap = load_bootstrap()
            preparation = bootstrap.preparation_plan(
                args.benchmark, args.work_dir, configurations=args.configurations
            )
            plan = reporting.plan(args, preparation)
            if not args.auto_prepare:
                plan["images"] = dict.fromkeys(
                    args.benchmark, "existing source; no simulator image preparation"
                )
            print(redact(json.dumps(plan, indent=2, default=str)))
            return 0
        if not args.prepare_only:
            if args.run_dir.exists():
                raise FileExistsError(
                    f"Run already exists: {args.run_dir}; choose a new --run-id"
                )
            preflight.check(args)
            args.run_dir.mkdir(parents=True)
            owned_run_dir = True
            print(f"Run: {args.run_id} | Diagnostics: {args.run_dir}", flush=True)
            reporting.write_json(
                args.run_dir / "resolved_config.json",
                {**reporting.plan(args, {}), "network_requests": True},
                redact,
            )
        bootstrap = load_bootstrap()
        tools_deployment = None
        tools_work = args.work_dir / "libero_tools"
        if args.framework == "vla_sam3" and not args.tool_urls:
            from scripts.deploy import libero_tools

            tools_deployment = args.models["libero_pro"].tools["deployment"]
            host = tools_deployment["advertise_host"]
            if not host or any(c in host for c in "<>/:"):
                raise ValueError(
                    "set model tool_runtime.advertise_host before deployment"
                )
            libero_tools.prepare(tools_deployment, tools_work)
        prepared = (
            bootstrap.prepare_benchmarks
            if args.auto_prepare
            else bootstrap.prepare_agents
        )(args.benchmark, args.work_dir, configurations=args.configurations)
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
        prepared.log_dir = args.run_dir / "logs/services"

        from contextlib import nullcontext

        with (
            (
                libero_tools.services(
                    tools_deployment,
                    tools_work,
                    log_dir=prepared.log_dir / "libero_tools",
                )
                if tools_deployment
                else nullcontext(args.tool_urls)
            ) as tool_urls,
            (
                bootstrap.benchmark_services(
                    prepared,
                    libero_suite=args.libero_suite,
                    robocasa_cells=args.robocasa_cells,
                    pro_eval_config=args.eval_config,
                    framework=args.framework,
                    image_observation_tools=args.image_observation_tools,
                    service_tool_urls=tool_urls,
                )
                if args.auto_prepare
                else nullcontext(args.endpoints)
            ) as endpoints,
            vla.services(
                getattr(prepared, "vla", {}),
                args.presets,
                args.work_dir,
                log_dir=prepared.log_dir,
            ) as bound,
        ):
            args.presets = {**args.presets, **bound}
            args.tool_urls = tool_urls
            args.endpoints = endpoints
            args.benchmark_url = [f"{name}={url}" for name, url in endpoints.items()]
            return run_evaluation(parser, args)
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        if owned_run_dir:
            reporting.write_json(
                args.run_dir / "failure.json", {"error": str(exc)}, redact
            )
        parser.error(redact(str(exc)))
    except KeyboardInterrupt:
        print(
            "Interrupted; managed Benchmark services have been stopped.",
            file=sys.stderr,
        )
        return 130


def run_child(*args, **kwargs):
    from scripts.evaluation.reporting import run_child as stream

    return stream(*args, **kwargs)


def run_evaluation(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Preflight all groups, stream progress, and persist invocation artifacts."""
    from roborsi.utils.rpc import RpcError
    from scripts.evaluation import reporting

    redact = reporting.redactor(args)
    started = time.monotonic()
    reports = []
    code = 1
    try:
        commands = build_commands(args)

        def values(command, option):
            return [
                command[index + 1]
                for index, value in enumerate(command)
                if value == option
            ]

        groups = [
            {
                "run_id": values(command, "--run-id")[0],
                "benchmark": values(command, "--benchmark")[0],
                "tasks": values(command, "--task"),
                "seeds": values(command, "--seed"),
            }
            for command in commands
        ]
        if args.resume_from:
            previous_path = (
                args.results_root / args.resume_from / "resolved_config.json"
            )
            previous = json.loads(previous_path.read_text(encoding="utf-8"))

            def identity(items):
                return [
                    {key: value for key, value in group.items() if key != "run_id"}
                    for group in items
                ]

            if identity(previous.get("groups", [])) != identity(groups):
                raise ValueError(
                    "Resume selection differs from the previous invocation; use the same tasks/seeds/configs"
                )
            for command, old_group in zip(commands, previous["groups"], strict=True):
                old_id = old_group["run_id"]
                if (args.results_root / old_id / "batch_plan.json").is_file():
                    command += ["--resume-from", old_id]
        document = {
            **reporting.plan(args, {}),
            "network_requests": True,
            "groups": groups,
            "commands": commands,
        }
        from roborsi.agents.config import AgentConfig

        document["agents"] = {}
        for command in commands:
            config = AgentConfig.from_file(
                values(command, "--agent-config")[0],
                overrides=values(command, "--agent-set"),
            )
            document["agents"][values(command, "--run-id")[0]] = {
                "fingerprint": config.fingerprint,
                "configuration": config.document,
            }
        reporting.write_json(args.run_dir / "resolved_config.json", document, redact)
        environment = os.environ.copy()
        for variable in {
            "OPENAI_API_KEY",
            *(model.api_key_env for model in args.models.values()),
            *(variable for variable in args.benchmark_tokens.values() if variable),
        }:
            environment.pop(variable, None)
        environment.update(getattr(args, "prepared_env", {}))
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONUNBUFFERED"] = "1"
        for name, model in args.models.items():
            print(
                redact(f"Model ({name}): {model.model_name} | API: {model.base_url}"),
                flush=True,
            )
        planned = sum(len(group["tasks"]) * len(group["seeds"]) for group in groups)
        print(f"Cases: {planned} | Results and logs: {args.run_dir}", flush=True)
        failed = False
        for dry_run in (True, False):
            for command in commands:
                run_id = values(command, "--run-id")[0]
                benchmark = values(command, "--benchmark")[0]
                print(f"{'Preflight' if dry_run else 'Evaluate'}: {run_id}", flush=True)
                model = args.models[benchmark]
                child_environment = environment.copy()
                if model.api_key:
                    child_environment["OPENAI_API_KEY"] = model.api_key
                variable = args.benchmark_tokens.get(benchmark)
                if variable:
                    child_environment[variable] = os.environ[variable]
                completed = run_child(
                    command + (["--dry-run"] if dry_run else []),
                    cwd=ROOT,
                    env=child_environment,
                    log_path=args.run_dir
                    / "logs"
                    / f"{run_id}-{'preflight' if dry_run else 'eval'}.log",
                    redact=redact,
                )
                if completed.returncode:
                    code = completed.returncode
                    return code
                if not dry_run:
                    try:
                        report = json.loads(completed.stdout.strip().splitlines()[-1])
                        if not report["cases"]:
                            raise ValueError("empty evaluation report")
                        group_summary = reporting.summary([report], started)
                    except (
                        ValueError,
                        KeyError,
                        IndexError,
                        TypeError,
                        AttributeError,
                    ):
                        raise ValueError(
                            "Runner did not return a valid evaluation report."
                        ) from None
                    reports.append(report)
                    if group_summary["runtime_errors"]:
                        print(
                            f"Runtime errors recorded in {run_id}; inspect the case results.",
                            file=sys.stderr,
                        )
                        failed = True
        code = int(failed)
        return code
    except (ValueError, OSError, KeyError, RpcError) as exc:
        print(redact(str(exc)), file=sys.stderr)
        code = 1
        return code
    except KeyboardInterrupt:
        code = 130
        raise
    finally:
        overview = {
            **reporting.summary(reports, started),
            "exit_code": code,
            "complete": code in {0, 1}
            and bool(reports)
            and len(reports) == len(locals().get("commands", [])),
            "results": str(args.run_dir),
        }
        reporting.write_json(args.run_dir / "summary.json", overview, redact)
        print(
            f"Overview: {overview['successes']}/{overview['total']} successful "
            f"({overview['success_rate']:.1%}), runtime errors={overview['runtime_errors']}, "
            f"elapsed={overview['elapsed_s']:.1f}s | {args.run_dir / 'summary.json'}",
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
