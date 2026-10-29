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

"""One entry point for preparing, exploring and evaluating navigation campaigns."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import shutil
import signal
import sys
import threading
from pathlib import Path

import yaml

from roborsi.agents.config import AgentConfig
from roborsi.evaluation.benchmark_client import BenchmarkClient
from roborsi.evaluation.media import CaseMediaChannel
from roborsi.evaluation.messages import Action
from roborsi.planner.codex_cli import preflight_codex_cli
from roborsi.planner.codex_cli_config import CodexCliConfig
from robots.vln_shared.benchmark import NavigationBenchmark, serve
from robots.vln_shared.campaign import Campaign, campaign_source_identity
from robots.vln_shared.execution import BenchExecutor, run_process
from robots.vln_shared.prepare import DATASETS, prepare
from robots.vln_shared.progress import Progress
from robots.vln_shared.storage import (
    digest,
    freeze_snapshot,
    inventory,
    read_json,
    verify_snapshot,
    write_json,
)

AGENT_ROOT = Path(__file__).resolve().parent / "agents/baseline"


def agent_preset(dataset: str | None = None) -> Path:
    """Select the shared baseline or a dataset's standard Benchmark entry."""
    if dataset is None:
        return AGENT_ROOT / "agent.yaml"
    benchmarks = {"r2r": "vlnce", "rxr": "vlnce", "lhpr_vln": "lhpr_vln"}
    if dataset not in benchmarks:
        raise ValueError(f"unsupported navigation dataset: {dataset}")
    return Path(__file__).resolve().parents[1] / benchmarks[dataset] / "agent.yaml"


def make_preset(
    directory: Path,
    cli: dict,
    snapshot: Path | None = None,
    *,
    dataset: str | None = None,
) -> Path:
    """Create a self-contained, exportable Agent input without modifying templates."""
    config = AgentConfig.from_file(agent_preset(dataset))
    document = config.document
    directory.mkdir(parents=True, exist_ok=False)
    module = "vln_agent_" + directory.name.replace("-", "_")
    factory = config.source_root / (
        config.factory.split(":")[0].replace(".", "/") + ".py"
    )
    shutil.copy2(factory, directory / f"{module}.py")
    shutil.copy2(config.path.parent / document["prompt_file"], directory / "prompt.md")
    document["resources"] = {}
    for name, source in config.resources.items():
        relative = Path("resources") / name
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
        document["resources"][name] = relative.as_posix()
    document.update(
        source_root=".",
        factory=f"{module}:make_agent",
        code=[f"{module}.py"],
        prompt_file="prompt.md",
    )
    document["agent"]["codex_cli"].update(cli)
    if snapshot:
        verify_snapshot(snapshot)
        shutil.copytree(snapshot, directory / "exploration")
        document["resources"]["exploration"] = "exploration"
    path = directory / "agent.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    AgentConfig.from_file(path)
    return path


def validate_config(config: dict) -> None:
    """Verify inputs and split isolation without starting a model or simulator."""
    if config.get("schema_version") != 1 or not config.get("datasets"):
        raise ValueError("unsupported or empty navigation config")
    for name, dataset in config["datasets"].items():
        if name not in {"lhpr_vln", "r2r", "rxr"}:
            raise ValueError("unsupported navigation dataset")
        train, test = dataset["train"], dataset["test"]
        if {e["scene"] for e in train["episodes"]} & {
            e["scene"] for e in test["episodes"]
        }:
            raise ValueError("training and evaluation scenes overlap")
        for section in (train, test):
            if section["spec"]["dataset"] != name:
                raise ValueError("dataset identity differs from native configuration")
            provider = NavigationBenchmark(section["spec"], section["episodes"], 500)
            provider.runtime_info()


def select_config(
    config: dict, datasets: list[str] | None, episodes_per_dataset: int | None
) -> dict:
    """Select test prefixes without changing training, original indices or inputs."""
    available = config["datasets"]
    names = list(available) if datasets is None else datasets
    if not names or len(set(names)) != len(names) or set(names) - set(DATASETS):
        raise ValueError("select unique supported navigation datasets")
    if missing := set(names) - set(available):
        raise ValueError(
            f"datasets absent from configuration: {', '.join(sorted(missing))}"
        )
    if episodes_per_dataset is not None and episodes_per_dataset < 1:
        raise ValueError("--episodes-per-dataset must be positive")
    selected = {}
    for name, dataset in available.items():
        if name not in names:
            continue
        episodes = dataset["test"]["episodes"]
        if not episodes or (
            episodes_per_dataset is not None and episodes_per_dataset > len(episodes)
        ):
            raise ValueError(
                f"{name} has {len(episodes)} episodes; requested {episodes_per_dataset or 'all'}"
            )
        selected[name] = {
            **dataset,
            "test": {**dataset["test"], "episodes": episodes[:episodes_per_dataset]},
        }
    return {**config, "datasets": selected}


def run(args: argparse.Namespace) -> dict:
    """Report live progress on stderr while retaining the final JSON on stdout."""
    with Progress(args.progress_interval) as progress:
        progress.phase(f"event=run_start output={args.output.resolve()}")
        try:
            report = _run(args, progress)
        except BaseException:
            progress.phase("event=run_end status=interrupted_or_error")
            raise
        if not args.dry_run:
            for name, group in report["datasets"].items():
                progress.phase(
                    f"event=dataset_end dataset={name} scored={group['scored']}/{group['requested']} "
                    f"successful={group['successful']} incomplete={group['incomplete']}"
                )
        status = (
            "dry_run"
            if args.dry_run
            else "complete"
            if report["complete"]
            else "incomplete"
        )
        progress.phase(f"event=run_end status={status} output={args.output.resolve()}")
        return report


def _run(args: argparse.Namespace, progress: Progress) -> dict:
    """Own the entire preparation/exploration/evaluation transaction under one lock."""
    if (
        args.concurrency < 1
        or args.retries < 0
        or any(
            not math.isfinite(value) or value <= 0
            for value in (args.case_timeout, args.exploration_timeout)
        )
    ):
        raise ValueError(
            "concurrency/timeouts must be positive and finite; retries must be nonnegative"
        )
    output = args.output.resolve()
    roots = [args.source_project, args.history]
    if args.config:
        provenance = read_json(args.config).get("provenance", {})
        roots += [
            Path(provenance[k])
            for k in ("source_project", "history")
            if provenance.get(k)
        ]
    if any(root and output.is_relative_to(root.resolve()) for root in roots):
        raise ValueError(
            "output must be outside the preserved source project and history"
        )
    output.mkdir(parents=True, exist_ok=True)
    cancel = threading.Event()
    previous = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.signal(sig, lambda *_: cancel.set())
    try:
        with (output / "launch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = args.config or output / "inputs/campaign.json"
            if not path.exists():
                progress.phase("event=prepare_inputs")
                if not args.source_project or not args.history:
                    raise ValueError(
                        "provide --config or both --source-project and --history"
                    )
                path = prepare(
                    args.source_project,
                    args.history,
                    output / "inputs",
                    reuse_exploration=args.reuse_exploration,
                    cli=args.codex,
                    cli_version=args.codex_version,
                    selected_datasets=args.datasets,
                )
            config = select_config(
                read_json(path), args.datasets, args.episodes_per_dataset
            )
            progress.phase("event=validate_inputs")
            validate_config(config)
            identity = {
                "config": digest(config),
                "budgets": [500] if args.only_500 else [500, 5000],
                "implementation": campaign_source_identity(),
                "agent_templates": {
                    name: AgentConfig.from_file(agent_preset(name)).fingerprint
                    for name in config["datasets"]
                },
            }
            counts = {
                k: len(v["test"]["episodes"]) for k, v in config["datasets"].items()
            }
            if args.dry_run:
                return {
                    "dry_run": True,
                    "datasets": counts,
                    "budgets": identity["budgets"],
                    "stage_5000": "disabled"
                    if args.only_500
                    else "only new scored failures",
                    "exploration": {
                        k: "historical_import"
                        if v.get("import_snapshot")
                        else "new_training"
                        for k, v in config["datasets"].items()
                    },
                }
            progress.phase("event=check_cli")
            preflight = preflight_codex_cli(
                CodexCliConfig.from_dict(
                    {
                        **config["agent"],
                        "tools": "mcp",
                    }
                ),
                repo_root=output,
            )
            identity["cli"] = preflight
            owner = output / "launch-identity.json"
            if owner.exists():
                if not args.resume or read_json(owner) != identity:
                    raise ValueError(
                        "use --resume with unchanged configuration, code and CLI"
                    )
            else:
                write_json(owner, identity)
            write_json(output / "preflight.json", preflight)
            snapshots, presets = {}, {}
            for name, dataset in config["datasets"].items():
                progress.phase(f"event=prepare_exploration dataset={name}")
                if cancel.is_set():
                    raise InterruptedError("campaign cancelled")
                snapshot = output / "snapshots" / name
                if not snapshot.exists():
                    if dataset.get("import_snapshot"):
                        progress.phase(f"event=import_exploration dataset={name}")
                        source = Path(dataset["import_snapshot"])
                        manifest = read_json(source / "manifest.json")
                        expected = {f["path"]: f["sha256"] for f in manifest["files"]}
                        if inventory(source / "workspace") != expected:
                            raise ValueError("historical exploration files changed")
                        freeze_snapshot(source / "workspace", snapshot)
                    else:
                        explorer = (
                            output / "agent-input" / f"explorer-{name}" / "agent.yaml"
                        )
                        if not explorer.exists():
                            make_preset(explorer.parent, config["agent"], dataset=name)
                        attempts = output / "exploration" / name
                        attempts.mkdir(parents=True, exist_ok=True)
                        number = (
                            max(
                                (
                                    int(p.name.removeprefix("attempt-"))
                                    for p in attempts.glob("attempt-*")
                                ),
                                default=0,
                            )
                            + 1
                        )
                        directory = attempts / f"attempt-{number:04d}"
                        directory.mkdir()
                        progress.phase(
                            f"event=explore dataset={name} artifacts={directory}"
                        )
                        command = [
                            sys.executable,
                            "-m",
                            "robots.vln_shared.cli",
                            "explore-worker",
                            "--config",
                            str(path.resolve()),
                            "--dataset",
                            name,
                            "--preset",
                            str(explorer),
                            "--output",
                            str(directory / "run"),
                            "--snapshot",
                            str(snapshot),
                        ]
                        code = run_process(
                            command, directory, cancel, args.exploration_timeout
                        )
                        if code:
                            raise RuntimeError(
                                f"exploration failed for {name}; see {directory / 'stderr.log'}"
                            )
                verify_snapshot(snapshot)
                snapshots[name] = snapshot
                preset = output / "agent-input" / name / "agent.yaml"
                if not preset.exists():
                    make_preset(preset.parent, config["agent"], snapshot, dataset=name)
                if verify_snapshot(preset.parent / "exploration") != verify_snapshot(
                    snapshot
                ):
                    raise ValueError(
                        "Agent resource differs from the frozen exploration"
                    )
                presets[name] = preset
                progress.phase(f"event=exploration_ready dataset={name}")
            cases = [
                e for d in config["datasets"].values() for e in d["test"]["episodes"]
            ]
            executor = BenchExecutor(
                config["datasets"],
                presets,
                cancel,
                timeout=args.case_timeout,
                progress=progress,
            )
            identity["presets"] = {
                k: AgentConfig.from_file(v).fingerprint for k, v in presets.items()
            }
            report = Campaign(
                output / "evaluation",
                identity,
                cases,
                snapshots,
                executor,
                concurrency=args.concurrency,
                retries=args.retries,
                cancel=cancel,
                only_500=args.only_500,
                progress=progress,
            ).run(resume=args.resume)
            report["exploration"] = {
                name: {
                    "snapshot_sha256": verify_snapshot(snapshots[name]),
                    "origin": "historical_import"
                    if dataset.get("import_snapshot")
                    else "new_training",
                    "attempts": [
                        read_json(p)
                        for p in sorted(
                            (output / "exploration" / name).glob(
                                "attempt-*/run/exploration-result.json"
                            )
                        )
                    ],
                }
                for name, dataset in config["datasets"].items()
            }
            write_json(output / "evaluation/summary.json", report)
            return report
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def smoke(config: dict, dataset: str, directory: Path, budget: int) -> dict:
    """Exercise real HTTP images and native scoring with fixed actions, without a model."""
    section = config["datasets"][dataset]["test"]
    provider = NavigationBenchmark(section["spec"], section["episodes"][:1], budget)
    provider.runtime_info()
    directory.mkdir(parents=True, exist_ok=False)
    channel = CaseMediaChannel(directory / "client/media")
    with serve(provider, directory / "service") as endpoint:
        client = BenchmarkClient(endpoint)
        try:
            task = client.tasks()[0]
            observation = client.start(task, "smoke", timeout_s=600).observation
            client.collect_observation(observation, channel, timeout_s=600)
            # End through real actions; LHPR STOP can submit multiple subtasks.
            for _ in range(budget):
                result = client.step(
                    Action(
                        benchmark=task.benchmark,
                        episode_id="smoke",
                        step_index=observation.step_index,
                        data={"actions": ["stop"]},
                    ),
                    timeout_s=600,
                )
                observation = result.observation
                client.collect_observation(observation, channel, timeout_s=600)
                if result.terminated or result.truncated:
                    break
            native = client.finish(timeout_s=600)
            client.collect_native(native, directory / "client", timeout_s=600)
            record = native.as_dict()
            write_json(directory / "smoke.json", record)
            return record
        finally:
            client.close()


def main(argv: list[str] | None = None) -> int:
    """Run preparation, a campaign, or an explicit native smoke test."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--source-project", required=True, type=Path)
    preparation.add_argument("--history", required=True, type=Path)
    preparation.add_argument("--output", required=True, type=Path)
    preparation.add_argument("--reuse-exploration", action="store_true")
    evaluation = commands.add_parser("run")
    evaluation.add_argument("--config", type=Path)
    evaluation.add_argument("--source-project", type=Path)
    evaluation.add_argument("--history", type=Path)
    evaluation.add_argument("--output", required=True, type=Path)
    evaluation.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        type=lambda name: "lhpr_vln" if name == "lhvln" else name,
        help="datasets to evaluate (default: all configured; lhvln aliases lhpr_vln)",
    )
    evaluation.add_argument(
        "--episodes-per-dataset",
        type=int,
        help="evaluate the first N test episodes of each selected dataset (default: all)",
    )
    evaluation.add_argument(
        "--only-500",
        action="store_true",
        help="stop after the 500-action stage, including scored failures",
    )
    evaluation.add_argument("--reuse-exploration", action="store_true")
    evaluation.add_argument("--codex", default="codex")
    evaluation.add_argument("--codex-version", default="codex-cli 0.161.0")
    evaluation.add_argument("--concurrency", type=int, default=4)
    evaluation.add_argument("--retries", type=int, default=1)
    evaluation.add_argument("--case-timeout", type=float, default=7200)
    evaluation.add_argument("--exploration-timeout", type=float, default=14400)
    evaluation.add_argument("--resume", action="store_true")
    evaluation.add_argument("--dry-run", action="store_true")
    evaluation.add_argument(
        "--progress-interval",
        type=float,
        default=30,
        help="seconds between waiting heartbeats on stderr (default: 30)",
    )
    check = commands.add_parser("smoke")
    check.add_argument("--config", required=True, type=Path)
    check.add_argument("--dataset", required=True, choices=["lhpr_vln", "r2r", "rxr"])
    check.add_argument("--output", required=True, type=Path)
    check.add_argument("--budget", type=int, default=5)
    worker = commands.add_parser("explore-worker", help=argparse.SUPPRESS)
    worker.add_argument("--config", required=True, type=Path)
    worker.add_argument("--dataset", required=True)
    worker.add_argument("--preset", required=True, type=Path)
    worker.add_argument("--output", required=True, type=Path)
    worker.add_argument("--snapshot", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        if args.episodes_per_dataset is not None and args.episodes_per_dataset < 1:
            parser.error("--episodes-per-dataset must be positive")
        if args.datasets and len(set(args.datasets)) != len(args.datasets):
            parser.error("--datasets must not repeat a dataset or its alias")
    if args.command == "prepare":
        print(
            prepare(
                args.source_project,
                args.history,
                args.output,
                reuse_exploration=args.reuse_exploration,
            )
        )
        return 0
    if args.command == "explore-worker":
        from robots.vln_shared.exploration import explore

        config = read_json(args.config)
        explore(
            config["datasets"][args.dataset]["train"],
            args.preset,
            args.output,
            args.snapshot,
        )
        return 0
    report = (
        run(args)
        if args.command == "run"
        else smoke(read_json(args.config), args.dataset, args.output, args.budget)
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "cases"},
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0 if report.get("complete", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
