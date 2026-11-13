# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Download or learn ICL files and prepare a standalone task Agent source tree."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import yaml

from roborsi.agents.config import AgentConfig
from roborsi.components.icl.build import (
    ContextExports,
    _checked_absolute,
    build_package,
)
from roborsi.components.icl.package import ContextPackage
from roborsi.embodied_agent import LocalToolSpec
from roborsi.tools.toolkit import ToolResult
from robots.behavior_1k.agents.baseline.factory import prepare_workspace
from robots.behavior_1k.catalog import DATA_REVISION
from robots.behavior_1k.official import EEF21


def learn(
    config: AgentConfig, materials: Path, case_dir: Path, output: Path
) -> ContextPackage:
    """Use EmbodiedAgent.run with public demos, then seal only learned files."""
    materials, case_dir, output = (
        _checked_absolute(path) for path in (materials, case_dir, output)
    )
    paths = (materials, case_dir, output)
    if any(
        left == right or left in right.parents or right in left.parents
        for i, left in enumerate(paths)
        for right in paths[i + 1 :]
    ):
        raise ValueError("materials, case directory and output must be disjoint")
    from benchmarks.behavior_1k.official_demonstrations import validate_rich_materials

    index = validate_rich_materials(
        materials, source_revision=DATA_REVISION, action_profile=EEF21
    )
    if len(index["episodes"]) != 200:
        raise ValueError(
            "official ICL requires all 200 public demonstrations for the task"
        )
    case_dir.mkdir(parents=True, exist_ok=False)
    workspace = case_dir / "workspace"
    workspace.mkdir()
    prepare_workspace(config, workspace)
    # Real copies prevent the CLI from changing the pinned demonstration source.
    shutil.copytree(materials, workspace / "training")
    shutil.copytree(config.resources["icl_tools"], workspace / "tools")
    (workspace / "learned").mkdir()

    def demonstration(arguments: dict) -> ToolResult:
        episode_id = arguments.get("episode_id")
        matches = [row for row in index["episodes"] if row["id"] == episode_id]
        return ToolResult(
            "icl_demonstration",
            {"episode": matches[0]} if matches else {"error": "unknown episode_id"},
        )

    agent = config.embodied_agent(
        case_dir,
        working_dir=workspace,
        local_tools=[
            LocalToolSpec(
                "icl_demonstration",
                "Read one public demonstration's paths and valid video intervals.",
                {
                    "type": "object",
                    "properties": {"episode_id": {"type": "string"}},
                    "required": ["episode_id"],
                    "additionalProperties": False,
                },
                handler=demonstration,
            )
        ],
    )
    try:
        result = agent.run(
            f"Prepare task {index['task_name']} using training/ and save output in learned/.",
            system_prompt=config.resources["icl_prompt"].read_text(encoding="utf-8"),
        )
    finally:
        agent.close()
    if result.error:
        raise RuntimeError(f"ICL failed: {result.error}")
    # Agent wheel resources omit Python bytecode. Remove generated caches before
    # sealing so the ICL manifest never requires files absent from that wheel.
    learned = workspace / "learned"
    if learned.is_symlink():
        raise ValueError("learned output must be a regular directory")
    for cache in sorted(
        learned.rglob("__pycache__"), key=lambda path: len(path.parts), reverse=True
    ):
        if cache.is_symlink():
            raise ValueError("learned output must not contain cache links")
        shutil.rmtree(cache)
    for pattern in ("*.pyc", "*.pyo"):
        for bytecode in learned.rglob(pattern):
            bytecode.unlink()
    for name in ("TASK.md", "SOURCES.md"):
        if not (workspace / "learned" / name).read_text(encoding="utf-8").strip():
            raise ValueError(f"ICL needs nonempty {name}")
    return build_package(
        workspace / "learned",
        output,
        package_id=f"behavior-{index['task_name']}-icl",
        exports=ContextExports(memory=(("Task", "TASK.md"), ("Sources", "SOURCES.md"))),
        metadata={
            "benchmark": "behavior_1k",
            "task_name": index["task_name"],
            "stage": "icl",
            "action_profile": EEF21,
            "dataset_revision": DATA_REVISION,
            "materials_sha256": hashlib.sha256(
                (materials / "index.json").read_bytes()
            ).hexdigest(),
            "agent_config_sha256": config.fingerprint,
        },
    )


def attach_preset(package: Path, baseline: Path, output: Path) -> Path:
    """Copy declared baseline inputs and learned data outside the source tree."""
    config = AgentConfig.load(baseline)
    context = ContextPackage.load(package)
    output = output.resolve()
    if output.is_relative_to(config.source_root):
        raise ValueError("prepared Agent output must be outside source_root")
    if output.is_relative_to(context.root):
        raise ValueError("prepared Agent output must be outside the ICL package")
    output.mkdir(parents=True, exist_ok=False)
    for original in config.input_files:
        target = output / original.relative_to(config.source_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
    path = output / config.path.relative_to(config.source_root)
    learned = output / "robots/behavior_1k/agents/baseline/learned_icl"
    shutil.copytree(context.root, learned)
    import os

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document.setdefault("resources", {})["icl"] = os.path.relpath(learned, path.parent)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    config.verify()
    prepared = AgentConfig.load(path)
    if prepared.source_root != output:
        raise ValueError("prepared Agent must retain a relocatable source_root")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser(
        "download", help="Fetch a frozen task from SenseNova-RoboRSI-Packages"
    )
    download.add_argument("--task", required=True)
    download.add_argument("--revision", required=True, help="Full HF commit ID")
    download.add_argument("--version", default="v1.0.0")
    download.add_argument("--cache-dir", type=Path, required=True)
    download.add_argument("--max-bytes", type=int, default=1024 * 1024 * 1024)
    learning = commands.add_parser("learn")
    learning.add_argument(
        "--agent-config",
        type=Path,
        default=Path(__file__).with_name("icl.yaml"),
    )
    learning.add_argument("--materials", type=Path, required=True)
    learning.add_argument(
        "--agent-set",
        action="append",
        default=[],
        help="Agent configuration override as PATH=JSON; never supply credentials",
    )
    learning.add_argument("--case-dir", type=Path, required=True)
    learning.add_argument("--output", type=Path, required=True)
    preset = commands.add_parser("preset")
    preset.add_argument("--package", type=Path, required=True)
    preset.add_argument(
        "--baseline", type=Path, default=Path(__file__).with_name("agent.yaml")
    )
    preset.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "download":
        from robots.behavior_1k.agents.baseline.download import download_icl

        package = download_icl(
            task=args.task,
            revision=args.revision,
            version=args.version,
            cache_dir=args.cache_dir,
            max_bytes=args.max_bytes,
        )
        print(package.root)
    elif args.command == "learn":
        learn(
            AgentConfig.load(args.agent_config, overrides=args.agent_set),
            args.materials,
            args.case_dir,
            args.output,
        )
    else:
        print(attach_preset(args.package, args.baseline, args.output))


if __name__ == "__main__":
    main()
