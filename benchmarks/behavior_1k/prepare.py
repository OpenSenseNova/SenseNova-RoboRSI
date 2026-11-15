# Copyright 2026 The SenseNova-RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Seal licensed assets, inspect native tasks, and generate a Registry manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from roborsi.benchmarks.registry import (
    benchmark_manifest_schema,
    directory_sha256,
    snapshot_directory,
)
from roborsi.evaluation.messages import BenchmarkRuntimeInfo
from roborsi.evaluation.result import write_json_atomic
from robots.behavior_1k.benchmark_service import BehaviorBenchmark
from robots.behavior_1k.catalog import NATIVE_COMMIT, read_catalog
from robots.behavior_1k.roborsi_adapter import BehaviorRoboRSIAdapter


def _asset_subpath(path: Path, manifest_dir: Path) -> str:
    """Validate the lexical mount path before hashing or resolving symlinks."""
    if ".." in path.parts or ".." in manifest_dir.parts:
        raise ValueError("asset paths cannot contain parent traversal")
    absolute = path.absolute()
    relative = absolute.relative_to(manifest_dir.absolute())
    if not relative.parts or "\\" in relative.as_posix():
        raise ValueError("asset path must be a POSIX subpath of the manifest directory")
    if any(part.is_symlink() for part in (absolute, *absolute.parents)):
        raise ValueError("asset path contains a symlink")
    return relative.as_posix()


def build_manifest(
    image: str,
    assets: Path,
    catalog_root: Path,
    runtime: BenchmarkRuntimeInfo,
    manifest_dir: Path,
) -> dict:
    """Derive a manifest from actual image identity and locally verified assets."""
    repository, sep, digest = image.partition("@")
    if (
        not sep
        or runtime.benchmark != "behavior_1k"
        or runtime.benchmark_commit != NATIVE_COMMIT
    ):
        raise ValueError("provide the actual image digest and matching runtime info")
    paths = {
        path: _asset_subpath(path, manifest_dir) for path in (assets, catalog_root)
    }
    tasks = read_catalog(catalog_root / "catalog.json")
    catalog = json.loads((catalog_root / "catalog.json").read_text())
    data_hash = directory_sha256(assets)
    if catalog["assets_sha256"] != data_hash:
        raise ValueError("catalog and native assets differ")
    doc = {
        "schema_version": 1,
        "benchmark": "behavior_1k",
        "benchmark_version": runtime.benchmark_version,
        "benchmark_commit": runtime.benchmark_commit,
        "adapter_version": runtime.adapter_version,
        "image_repository": repository,
        "image_digest": digest,
        "assets_version": data_hash,
        "assets": [
            {
                "name": name,
                "kind": "directory",
                "path": paths[path],
                "version": content_hash,
                "sha256": content_hash,
            }
            for name, path, content_hash in (
                ("behavior_1k", assets, data_hash),
                ("behavior_1k-catalog", catalog_root, directory_sha256(catalog_root)),
            )
        ],
        "adapter_command": [
            "python",
            "-m",
            "roborsi.evaluation.service_host",
            "--kind",
            "benchmark",
            "--factory",
            "robots.behavior_1k.benchmark_service:make_behavior_benchmark",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
        "service_port": 8000,
        "resources": {
            "gpus": 1,
            "cpus": 8,
            "memory_mb": 32768,
            "startup_timeout_s": 900,
        },
        "capabilities": tasks[0].capabilities,
        "evaluation": runtime.evaluation,
        "tasks": [
            {
                "task_id": t.task_id,
                "seeds": [t.seed],
                "instruction": t.instruction,
                "native": t.native,
                "max_steps": t.max_steps,
            }
            for t in tasks
        ],
    }
    Draft202012Validator(benchmark_manifest_schema()).validate(doc)
    return doc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    seal = commands.add_parser("assets")
    seal.add_argument("--source", type=Path, required=True)
    seal.add_argument("--output-dir", type=Path, required=True)
    catalog = commands.add_parser("catalog")
    catalog.add_argument("--task", action="append", required=True)
    catalog.add_argument("--output-dir", type=Path, required=True)
    catalog.add_argument("--work-dir", type=Path, required=True)
    manifest = commands.add_parser("manifest")
    manifest.add_argument("--image", required=True)
    manifest.add_argument("--assets", type=Path, required=True)
    manifest.add_argument("--catalog", type=Path, required=True)
    manifest.add_argument(
        "--runtime-info",
        type=Path,
        required=True,
        help="Runtime info read from the actual delivery image",
    )
    manifest.add_argument("--output", type=Path, required=True)
    runtime_parser = commands.add_parser("runtime-info")
    runtime_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "runtime-info":
        write_json_atomic(args.output, BehaviorBenchmark().runtime_info().as_dict())
    elif args.command == "assets":
        for name in (
            "2026-challenge-task-instances",
            "behavior-1k-assets",
            "omnigibson-robot-assets",
        ):
            if not (args.source / name).is_dir():
                parser.error(f"missing asset directory: {name}")
        # The licensed native asset key remains in the asset mount, never the image.
        if not (args.source / "omnigibson.key").is_file():
            parser.error("native assets require their licensed omnigibson.key")
        root = snapshot_directory(args.source, args.output_dir / "snapshots")
        write_json_atomic(
            args.output_dir / "asset.json",
            {
                "name": "behavior_1k",
                "kind": "directory",
                "path": str(root),
                "version": root.name,
                "sha256": root.name,
            },
        )
    elif args.command == "catalog":
        provider = BehaviorBenchmark()
        if len(set(args.task)) != len(args.task):
            parser.error("task names must be unique")
        args.output_dir.mkdir(parents=True, exist_ok=False)
        args.work_dir.mkdir(parents=True, exist_ok=False)
        before = directory_sha256(provider.data)
        tasks = []
        for i, task_id in enumerate(args.task):
            adapter = BehaviorRoboRSIAdapter(args.work_dir / str(i))
            try:
                tasks.append(adapter.inspect({"task_id": task_id}).as_dict())
            finally:
                adapter.close()
        if directory_sha256(provider.data) != before:
            raise ValueError("assets changed during native catalog inspection")
        write_json_atomic(
            args.output_dir / "catalog.json",
            {
                "schema_version": 1,
                "native_commit": NATIVE_COMMIT,
                "assets_sha256": before,
                "tasks": tasks,
            },
        )
    else:
        runtime = BenchmarkRuntimeInfo.from_dict(
            json.loads(args.runtime_info.read_text())
        )
        write_json_atomic(
            args.output,
            build_manifest(
                args.image, args.assets, args.catalog, runtime, args.output.parent
            ),
        )


if __name__ == "__main__":
    main()
