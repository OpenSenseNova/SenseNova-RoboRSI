# Copyright 2026 The RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Export a Long++ or Ultra-Long RoboRSI manifest from the pinned LiLo image."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from jsonschema import Draft202012Validator

from roborsi.benchmarks.registry import benchmark_manifest_schema
from roborsi.evaluation.result import write_json_atomic
from robots.libero.three_benchmark_adapter import ACTION_SCHEMA
from robots.libero.three_benchmark_service import ThreeLiberoBenchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="repository@sha256:digest")
    parser.add_argument(
        "--benchmark",
        choices=("libero_long_plus_plus", "libero_ultra_long"),
        required=True,
    )
    parser.add_argument("--asset", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=("official", "migration_subset"), default="official"
    )
    parser.add_argument("--plus9-on-failure", action="store_true")
    args = parser.parse_args()
    repository, separator, digest = args.image.partition("@")
    if not separator:
        parser.error("--image needs repository@sha256:digest")
    os.environ["ROBORSI_LIBERO_BENCHMARK"] = args.benchmark
    os.environ["ROBORSI_LIBERO_PROFILE"] = args.profile
    os.environ["ROBORSI_LIBERO_PLUS9"] = "1" if args.plus9_on_failure else "0"
    provider = ThreeLiberoBenchmark()
    runtime = provider.runtime_info()
    if runtime is None:
        parser.error("image does not expose its pinned LiLo commit")
    assets = [json.loads(path.read_text(encoding="utf-8")) for path in args.asset]
    document = {
        "schema_version": 1,
        "benchmark": provider.name,
        "benchmark_version": runtime.benchmark_version,
        "benchmark_commit": runtime.benchmark_commit,
        "image_repository": repository,
        "image_digest": digest,
        "assets_version": "+".join(item["version"] for item in assets) or "embedded",
        "assets": assets,
        "adapter_version": runtime.adapter_version,
        "adapter_command": [
            "python",
            "-m",
            "robots.libero.three_benchmark_serve",
            "--benchmark",
            provider.name,
            "--profile",
            args.profile,
            *(["--plus9-on-failure"] if args.plus9_on_failure else []),
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
        "service_port": 8000,
        "resources": {
            "gpus": 1,
            "cpus": 2,
            "memory_mb": 6144,
            "startup_timeout_s": 180,
        },
        "capabilities": {
            "action_schema": ACTION_SCHEMA,
            "observation_channels": [
                "main_images",
                "wrist_images",
                "states",
                "robot_state_feedback",
                "instance_pose_feedback",
            ],
        },
        "evaluation": runtime.evaluation,
        "tasks": [case.manifest_task() for case in provider._cases()],
    }
    Draft202012Validator(benchmark_manifest_schema()).validate(document)
    write_json_atomic(args.output, document)


if __name__ == "__main__":
    main()
