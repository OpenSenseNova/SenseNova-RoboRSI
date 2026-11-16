# Copyright 2026 The SenseNova-RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export native task descriptions from inside the built LIBERO image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from roborsi.benchmarks.registry import benchmark_manifest_schema
from roborsi.evaluation.result import write_json_atomic
from robots.libero.roborsi_adapter import ACTION_SCHEMA, runtime_info


def main() -> None:
    """Write a manifest for a pushed image and a host-side asset snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="repository@sha256:digest")
    parser.add_argument("--asset", type=Path, required=True, help="asset.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--max-steps", type=int, default=600)
    args = parser.parse_args()

    from rlinf.envs.libero.utils import benchmark

    repository, separator, digest = args.image.partition("@")
    if not separator:
        parser.error("--image needs a repository and digest, separated by @")
    asset = json.loads(args.asset.read_text())
    suite = benchmark.get_benchmark(args.suite)()
    runtime = runtime_info()
    if runtime is None:
        parser.error("LIBERO image does not record its source commit")
    document = {
        "schema_version": 1,
        "benchmark": "libero",
        "benchmark_version": runtime.benchmark_version,
        "benchmark_commit": runtime.benchmark_commit,
        "image_repository": repository,
        "image_digest": digest,
        "assets_version": asset["version"],
        "assets": [asset],
        "adapter_version": runtime.adapter_version,
        "adapter_command": [
            "python",
            "-m",
            "roborsi.evaluation.service_host",
            "--kind",
            "benchmark",
            "--factory",
            "robots.libero.benchmark_service:make_libero_benchmark",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
        "service_port": 8000,
        "resources": {
            "gpus": 0,
            "cpus": 2,
            "memory_mb": 4096,
            "startup_timeout_s": 120,
        },
        "capabilities": {
            "action_schema": ACTION_SCHEMA,
            "observation_channels": ["main_images", "wrist_images", "states"],
        },
        "evaluation": runtime.evaluation,
        "tasks": [
            {
                "task_id": str(index),
                "seeds": args.seed,
                "native": {"suite": args.suite},
                "instruction": suite.get_task(index).language,
                "max_steps": args.max_steps,
            }
            for index in range(suite.n_tasks)
        ],
    }
    Draft202012Validator(benchmark_manifest_schema()).validate(document)
    write_json_atomic(args.output, document)


if __name__ == "__main__":
    main()
