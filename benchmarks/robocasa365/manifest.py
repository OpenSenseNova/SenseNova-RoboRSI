# Copyright 2026 The SenseNova-RoboRSI Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Write the RoboRSI Registry manifest for a pushed RoboCasa365 image."""

from __future__ import annotations

import argparse
from pathlib import Path

from jsonschema import Draft202012Validator

from roborsi.benchmarks.registry import benchmark_manifest_schema
from roborsi.evaluation.result import write_json_atomic
from robots.robocasa.benchmark_service import (
    CAPABILITIES,
    TARGET50,
    make_task,
    runtime_info,
)

ADAPTER_COMMAND = [
    "python",
    "-m",
    "roborsi.evaluation.service_host",
    "--kind",
    "benchmark",
    "--factory",
    "robots.robocasa.benchmark_service:make_robocasa365_benchmark",
    "--host",
    "0.0.0.0",
    "--port",
    "8000",
]


def manifest(image: str, cells: str = "all") -> dict:
    """Describe the image, its in-image assets and the Target50 cells."""
    repository, separator, digest = image.partition("@")
    if not separator:
        raise ValueError("image needs a repository and digest, separated by @")
    runtime = runtime_info()
    tasks = []
    for split in TARGET50["splits"].values():
        for name in split["tasks"]:
            seeds = [s for s in split["seeds"] if cells == "all" or s == 1]
            cell = [make_task(name, seed) for seed in seeds]
            tasks.append(
                {
                    "task_id": name,
                    "seeds": seeds,
                    "native": cell[0].native,
                    "instruction": cell[0].instruction,
                    "instructions_by_seed": {
                        str(task.seed): task.instruction for task in cell
                    },
                    "max_steps": cell[0].max_steps,
                }
            )
    document = {
        "schema_version": 1,
        "benchmark": "robocasa365",
        "benchmark_version": runtime.benchmark_version,
        "benchmark_commit": runtime.benchmark_commit,
        "image_repository": repository,
        "image_digest": digest,
        # Kitchen assets, the simulator and RLDX-1 weights are inside the image.
        "assets_version": f"in-image-robocasa-{runtime.benchmark_commit}",
        "assets": [],
        "adapter_version": runtime.adapter_version,
        "adapter_command": ADAPTER_COMMAND,
        "service_port": 8000,
        # RLDX-1 runs in the same container, started by the service itself.
        "resources": {
            "gpus": 1,
            "cpus": 8,
            "memory_mb": 32768,
            "startup_timeout_s": 900,
        },
        "capabilities": CAPABILITIES,
        "evaluation": runtime.evaluation,
        "tasks": tasks,
    }
    Draft202012Validator(benchmark_manifest_schema()).validate(document)
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="repository@sha256:digest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cells", choices=("all", "seed1"), default="all")
    args = parser.parse_args()
    write_json_atomic(args.output, manifest(args.image, args.cells))


if __name__ == "__main__":
    main()
