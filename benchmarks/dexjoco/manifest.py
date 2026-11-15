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

"""Generate a registry manifest for an actually published DexJoCo image."""

from __future__ import annotations

import argparse
from pathlib import Path

from jsonschema import Draft202012Validator

from roborsi.benchmarks.registry import benchmark_manifest_schema
from roborsi.evaluation.result import write_json_atomic
from robots.dexjoco.benchmark_service import DexJocoBenchmark
from robots.dexjoco.contracts import CAPABILITIES, COMMIT, LIMITS, SEEDS


def build_manifest(provider: DexJocoBenchmark, image: str) -> dict:
    """Describe in-image native assets and the five-seed two-session cohort."""
    repository, separator, digest = image.partition("@")
    if not separator:
        raise ValueError("Supply an actual repository@sha256:digest")
    runtime = provider.runtime_info()
    catalog = {task.task_id: task for task in provider.tasks()}
    document = {
        "schema_version": 1,
        "benchmark": "dexjoco",
        "benchmark_version": runtime.benchmark_version,
        "benchmark_commit": COMMIT,
        "image_repository": repository,
        "image_digest": digest,
        "assets_version": f"in-image-dexjoco-{COMMIT}",
        "assets": [],
        "adapter_version": runtime.adapter_version,
        "adapter_command": [
            "python",
            "-m",
            "roborsi.evaluation.service_host",
            "--kind",
            "benchmark",
            "--factory",
            "robots.dexjoco.benchmark_service:make_dexjoco_benchmark",
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
        "capabilities": CAPABILITIES,
        "evaluation": runtime.evaluation,
        "tasks": [
            {
                "task_id": name,
                "seeds": list(SEEDS),
                "native": catalog[name].native,
                "instruction": catalog[name].instruction,
                "max_steps": catalog[name].max_steps,
            }
            for name in LIMITS
        ],
    }
    Draft202012Validator(benchmark_manifest_schema()).validate(document)
    return document


def main() -> None:
    """Write the manifest under an operator-owned registry root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = build_manifest(DexJocoBenchmark(args.upstream), args.image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output, document)


if __name__ == "__main__":
    main()
