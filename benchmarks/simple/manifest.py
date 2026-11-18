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

"""Create a standard Registry manifest from a delivered image and local assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from roborsi.benchmarks.registry import benchmark_manifest_schema, directory_sha256
from roborsi.evaluation.docker_common import validate_image_reference
from roborsi.evaluation.messages import BenchmarkRuntimeInfo
from robots.simple.catalog import Catalog
from robots.simple.contracts import EVALUATION_RULES, SIMPLE_REVISION, SOURCES


def build_manifest(
    image: str, catalog: Path, assets: Path, output: Path, runtime: dict
) -> dict:
    """Hash materialized assets and preserve every task's native identity.

    Both asset roots must be below the manifest directory, without symlinks.
    Runtime metadata must come from the delivered service, not a placeholder.
    """
    validate_image_reference(image)
    info = BenchmarkRuntimeInfo.from_dict(runtime)
    evaluation = {k: v for k, v in EVALUATION_RULES.items() if k != "score_protocol"}
    if (
        info.benchmark != "simple"
        or info.benchmark_commit != SIMPLE_REVISION
        or info.evaluation != evaluation
        or not info.adapter_version.startswith("roborsi-source-sha256:")
    ):
        raise ValueError("runtime does not match this SIMPLE profile")
    root = output.parent.resolve()
    entries = []
    for name, path, version in (
        ("simple-catalog", catalog, SOURCES["dataset_revision"]),
        ("simple-assets", assets, SIMPLE_REVISION),
    ):
        if path.is_symlink():
            raise ValueError("Registry asset root must not be a symlink")
        relative = path.absolute().relative_to(root)
        if not relative.parts or ".." in relative.parts:
            raise ValueError("Registry assets must be inside the manifest directory")
        current = root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise ValueError("Registry asset path must not contain a symlink")
        entries.append(
            {
                "name": name,
                "version": version,
                "kind": "directory",
                "path": relative.as_posix(),
                "sha256": directory_sha256(path),
            }
        )
    tasks = Catalog(catalog / "catalog.json").tasks()
    capabilities = tasks[0].capabilities
    if set(capabilities) != {"action_schema", "observation_channels"}:
        raise ValueError(
            "re-export the catalog with the current SIMPLE prepare command"
        )
    if any(t.capabilities != capabilities for t in tasks):
        raise ValueError("SIMPLE catalog capabilities differ between tasks")
    document = {
        "schema_version": 1,
        "benchmark": "simple",
        "benchmark_version": info.benchmark_version,
        "benchmark_commit": info.benchmark_commit,
        "adapter_version": info.adapter_version,
        "image_repository": image.split("@")[0],
        "image_digest": image.split("@")[1],
        "assets_version": hashlib.sha256(
            json.dumps(entries, sort_keys=True).encode()
        ).hexdigest(),
        "assets": entries,
        "adapter_command": [
            "env",
            "OMNI_KIT_ACCEPT_EULA=Y",
            "python",
            "-m",
            "roborsi.evaluation.service_host",
            "--kind",
            "benchmark",
            "--factory",
            "robots.simple.benchmark_service:make_simple_benchmark",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
        "service_port": 8000,
        "resources": {
            "gpus": 1,
            "cpus": 16,
            "memory_mb": 32768,
            "startup_timeout_s": 180,
        },
        "capabilities": capabilities,
        "evaluation": evaluation,
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
    Draft202012Validator(benchmark_manifest_schema()).validate(document)
    return document


def main() -> None:
    """Write a new manifest only after explicit NVIDIA EULA acceptance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--runtime-info", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accept-nvidia-eula", action="store_true", required=True)
    args = parser.parse_args()
    document = build_manifest(
        args.image,
        args.catalog,
        args.assets,
        args.output,
        json.loads(args.runtime_info.read_text()),
    )
    with args.output.open("x") as stream:
        json.dump(document, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
