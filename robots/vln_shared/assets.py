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

"""Seal prepared native assets into the standard RoboRSI Registry manifest."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from jsonschema import Draft202012Validator

from roborsi.benchmarks.registry import benchmark_manifest_schema, directory_sha256
from robots.vln_shared.benchmark import EVALUATION, portable_provider
from robots.vln_shared.storage import file_digest, read_json, write_json


def stage(
    section: dict,
    mappings: dict[str, str],
    destination: Path,
    native_python: str = "/opt/native/bin/python",
) -> Path:
    """Copy explicit input roots, dereference source links, and rebase catalog paths.

    Mapping keys are source directories; values are disjoint relative output
    directories. This operation can copy large licensed scenes and is explicit.
    The finished bundle contains regular copies, never links to the source.
    """
    destination = destination.resolve()
    roots = {Path(k).absolute(): Path(v) for k, v in mappings.items()}
    if not roots or any(
        v.is_absolute() or ".." in v.parts or not v.parts for v in roots.values()
    ):
        raise ValueError("asset mappings need nonempty relative destinations")
    for source, target in roots.items():
        if destination.is_relative_to(source.resolve()):
            raise ValueError("asset staging must be outside the source directories")
        if any(
            target != other
            and (target.is_relative_to(other) or other.is_relative_to(target))
            for other in roots.values()
        ):
            raise ValueError("asset destination roots must not overlap")
    if len(set(roots.values())) != len(roots):
        raise ValueError("asset destination roots must be distinct")

    def relative(value: str) -> str:
        path = Path(value).absolute()
        matches = [root for root in roots if path.is_relative_to(root)]
        if not matches:
            raise ValueError(f"input has no asset mapping: {value}")
        root = max(matches, key=lambda p: len(p.parts))
        return (roots[root] / path.relative_to(root)).as_posix()

    spec = dict(section["spec"])
    for key in (
        "repo_root",
        "runtime_root",
        "data_root",
        "scenes_root",
        "scene_dataset",
        "regions_root",
    ):
        if key in spec:
            spec[key] = relative(spec[key])
    inputs = {relative(k): v for k, v in spec["inputs"].items()}
    destination.mkdir(parents=True, exist_ok=False)
    for source, target in roots.items():
        shutil.copytree(
            source,
            destination / target,
            symlinks=False,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
        )
    for name, expected in inputs.items():
        if file_digest(destination / name) != expected:
            raise ValueError(f"copied input differs from the pinned catalog: {name}")
    # Habitat scene-dataset configurations can embed absolute asset paths.
    if "scene_dataset" in spec:
        benchmark = "lhpr_vln" if spec["dataset"] == "lhpr_vln" else "vlnce"

        def rebase(value):
            if isinstance(value, dict):
                return {k: rebase(v) for k, v in value.items()}
            if isinstance(value, list):
                return [rebase(v) for v in value]
            if isinstance(value, str) and value.startswith("/"):
                return f"/roborsi/assets/{benchmark}/" + relative(value)
            return value

        path = destination / spec["scene_dataset"]
        write_json(path, rebase(read_json(path)))
        inputs[spec["scene_dataset"]] = file_digest(path)
    spec.update(python=native_python, inputs=inputs)
    write_json(
        destination / "catalog.json", {"spec": spec, "episodes": section["episodes"]}
    )
    directory_sha256(destination)
    return destination


def manifest(assets: Path, output: Path, image: str, budget: int = 500) -> dict:
    """Validate a copied portable bundle and create a manifest next to its assets."""
    assets, output = assets.absolute(), output.absolute()
    relative = assets.relative_to(output.parent)
    if not relative.parts or ".." in relative.parts or output.is_relative_to(assets):
        raise ValueError(
            "assets must be a proper subdirectory of the manifest directory"
        )
    if any(p.is_symlink() for p in (assets, *assets.parents)):
        raise ValueError("asset path must not contain symlinks")
    if output.exists():
        raise FileExistsError(output)
    provider = portable_provider(assets / "catalog.json", budget)
    runtime = provider.runtime_info()
    checksum = directory_sha256(assets)
    repository, separator, image_digest = image.partition("@")
    if not separator:
        raise ValueError("provide an image repository@sha256:digest")
    tasks = provider.tasks()
    document = {
        "schema_version": 1,
        "benchmark": provider.name,
        "benchmark_version": provider.spec["native_commit"][:12],
        "benchmark_commit": provider.spec["native_commit"],
        "image_repository": repository,
        "image_digest": image_digest,
        "assets_version": checksum,
        "assets": [
            {
                "name": provider.name,
                "version": checksum,
                "kind": "directory",
                "path": relative.as_posix(),
                "sha256": checksum,
            }
        ],
        "adapter_version": runtime.adapter_version,
        "adapter_command": [
            "env",
            f"ROBORSI_VLN_BUDGET={budget}",
            "python",
            "-m",
            "roborsi.evaluation.service_host",
            "--kind",
            "benchmark",
            "--factory",
            f"robots.{provider.name}.benchmark_service:make_benchmark",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
        "service_port": 8000,
        "resources": {
            "gpus": 1,
            "cpus": 2,
            "memory_mb": 8192,
            "startup_timeout_s": 600,
        },
        "capabilities": {
            "action_schema": tasks[0].capabilities["action_schema"],
            "observation_channels": ["front", "left", "right"]
            if provider.name == "lhpr_vln"
            else ["front"],
        },
        "evaluation": EVALUATION,
        "tasks": [
            {
                "task_id": t.task_id,
                "seeds": [t.seed],
                "native": t.native,
                "instruction": t.instruction,
                "max_steps": t.max_steps,
            }
            for t in tasks
        ],
    }
    Draft202012Validator(benchmark_manifest_schema()).validate(document)
    write_json(output, document)
    return document


def main() -> None:
    """Seal a staged bundle; large licensed data is never written into Git."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    seal = commands.add_parser("manifest")
    seal.add_argument("--assets", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    seal.add_argument("--image", required=True)
    prepare = commands.add_parser("stage")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--split", choices=["train", "test"], default="test")
    prepare.add_argument("--mapping", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "manifest":
        manifest(args.assets, args.output, args.image)
    else:
        section = read_json(args.config)["datasets"][args.dataset][args.split]
        stage(section, read_json(args.mapping), args.output)


if __name__ == "__main__":
    main()
