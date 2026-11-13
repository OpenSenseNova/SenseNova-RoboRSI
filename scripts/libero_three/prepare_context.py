# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Copy frozen ICL packages into a run-local, verified RoboRSI input bundle."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_REPOSITORY = Path(__file__).resolve().parents[2]
if str(_REPOSITORY) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY))

from roborsi.agents.context_package import ContextPackage  # noqa: E402
from roborsi.evaluation.result import write_json_atomic  # noqa: E402


def _sources(root: Path, benchmark: str) -> dict[str, Path]:
    if benchmark == "libero_long_plus_plus":
        return {
            f"t{index:02d}-i01": root / "longpp" / f"task-{index:02d}" / "sealed"
            for index in range(12)
        }
    if benchmark == "libero_ultra_long":
        return {
            f"t{index:02d}-i01": root / "ultra" / f"task-{index:02d}" / "sealed"
            for index in range(9)
        }
    if benchmark == "libero_pro":
        result = {}
        for family in ("spatial", "object", "goal", "10"):
            for index in range(10):
                package = root / f"libero_{family}" / f"task-{index:02d}" / "sealed"
                for variant in ("task", "swap"):
                    result[f"libero_{family}_{variant}-t{index:02d}-i03"] = package
        return result
    raise ValueError(f"unsupported benchmark: {benchmark}")


def _destination_name(benchmark: str, task_id: str, package: ContextPackage) -> str:
    if benchmark != "libero_pro":
        return package.package_id
    suite, task, _init_state = task_id.rsplit("-", 2)
    family = suite.removeprefix("libero_").removesuffix("_task").removesuffix("_swap")
    return f"libero-pro-context-{family}-{task}"


def prepare(source: Path, destination: Path, benchmark: str) -> None:
    """Verify and copy the packages consumed by a repository Agent preset."""
    source = source.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    packages = destination / "packages"
    packages.mkdir(parents=True)
    mapping = {}
    copied: dict[Path, str] = {}
    try:
        for task_id, package_path in _sources(source, benchmark).items():
            package = ContextPackage.load(package_path)
            relative = copied.get(package_path)
            if relative is None:
                relative = _destination_name(benchmark, task_id, package)
                target = packages / relative
                shutil.copytree(package_path, target, copy_function=shutil.copy2)
                copied[package_path] = relative
                copied_package = ContextPackage.load(target)
                if copied_package.manifest_sha256 != package.manifest_sha256:
                    raise ValueError("copied context package identity changed")
            mapping[task_id] = relative
        write_json_atomic(
            destination / "package-map.json",
            {"schema_version": 1, "packages": mapping},
        )
    except BaseException:
        shutil.rmtree(destination)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        choices=("libero_pro", "libero_long_plus_plus", "libero_ultra_long"),
        required=True,
    )
    parser.add_argument(
        "--source", type=Path, required=True, help="frozen package family root"
    )
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.destination, args.benchmark)


if __name__ == "__main__":
    main()
