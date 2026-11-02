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

"""Prepare frozen evaluation inputs with existing licensed scenes, without downloads."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path, PurePosixPath

from robots.vln_shared.assets import manifest
from robots.vln_shared.storage import file_digest, read_json


def _relative(name: str) -> Path:
    """Accept only portable descendant paths from the pinned input index."""
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or "\\" in name
        or any(part in {"", ".", ".."} for part in name.split("/"))
    ):
        raise ValueError(f"invalid input path: {name}")
    return Path(*path.parts)


def prepare(
    package: Path,
    dataset: str,
    scenes_root: Path,
    output: Path,
    image: str,
    *,
    budget: int = 500,
    copy_scenes: bool = False,
) -> Path:
    """Verify inputs, reuse scenes and seal a fresh single-dataset Registry.

    Args:
        package: Extracted, checksum-verified evaluation input package.
        dataset: One of lhpr_vln, r2r or rxr.
        scenes_root: HM3D root for LHPR; a root containing mp3d/ for VLNCE.
        output: New Registry root, outside both source trees, without symlinks.
        image: Published benchmark repository@sha256:digest.
        budget: Physical action budget, 500 or 5000.
        copy_scenes: Explicitly allow copies instead of default hard links.

    Returns:
        Path to the generated benchmark manifest.
    """
    if dataset not in {"lhpr_vln", "r2r", "rxr"} or budget not in {500, 5000}:
        raise ValueError("select a supported dataset and a 500 or 5000 budget")
    package, scenes_root = package.resolve(), scenes_root.resolve()
    output = output.absolute()
    if any(path.is_symlink() for path in (output, *output.parents)):
        raise ValueError("Registry output must not contain symlinks")
    if output.exists():
        raise FileExistsError(output)
    if any(output.is_relative_to(root) for root in (package, scenes_root)):
        raise ValueError("Registry output must be outside input directories")
    index = read_json(package / "input-index.json")
    if index.get("schema_version") != 1:
        raise ValueError("unsupported evaluation input index")
    entry = index["datasets"][dataset]
    benchmark = "lhpr_vln" if dataset == "lhpr_vln" else "vlnce"
    if entry["benchmark"] != benchmark:
        raise ValueError("dataset and benchmark identity differ")
    scene_subdir = _relative(entry["scene_subdir"])
    transfers = []
    destinations = set()
    for group, source_root in [("files", package / dataset), ("scenes", scenes_root)]:
        for row in entry[group]:
            relative = _relative(row["path"])
            destination = relative if group == "files" else scene_subdir / relative
            if destination in destinations:
                raise ValueError(f"duplicate asset destination: {destination}")
            destinations.add(destination)
            source = source_root / relative
            if not source.is_file():
                raise FileNotFoundError(f"required {group} input missing: {source}")
            if (
                source.stat().st_size != row["bytes"]
                or file_digest(source) != row["sha256"]
            ):
                raise ValueError(f"input checksum mismatch: {source}")
            transfers.append((source, destination, group == "scenes"))
    if Path("catalog.json") not in destinations:
        raise ValueError("input package must include a pinned catalog")
    catalog = read_json(package / dataset / "catalog.json")
    if (
        catalog["spec"]["dataset"] != dataset
        or len(catalog["episodes"]) != entry["task_count"]
    ):
        raise ValueError("catalog identity or task count differs from the input index")
    if not copy_scenes:
        existing_parent = next(parent for parent in output.parents if parent.exists())
        if any(
            is_scene and source.stat().st_dev != existing_parent.stat().st_dev
            for source, _, is_scene in transfers
        ):
            raise ValueError(
                "hard links require the same filesystem; select --copy-scenes explicitly"
            )
    # Preflight every byte before creating output. Failed writes keep the new partial
    # directory for inspection; an existing directory is never silently overwritten.
    assets = output / benchmark / "assets"
    assets.mkdir(parents=True, exist_ok=False)
    for source, relative, is_scene in transfers:
        target = assets / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if is_scene and not copy_scenes:
            os.link(source.resolve(), target)
        else:
            shutil.copyfile(source, target)
    # Native input hashes are checked again by portable_provider; full scene contents
    # are sealed by the existing Registry manifest implementation.
    destination = output / benchmark / "manifest.json"
    manifest(assets, destination, image, budget=budget)
    return destination


def main() -> None:
    """Create a Registry from a released small input package and local scenes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--dataset", choices=["lhpr_vln", "r2r", "rxr"], required=True)
    parser.add_argument("--scenes-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--budget", type=int, choices=[500, 5000], default=500)
    parser.add_argument("--copy-scenes", action="store_true")
    args = parser.parse_args()
    print(
        prepare(
            args.package,
            args.dataset,
            args.scenes_root,
            args.output,
            args.image,
            budget=args.budget,
            copy_scenes=args.copy_scenes,
        )
    )


if __name__ == "__main__":
    main()
