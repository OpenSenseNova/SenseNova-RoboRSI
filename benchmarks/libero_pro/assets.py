# Copyright 2026 The RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Seal and stage the LIBERO-PRO evaluation assets for an image build."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from roborsi.benchmarks.registry import directory_sha256, snapshot_directory
from roborsi.evaluation.result import write_json_atomic

ASSET_VERSION = "zxy-libero-pro-eafdb809426b13153aa1e4c42d6601844217dfec-eval-v1"
ASSET_SHA256 = "04ae2020acf89315108a68743f99f8fe9c4cb9fc47e67712298a25a7179d616a"


def prepare_asset(source: Path, output: Path) -> dict[str, str]:
    """Copy, verify and seal only the evaluation files required at runtime."""
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    required = ("bddl_files", "init_files")
    for name in required:
        if not (source / name).is_dir():
            raise FileNotFoundError(
                f"LIBERO-PRO asset directory is missing: {source / name}"
            )
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="libero-pro-", dir=output) as temporary:
        staging = Path(temporary) / "evaluation"
        staging.mkdir()
        for name in required:
            shutil.copytree(source / name, staging / name)
        actual = directory_sha256(staging)
        if actual != ASSET_SHA256:
            raise ValueError(
                f"LIBERO-PRO evaluation assets differ: {actual}, expected {ASSET_SHA256}"
            )
        snapshot = snapshot_directory(staging, output / "snapshots")
    asset = {
        "name": "libero_pro",
        "kind": "directory",
        "version": ASSET_VERSION,
        "path": snapshot.relative_to(output).as_posix(),
        "sha256": ASSET_SHA256,
    }
    write_json_atomic(output / "asset.json", asset)
    return asset


def stage_image_asset(snapshot: Path, build_context: Path) -> Path:
    """Copy one verified snapshot into an exported Docker build context."""
    snapshot = snapshot.expanduser().resolve()
    build_context = build_context.expanduser().resolve()
    if not (build_context / "Dockerfile").is_file():
        raise FileNotFoundError(f"exported build context is missing: {build_context}")
    actual = directory_sha256(snapshot)
    if actual != ASSET_SHA256:
        raise ValueError(
            f"LIBERO-PRO evaluation assets differ: {actual}, expected {ASSET_SHA256}"
        )
    target = build_context / "evaluation-assets"
    if target.exists():
        raise FileExistsError(f"image asset destination already exists: {target}")
    shutil.copytree(snapshot, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="directory containing bddl_files/ and init_files/",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--build-context",
        type=Path,
        help="also stage the verified snapshot into an exported Docker context",
    )
    args = parser.parse_args()
    asset = prepare_asset(args.source, args.output_dir)
    if args.build_context is not None:
        stage_image_asset(args.output_dir / asset["path"], args.build_context)
    print(json.dumps(asset, sort_keys=True))


if __name__ == "__main__":
    main()
