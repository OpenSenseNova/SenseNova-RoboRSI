# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Verify and stage the sealed three-LIBERO Agent inputs for an image build."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

_REPOSITORY = Path(__file__).resolve().parents[2]
if str(_REPOSITORY) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY))

from roborsi.agents.context_package import load_package_map  # noqa: E402
from roborsi.benchmarks.registry import directory_sha256  # noqa: E402
from roborsi.evaluation.result import write_json_atomic  # noqa: E402

_SPECS = {
    "longpp": ("libero_long_plus_plus", 12, 12),
    "ultra": ("libero_ultra_long", 9, 9),
    "pro": ("libero_pro", 80, 40),
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_packages(
    name: str, root: Path, lock: dict, *, repository: Path
) -> dict[str, int | str]:
    benchmark, case_count, package_count = _SPECS[name]
    mapping = repository / "configs/agents/libero/context_maps" / f"{benchmark}.json"
    loaded = load_package_map(mapping, root)
    unique = {package.root for package in loaded.values()}
    if len(loaded) != case_count or len(unique) != package_count:
        raise ValueError(
            f"{name} ICL count differs: cases={len(loaded)}, packages={len(unique)}"
        )
    selected = {path.name for path in unique}
    actual = {path.name for path in root.iterdir()}
    if actual != selected or any(not path.is_dir() for path in unique):
        raise ValueError(f"{name} ICL root contains undeclared package entries")
    asset = lock["assets"][f"{benchmark}_icl"]
    if asset["package_count"] != package_count:
        raise ValueError(f"{name} package count differs from assets.lock.json")
    digest = directory_sha256(root)
    if digest != asset["snapshot_sha256"]:
        raise ValueError(f"{name} ICL snapshot differs from assets.lock.json")
    return {"cases": case_count, "packages": package_count, "sha256": digest}


def prepare(
    roots: dict[str, Path],
    tokenizer: Path,
    destination: Path,
    *,
    repository: Path,
    asset_lock: Path | None = None,
) -> Path:
    """Copy only fully verified Agent inputs into one fresh Docker input tree."""
    if set(roots) != set(_SPECS):
        raise ValueError(f"expected ICL roots {sorted(_SPECS)}, got {sorted(roots)}")
    repository = repository.resolve()
    destination = destination.resolve()
    lock = json.loads(
        (
            asset_lock
            or repository / "benchmarks/libero_three_combined/assets.lock.json"
        ).read_text(encoding="utf-8")
    )
    verified = {
        name: _verify_packages(name, root.resolve(), lock, repository=repository)
        for name, root in roots.items()
    }
    tokenizer = tokenizer.resolve()
    if tokenizer.is_symlink() or not tokenizer.is_file():
        raise ValueError("context tokenizer must be a regular non-link file")
    tokenizer_sha256 = _file_sha256(tokenizer)
    expected_tokenizer = lock["assets"]["context_tokenizer"]["sha256"]
    if tokenizer_sha256 != expected_tokenizer:
        raise ValueError("context tokenizer differs from assets.lock.json")
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for name, root in roots.items():
            shutil.copytree(root.resolve(), destination / name / "packages")
        shutil.copy2(tokenizer, destination / "tokenizer.json")
        for name in roots:
            copied = _verify_packages(
                name,
                destination / name / "packages",
                lock,
                repository=repository,
            )
            if copied != verified[name]:
                raise ValueError(f"{name} ICL identity changed while copying")
        if _file_sha256(destination / "tokenizer.json") != tokenizer_sha256:
            raise ValueError("context tokenizer changed while copying")
        write_json_atomic(
            destination / "asset.json",
            {
                "schema_version": 1,
                "packages": verified,
                "tokenizer_sha256": tokenizer_sha256,
            },
        )
        return destination
    except BaseException:
        shutil.rmtree(destination)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--longpp-root", type=Path, required=True)
    parser.add_argument("--ultra-root", type=Path, required=True)
    parser.add_argument("--pro-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = prepare(
            {
                "longpp": args.longpp_root,
                "ultra": args.ultra_root,
                "pro": args.pro_root,
            },
            args.tokenizer,
            args.output,
            repository=_REPOSITORY,
        )
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(output)


if __name__ == "__main__":
    main()
