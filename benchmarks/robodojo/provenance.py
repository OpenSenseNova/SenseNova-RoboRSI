# Copyright 2026 The RoboRSI Contributors.
# Licensed under the Apache License, Version 2.0.

"""Verify the published skills, tools, geometry and runtime dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def verify_submission(root: Path) -> dict:
    """Reject missing, modified or extra files within protected submission trees."""
    manifest = json.loads((root / "benchmarks/robodojo/resource-lock.json").read_text())
    policy = root / manifest["destination"]
    expected = manifest["protected"]
    directories = {name.split("/")[0] for name in expected}
    actual = {}
    for directory in directories:
        for path in (policy / directory).rglob("*"):
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            if path.is_symlink():
                raise ValueError("Protected RoboDojo payload contains a symlink")
            if path.is_file():
                actual[path.relative_to(policy).as_posix()] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    if actual != expected:
        changed = sorted(
            name
            for name in set(actual) | set(expected)
            if actual.get(name) != expected.get(name)
        )
        raise ValueError(
            "Protected RoboDojo payload differs: " + ", ".join(changed[:10])
        )
    return {"protected_files": len(actual)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(verify_submission(args.root.resolve()), indent=2))


if __name__ == "__main__":
    main()
