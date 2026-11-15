# Copyright 2026 The SenseNova-RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Build the exported minimal context with its pinned upstream native image."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    lock = json.loads((root / "build-lock.json").read_text())
    if (
        not (root / "benchmark-runtime").is_dir()
        or "@sha256:" not in lock["base_image"]
    ):
        parser.error("run from a fresh exported context with a pinned base image")
    subprocess.run(
        [
            "docker",
            "build",
            "--platform",
            lock["platform"],
            "--build-arg",
            "BEHAVIOR_RUNTIME_IMAGE=" + lock["base_image"],
            "--tag",
            args.tag,
            str(root),
        ],
        check=True,
    )
    if args.push:
        subprocess.run(["docker", "push", args.tag], check=True)
    subprocess.run(
        ["docker", "image", "inspect", args.tag, "--format", "{{json .RepoDigests}}"],
        check=True,
    )


if __name__ == "__main__":
    main()
