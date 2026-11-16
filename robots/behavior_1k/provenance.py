# Copyright 2026 The SenseNova-RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Verify the pinned native checkout without importing simulator dependencies."""

from __future__ import annotations

import subprocess
from pathlib import Path

from robots.behavior_1k.catalog import NATIVE_COMMIT


def verify_native_source(root: Path) -> str:
    """Reject another commit or modified tracked native files."""
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    changes = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"],
        text=True,
    )
    if revision != NATIVE_COMMIT or changes.strip():
        raise ValueError("BEHAVIOR native checkout must match the clean pinned commit")
    return revision
