# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Fetch a frozen BEHAVIOR task package from the shared HF release repository."""

from __future__ import annotations

import re
from pathlib import Path

from roborsi.components.icl.package import ContextPackage
from roborsi.components.icl.sources import HfPackageSource, fetch_package

ICL_REPOSITORY = "sensenova/SenseNova-RoboRSI-Packages"


def download_icl(
    *,
    task: str,
    revision: str,
    cache_dir: Path,
    version: str = "v1.0.0",
    max_bytes: int = 1024 * 1024 * 1024,
) -> ContextPackage:
    """Fetch and verify a sealed task directory without rewriting its contents.

    Args:
        task: Native BEHAVIOR task name.
        revision: Full immutable commit ID in the shared HF repository.
        cache_dir: Local cache for the package and its HF provenance receipt.
        version: Published BEHAVIOR experience release, independent of the image.
        max_bytes: Maximum total downloaded and unpacked package size.

    Returns:
        Verified package under cache_dir, ready for ``icl preset --package``.
    """
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", task):
        raise ValueError("task must be a native BEHAVIOR task name")
    if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", version):
        raise ValueError("version must have the form v1.0.0")
    package = fetch_package(
        HfPackageSource(ICL_REPOSITORY, revision, f"behavior/{version}/tasks/{task}"),
        cache_dir,
        max_download_bytes=max_bytes,
        max_unpacked_bytes=max_bytes,
    )
    expected = {
        "benchmark": "behavior_1k",
        "task_name": task,
        "stage": "icl",
        "action_profile": "r1pro_eef21_v1",
    }
    if any(package.metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("ICL package task, action profile or learning stage differs")
    for name in ("TASK.md", "SOURCES.md"):
        if not (package.root / "payload" / name).read_text(encoding="utf-8").strip():
            raise ValueError(f"ICL package needs nonempty {name}")
    return package
