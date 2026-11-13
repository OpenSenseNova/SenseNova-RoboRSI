# Copyright 2026 The RoboRSI Contributors.
# Licensed under the Apache License, Version 2.0.

"""Shared synthetic native task fixture."""

import hashlib

import pytest

from robots.bench2dex import contracts, icl
from robots.bench2dex.reference.core import canonical


@pytest.fixture
def task():
    return contracts.task_for("26", 5, smoke_steps=4)


@pytest.fixture
def package(tmp_path, monkeypatch, task):
    root = tmp_path / "package"
    (root / "payload").mkdir(parents=True)
    for name in ("TASK.md", "SOURCES.md"):
        (root / "payload" / name).write_text("Synthetic offline fixture.\n")
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (root / "payload").iterdir()
    }
    manifest = {
        "setting": "task_icl_v3",
        "task_id": 26,
        "contract": icl.ICL_CATALOG["contract"],
        "preparation_model": icl.MODEL,
        "ready": True,
        "files": files,
        "robot_key": task.native["robot_key"],
        "instruction_sha256": hashlib.sha256(task.instruction.encode()).hexdigest(),
    }
    (root / "manifest.json").write_bytes(canonical(manifest))
    entry = {
        "task_id": 26,
        "manifest_sha256": contracts.file_sha256(root / "manifest.json"),
    }
    monkeypatch.setitem(icl.ICL_CATALOG, "packages", [entry])
    return root
