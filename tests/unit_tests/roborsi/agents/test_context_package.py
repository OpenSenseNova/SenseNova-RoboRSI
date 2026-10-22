# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Sealed Agent context packages reject incomplete or altered inputs."""

import hashlib
import json
import os
from pathlib import Path

import pytest

from roborsi.agents.context_package import ContextPackage, load_package_map


def _package(root: Path) -> Path:
    sealed = root / "sealed"
    payload = sealed / "payload"
    (payload / "skills" / "pick").mkdir(parents=True)
    (payload / "TASK.md").write_text("Remember this task.\n")
    (payload / "initial.txt").write_text("Initial demonstration.\n")
    (payload / "skills" / "pick" / "SKILL.md").write_text("# Pick\n")
    files = {
        path.relative_to(payload).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in payload.rglob("*")
        if path.is_file()
    }
    manifest = {
        "schema_version": 1,
        "package_type": "rpent.icl.context",
        "package_id": "demo-context",
        "files": files,
        "memory": [{"title": "TASK", "path": "TASK.md"}],
        "skills": [],
        "skill_paths": ["skills/pick"],
        "initial_context": [{"type": "text", "path": "initial.txt"}],
        "metadata": {"evaluation_trajectories_used": False},
    }
    (sealed / "manifest.json").write_text(json.dumps(manifest))
    return sealed


def test_context_package_exposes_only_verified_declared_resources(tmp_path: Path):
    package = ContextPackage.load(_package(tmp_path))
    assert package.package_id == "demo-context"
    assert package.memory[0][0] == "TASK"
    assert package.skill_paths[0].name == "pick"
    assert package.initial_context[0].path.name == "initial.txt"
    assert len(package.manifest_sha256) == 64


@pytest.mark.parametrize("change", ("alter", "extra", "missing"))
def test_context_package_rejects_inventory_changes(tmp_path: Path, change: str):
    sealed = _package(tmp_path)
    if change == "alter":
        (sealed / "payload/TASK.md").write_text("changed")
    elif change == "extra":
        (sealed / "payload/extra.txt").write_text("extra")
    else:
        (sealed / "payload/TASK.md").unlink()
    with pytest.raises(ValueError, match="inventory differs"):
        ContextPackage.load(sealed)


def test_context_package_rejects_links_and_unsafe_map_paths(tmp_path: Path):
    sealed = _package(tmp_path)
    target = sealed / "payload/initial.txt"
    target.unlink()
    os.symlink("TASK.md", target)
    with pytest.raises(ValueError, match="symlinks"):
        ContextPackage.load(sealed)

    mapping = tmp_path / "map.json"
    mapping.write_text(
        json.dumps({"schema_version": 1, "packages": {"task": "../sealed"}})
    )
    (tmp_path / "packages").mkdir()
    with pytest.raises(ValueError, match="normalized relative"):
        load_package_map(mapping, tmp_path / "packages")


def test_context_package_map_loads_relative_sealed_packages(tmp_path: Path):
    root = tmp_path / "packages"
    package = _package(root / "task-00")
    mapping = tmp_path / "map.json"
    mapping.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "packages": {"libero_long_plus_plus:t00-i01": "task-00/sealed"},
            }
        )
    )
    loaded = load_package_map(mapping, root)
    assert loaded["libero_long_plus_plus:t00-i01"].root == package


def test_context_package_rejects_package_id_path_traversal(tmp_path: Path):
    sealed = _package(tmp_path)
    manifest = json.loads((sealed / "manifest.json").read_text())
    manifest["package_id"] = "../outside"
    (sealed / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="package_id"):
        ContextPackage.load(sealed)
