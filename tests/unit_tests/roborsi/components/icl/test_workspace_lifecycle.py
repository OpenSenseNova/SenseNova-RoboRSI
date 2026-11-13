# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Working copies are independent files with explicit snapshot ancestry."""

from dataclasses import replace

import pytest

from roborsi.components.icl import (
    ContextExports,
    ContextPackage,
    materialize_package,
    snapshot_workspace,
)


def test_copies_and_snapshot_are_independent(sealed_pkg, tmp_path):
    a = materialize_package(sealed_pkg, tmp_path / "a")
    b = materialize_package(sealed_pkg, tmp_path / "b")
    assert a.source_package_id == "reference"
    assert a.exports == ContextExports(
        memory=(("Task", "notes/task.md"),), skill_paths=("skills",)
    )
    assert sorted(path.name for path in a.root.iterdir()) == [
        "notes",
        "programs",
        "skills",
    ]
    assert (a.root / "notes/task.md").stat().st_nlink == 1
    (a.root / "notes/task.md").write_text("new experience", encoding="utf-8")
    revised = snapshot_workspace(a, tmp_path / "revised", package_id="revision")
    assert (b.root / "notes/task.md").read_text() == "reference\n"
    assert revised.manifest_sha256 != sealed_pkg.manifest_sha256
    assert (
        revised.metadata["icl"]["parent_manifest_sha256"] == sealed_pkg.manifest_sha256
    )
    assert (
        ContextPackage.load(sealed_pkg.root).manifest_sha256
        == sealed_pkg.manifest_sha256
    )


def test_deleted_exports_require_explicit_update(sealed_pkg, tmp_path):
    workspace = materialize_package(sealed_pkg, tmp_path / "work")
    (workspace.root / "notes/task.md").unlink()
    with pytest.raises(ValueError):
        snapshot_workspace(workspace, tmp_path / "bad", package_id="bad")
    assert not (tmp_path / "bad").exists()
    revised = snapshot_workspace(
        workspace, tmp_path / "revised", package_id="revision", exports=ContextExports()
    )
    assert revised.memory == ()
    assert ContextPackage.load(sealed_pkg.root) == sealed_pkg


@pytest.mark.parametrize(
    "metadata",
    [
        {"icl": {"parent_manifest_sha256": "0" * 64}},
        {"icl": "invalid"},
        {"number": float("nan")},
    ],
)
def test_snapshot_rejects_invalid_metadata(sealed_pkg, tmp_path, metadata):
    workspace = materialize_package(sealed_pkg, tmp_path / "work")
    with pytest.raises(ValueError):
        snapshot_workspace(
            workspace, tmp_path / "bad", package_id="bad", metadata=metadata
        )
    assert not (tmp_path / "bad").exists()
    assert (workspace.root / "notes/task.md").read_text() == "reference\n"


def test_matching_parent_preserves_metadata(sealed_pkg, tmp_path):
    workspace = materialize_package(sealed_pkg, tmp_path / "work")
    revised = snapshot_workspace(
        workspace,
        tmp_path / "revised",
        package_id="revision",
        metadata={
            "extra": True,
            "icl": {"parent_manifest_sha256": sealed_pkg.manifest_sha256, "custom": 3},
        },
    )
    assert revised.metadata["extra"] is True
    assert revised.metadata["icl"]["custom"] == 3


@pytest.mark.parametrize("change", ["content", "identity", "nested", "existing"])
def test_materialize_rejects_invalid_input(sealed_pkg, tmp_path, change):
    destination = tmp_path / "work"
    if change == "content":
        (sealed_pkg.root / "payload/notes/task.md").write_text("tampered")
    elif change == "identity":
        sealed_pkg = replace(sealed_pkg, manifest_sha256="0" * 64)
    elif change == "nested":
        destination = sealed_pkg.root / "work"
    else:
        destination.mkdir()
    with pytest.raises(ValueError):
        materialize_package(sealed_pkg, destination)
    if change != "existing":
        assert not destination.exists()
