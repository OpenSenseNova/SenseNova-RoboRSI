# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""ICL builders preserve complete material trees and publish only valid packages."""

import hashlib
import json
import os

import pytest

from roborsi.components.icl import ContextExports, build_package


def test_build_is_deterministic_and_copies_every_file(source_tree, tmp_path):
    (source_tree / "notes/经验 01.md").write_text("中文", encoding="utf-8")
    (source_tree / "empty.txt").touch()
    (source_tree / "empty-directory").mkdir()
    exports = ContextExports(
        memory=(("Task", "notes/task.md"),),
        skills=("skills/grasp/SKILL.md",),
        skill_paths=("skills",),
        initial_context=({"type": "text", "path": "notes/经验 01.md"},),
    )
    a = build_package(
        source_tree,
        tmp_path / "a",
        package_id="sample",
        exports=exports,
        metadata={"extra": {"中文": [1, None]}},
    )
    b = build_package(
        source_tree,
        tmp_path / "b",
        package_id="sample",
        exports=exports,
        metadata={"extra": {"中文": [1, None]}},
    )
    assert (a.root / "manifest.json").read_bytes() == (
        b.root / "manifest.json"
    ).read_bytes()
    assert a.manifest_sha256 == b.manifest_sha256
    assert (a.root / "payload/notes/经验 01.md").read_text(encoding="utf-8") == "中文"
    assert (a.root / "payload/empty.txt").read_bytes() == b""
    assert (a.root / "payload/empty-directory").is_dir()
    document = json.loads((a.root / "manifest.json").read_text())
    assert (
        document["files"]["notes/task.md"] == hashlib.sha256(b"reference\n").hexdigest()
    )
    assert len(document["files"]) == 5
    assert document["memory"] == [{"title": "Task", "path": "notes/task.md"}]
    assert document["initial_context"] == [{"type": "text", "path": "notes/经验 01.md"}]


@pytest.mark.parametrize(
    "kind", ["empty", "nested", "existing", "nan", "missing", "escape", "drive"]
)
def test_invalid_build_leaves_no_package(source_tree, tmp_path, kind):
    source = source_tree
    destination = tmp_path / "bad"
    kwargs = {}
    if kind == "empty":
        source = tmp_path / "empty"
        source.mkdir()
    elif kind == "nested":
        destination = source / "nested"
    elif kind == "existing":
        destination.mkdir()
        (destination / "keep").write_text("keep")
    elif kind == "nan":
        kwargs["metadata"] = {"number": float("nan")}
    elif kind == "missing":
        kwargs["exports"] = ContextExports(
            initial_context=({"type": "image", "path": "missing.png"},)
        )
    with pytest.raises(ValueError):
        if kind == "escape":
            kwargs["exports"] = ContextExports(memory=(("Task", "../outside"),))
        elif kind == "drive":
            kwargs["exports"] = ContextExports(skills=("C:/outside",))
        build_package(source, destination, package_id="test", **kwargs)
    if kind == "existing":
        assert (destination / "keep").read_text() == "keep"
    else:
        assert not destination.exists()
    assert (source_tree / "notes/task.md").read_text() == "reference\n"


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
def test_builder_rejects_links_and_special_files(source_tree, tmp_path, kind):
    target = source_tree / "invalid"
    try:
        if kind == "symlink":
            target.symlink_to(tmp_path)
        elif kind == "hardlink":
            os.link(source_tree / "notes/task.md", target)
        else:
            if not hasattr(os, "mkfifo"):
                pytest.skip("FIFO unavailable")
            os.mkfifo(target)
    except OSError as exc:
        pytest.skip(f"{kind} unavailable: {exc}")
    with pytest.raises(ValueError):
        build_package(source_tree, tmp_path / "bad", package_id="test")
    assert not (tmp_path / "bad").exists()


def test_builder_rejects_case_collisions(source_tree, tmp_path):
    (source_tree / "Case.md").write_text("first")
    (source_tree / "case.md").write_text("second")
    if len(list(source_tree.glob("*ase.md"))) != 2:
        pytest.skip("File system is case insensitive")
    with pytest.raises(ValueError, match="collision"):
        build_package(source_tree, tmp_path / "bad", package_id="test")
    assert not (tmp_path / "bad").exists()


def test_copy_failure_does_not_publish_or_modify_source(
    source_tree, tmp_path, monkeypatch
):
    import roborsi.components.icl.build as builder

    def fail_copy(*args, **kwargs):
        raise OSError("copy interrupted")

    monkeypatch.setattr(builder.shutil, "copyfile", fail_copy)
    with pytest.raises(OSError, match="copy interrupted"):
        build_package(source_tree, tmp_path / "bad", package_id="test")
    assert not (tmp_path / "bad").exists()
    assert (source_tree / "notes/task.md").read_text() == "reference\n"
    assert not list(tmp_path.glob(".bad-*"))


def test_publish_failure_does_not_leave_a_package(source_tree, tmp_path, monkeypatch):
    import roborsi.components.icl.build as builder

    def fail_rename(*args, **kwargs):
        raise OSError("publish interrupted")

    monkeypatch.setattr(builder.os, "rename", fail_rename)
    with pytest.raises(OSError, match="publish interrupted"):
        build_package(source_tree, tmp_path / "bad", package_id="test")
    assert not (tmp_path / "bad").exists()


def test_builder_rejects_source_through_directory_link(source_tree, tmp_path):
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(source_tree, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Symlinks unavailable: {exc}")
    with pytest.raises(ValueError, match="link"):
        build_package(linked / "notes", tmp_path / "bad", package_id="test")
