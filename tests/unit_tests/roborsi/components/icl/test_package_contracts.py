# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Compatibility and lightweight import contracts for ICL packages."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _package(tmp_path):
    root = tmp_path / "sealed"
    (root / "payload").mkdir(parents=True)
    (root / "payload/note.md").write_text("reference", encoding="utf-8")
    document = {
        "schema_version": 1,
        "package_type": "rpent.icl.context",
        "package_id": "reference",
        "files": {"note.md": hashlib.sha256(b"reference").hexdigest()},
        "metadata": {"unknown": {"items": [1, "中文", None]}},
    }
    (root / "manifest.json").write_text(json.dumps(document), encoding="utf-8")
    return root


def test_legacy_import_is_same_public_class():
    from roborsi.agents.context_package import ContextPackage as Legacy
    from roborsi.components.icl.package import ContextPackage

    assert Legacy is ContextPackage
    assert ContextPackage.__module__ == "roborsi.components.icl.package"


def test_unknown_metadata_and_task_map_keep_identity(tmp_path):
    from roborsi.agents.context_package import load_package_map as legacy
    from roborsi.components.icl import ContextPackage, load_package_map

    root = _package(tmp_path)
    package = ContextPackage.load(root)
    assert package.metadata == {"unknown": {"items": [1, "中文", None]}}
    mapping = tmp_path / "map.json"
    mapping.write_text(
        json.dumps({"schema_version": 1, "packages": {"task": "sealed"}})
    )
    assert legacy is load_package_map
    assert load_package_map(mapping, tmp_path)["task"] == package


@pytest.mark.parametrize("kind", ["hardlink", "fifo"])
def test_package_rejects_non_independent_files(tmp_path, kind):
    from roborsi.components.icl import ContextPackage

    root = _package(tmp_path)
    target = root / "payload/note.md"
    target.unlink()
    try:
        if kind == "hardlink":
            outside = tmp_path / "outside"
            outside.write_text("reference")
            os.link(outside, target)
        else:
            if not hasattr(os, "mkfifo"):
                pytest.skip("FIFO not supported")
            os.mkfifo(target)
    except OSError as exc:
        pytest.skip(f"File system does not support {kind}: {exc}")
    with pytest.raises(ValueError, match="hardlinks|regular files"):
        ContextPackage.load(root)


def test_core_import_does_not_import_agent_or_native_dependencies():
    code = """
import importlib.abc
import sys

class BlockNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "roborsi.embodied_agent" or fullname.split(".")[0] in {
            "torch", "mujoco", "robosuite", "openpi", "transformers",
        }:
            raise AssertionError("unexpected dependency: " + fullname)

sys.meta_path.insert(0, BlockNative())
import roborsi.components.icl
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[5],
    )
    assert result.returncode == 0, result.stderr
