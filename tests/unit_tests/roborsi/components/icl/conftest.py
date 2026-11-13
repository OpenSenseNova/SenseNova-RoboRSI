# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Small self-contained reference materials for ICL component contracts."""

import pytest


@pytest.fixture
def source_tree(tmp_path):
    root = tmp_path / "source"
    files = {
        "notes/task.md": "reference\n",
        "skills/grasp/SKILL.md": "---\nname: grasp\ndescription: Grasp reference\n---\nInspect first.\n",
        "programs/analyze.py": "raise RuntimeError('must not execute')\n",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
    return root


@pytest.fixture
def sealed_pkg(source_tree, tmp_path):
    from roborsi.components.icl import ContextExports, build_package

    return build_package(
        source_tree,
        tmp_path / "sealed",
        package_id="reference",
        exports=ContextExports(
            memory=(("Task", "notes/task.md"),), skill_paths=("skills",)
        ),
    )


@pytest.fixture
def symlink_support(tmp_path):
    target = tmp_path / "symlink-target"
    target.touch()
    link = tmp_path / "symlink-probe"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Host cannot create symbolic links: {exc}")
