# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""PolicyLab imports retain source files without interpreting or executing them."""

import json
import shutil

import pytest

from roborsi.components.icl import ContextExports
from roborsi.components.icl.importers import import_policylab_workspace


@pytest.fixture
def policylab_task(source_tree, tmp_path):
    task = tmp_path / "task"
    shutil.copytree(source_tree, task / "workspace")
    (task / "workspace/evidence").mkdir()
    (task / "workspace/evidence/data.bin").write_bytes(b"\x00\x01")
    provenance = {
        "training_only": True,
        "training_root": "/old/training",
        "unknown": {"preserve": [1, False]},
    }
    (task / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    (task / "README.md").write_text(
        "# Original sharing notes\n", encoding="utf-8", newline="\n"
    )
    return task


def test_import_keeps_workspace_and_provenance(policylab_task, tmp_path, caplog):
    task = policylab_task
    provenance_bytes = (task / "provenance.json").read_bytes()
    program_bytes = (task / "workspace/programs/analyze.py").read_bytes()
    package = import_policylab_workspace(
        task,
        tmp_path / "imported",
        package_id="study",
        exports=ContextExports(memory=(("Task", "notes/task.md"),)),
    )
    assert (package.root / "payload/notes/task.md").read_text() == "reference\n"
    assert (package.root / "payload/programs/analyze.py").read_bytes() == program_bytes
    assert (package.root / "payload/evidence/data.bin").read_bytes() == b"\x00\x01"
    assert package.metadata["icl"]["original_provenance"] == json.loads(
        provenance_bytes
    )
    assert package.metadata["icl"]["importer"] == "policylab"
    assert (
        package.root / "payload/__policylab_source__/provenance.json"
    ).read_bytes() == provenance_bytes
    assert (
        package.root / "payload/__policylab_source__/README.md"
    ).read_bytes() == b"# Original sharing notes\n"
    assert (
        package.memory[0][1].relative_to(package.root).as_posix()
        == "payload/notes/task.md"
    )
    assert (task / "provenance.json").read_bytes() == provenance_bytes
    assert (task / "workspace/programs/analyze.py").read_bytes() == program_bytes
    assert "historical" in caplog.text.lower()


@pytest.mark.parametrize(
    "invalid",
    [
        "missing-provenance",
        "non-object",
        "invalid-json",
        "nan",
        "batch",
        "collision",
        "missing-export",
        "outside-export",
        "nested-output",
    ],
)
def test_import_rejects_incomplete_or_unsafe_materials(
    policylab_task, tmp_path, invalid
):
    task = policylab_task
    destination = tmp_path / "bad"
    if invalid == "missing-provenance":
        (task / "provenance.json").unlink()
    elif invalid == "non-object":
        (task / "provenance.json").write_text("[]")
    elif invalid == "invalid-json":
        (task / "provenance.json").write_text("{")
    elif invalid == "nan":
        (task / "provenance.json").write_text('{"value": NaN}')
    elif invalid == "batch":
        task = tmp_path
    elif invalid == "collision":
        (task / "workspace/__policylab_source__").mkdir()
    elif invalid == "nested-output":
        destination = task / "imported"
    with pytest.raises(ValueError):
        exports = None
        if invalid == "missing-export":
            exports = ContextExports(skills=("unavailable/SKILL.md",))
        elif invalid == "outside-export":
            exports = ContextExports(memory=(("External", "/old/training/notes.md"),))
        import_policylab_workspace(
            task, destination, package_id="study", exports=exports
        )
    assert not destination.exists()


@pytest.mark.parametrize(
    "path", ["workspace", "provenance.json", "README.md", "workspace/notes"]
)
def test_import_rejects_linked_inputs(policylab_task, tmp_path, path):
    task = policylab_task
    original = task / path
    moved = tmp_path / "external"
    original.rename(moved)
    try:
        original.symlink_to(moved, target_is_directory=moved.is_dir())
    except OSError as exc:
        pytest.skip(f"Symlink unavailable: {exc}")
    with pytest.raises(ValueError, match="link"):
        import_policylab_workspace(task, tmp_path / "bad", package_id="study")
    assert not (tmp_path / "bad").exists()
