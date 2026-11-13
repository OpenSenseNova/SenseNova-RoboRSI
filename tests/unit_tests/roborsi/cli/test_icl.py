# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""The ICL CLI exposes file operations without starting an Agent or model."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from roborsi.cli.icl import main
from roborsi.components.icl import ContextPackage, build_package


@pytest.fixture
def materials(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "note.md").write_text("reference\n", encoding="utf-8")
    (source / "program.py").write_text("raise RuntimeError('do not execute')")
    return source


def _json(path, document):
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_cli_build_verify_inspect(materials, tmp_path, capsys):
    spec = _json(
        tmp_path / "build.json",
        {
            "source": "source",
            "destination": "package",
            "package_id": "cli",
            "exports": {"memory": [{"title": "Note", "path": "note.md"}]},
            "metadata": {"unknown": True},
        },
    )
    assert main(["build", "--spec", str(spec)]) == 0
    built = json.loads(capsys.readouterr().out)
    assert main(["verify", str(tmp_path / "package")]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert built["manifest_sha256"] == verified["manifest_sha256"]
    assert main(["inspect", str(tmp_path / "package")]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["file_count"] == 2
    assert inspected["metadata"] == {"unknown": True}
    assert inspected["exports"]["memory"] == [{"title": "Note", "path": "note.md"}]


def test_cli_materialize_edit_snapshot_process(materials, tmp_path):
    package = build_package(materials, tmp_path / "package", package_id="reference")
    record = tmp_path / "record.json"
    workspace = tmp_path / "work"
    repo = Path(__file__).resolve().parents[4]
    env = {**os.environ, "PYTHONPATH": str(repo)}

    def run(*arguments):
        result = subprocess.run(
            [sys.executable, "-m", "roborsi.cli.icl", *map(str, arguments)],
            cwd=tmp_path,
            env=env,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    materialized = run(
        "materialize", package.root, workspace, "--workspace-record", record
    )
    assert materialized["source_manifest_sha256"] == package.manifest_sha256
    data = json.loads(record.read_text())
    assert data["schema_version"] == 1
    assert data["root"] == "work"
    (workspace / "note.md").write_text("experience")
    result = run(
        "snapshot",
        "--workspace-record",
        record,
        tmp_path / "revised",
        "--package-id",
        "revised",
    )
    revised = ContextPackage.load(result["root"])
    assert revised.metadata["icl"]["parent_manifest_sha256"] == package.manifest_sha256
    assert (revised.root / "payload/note.md").read_text() == "experience"
    assert not (revised.root / "payload/record.json").exists()
    assert ContextPackage.load(package.root).manifest_sha256 == package.manifest_sha256


@pytest.mark.parametrize("kind", ["inside", "existing"])
def test_materialize_rejects_unsafe_record_before_copy(
    materials, tmp_path, kind, capsys
):
    package = build_package(materials, tmp_path / "package", package_id="reference")
    destination = tmp_path / "work"
    record = (
        destination / "record.json" if kind == "inside" else tmp_path / "record.json"
    )
    if kind == "existing":
        record.write_text("keep")
    assert (
        main(
            [
                "materialize",
                str(package.root),
                str(destination),
                "--workspace-record",
                str(record),
            ]
        )
        != 0
    )
    assert not destination.exists()
    assert capsys.readouterr().out == ""
    if kind == "existing":
        assert record.read_text() == "keep"


@pytest.mark.parametrize(
    "change", ["schema", "hash", "unknown", "exports", "inside", "link"]
)
def test_snapshot_rejects_invalid_record(materials, tmp_path, change, capsys):
    record = tmp_path / "record.json"
    document = {
        "schema_version": 1,
        "root": "source",
        "source_package_id": "source",
        "source_manifest_sha256": "a" * 64,
        "exports": {},
    }
    if change == "schema":
        document["schema_version"] = 2
    elif change == "hash":
        document["source_manifest_sha256"] = "bad"
    elif change == "unknown":
        document["extra"] = True
    elif change == "exports":
        document["exports"] = {"skills": "../outside"}
    elif change == "inside":
        record = materials / "record.json"
        document["root"] = "."
    elif change == "link":
        link = tmp_path / "linked"
        try:
            link.symlink_to(materials, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"Symlinks unavailable: {exc}")
        document["root"] = "linked"
    _json(record, document)
    destination = tmp_path / "bad"
    assert (
        main(
            [
                "snapshot",
                "--workspace-record",
                str(record),
                str(destination),
                "--package-id",
                "bad",
            ]
        )
        != 0
    )
    assert not destination.exists()
    assert capsys.readouterr().out == ""


def test_cli_fetch_local_and_offline(materials, tmp_path, capsys):
    package = build_package(materials, tmp_path / "package", package_id="source")
    source = _json(tmp_path / "source.json", {"type": "local", "path": "package"})
    cache = tmp_path / "cache"
    assert main(["fetch", "--source-json", str(source), "--cache-dir", str(cache)]) == 0
    fetched = json.loads(capsys.readouterr().out)
    assert fetched["manifest_sha256"] == package.manifest_sha256
    assert (
        main(
            [
                "fetch",
                "--source-json",
                str(source),
                "--cache-dir",
                str(cache),
                "--offline",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["root"] == fetched["root"]
    assert (
        main(
            [
                "fetch",
                "--source-json",
                str(source),
                "--cache-dir",
                str(tmp_path / "missing"),
                "--offline",
            ]
        )
        != 0
    )
    assert capsys.readouterr().out == ""


def test_cli_rejects_mixed_source_fields(tmp_path, capsys):
    source = _json(
        tmp_path / "source.json",
        {
            "type": "local",
            "path": "package",
            "url": "https://example.invalid/archive.zip",
        },
    )
    assert (
        main(
            [
                "fetch",
                "--source-json",
                str(source),
                "--cache-dir",
                str(tmp_path / "cache"),
            ]
        )
        != 0
    )
    assert not (tmp_path / "cache").exists()
    assert capsys.readouterr().out == ""


def test_cli_import_policylab(materials, tmp_path, capsys):
    task = tmp_path / "task"
    task.mkdir()
    materials.rename(task / "workspace")
    _json(task / "provenance.json", {"training_only": True})
    assert (
        main(
            [
                "import-policylab",
                str(task),
                str(tmp_path / "package"),
                "--package-id",
                "imported",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert (
        ContextPackage.load(result["root"]).metadata["icl"]["original_provenance"][
            "training_only"
        ]
        is True
    )


def test_cli_help_works_without_native_dependencies():
    code = """
import importlib.abc
import sys
class BlockNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "roborsi.embodied_agent" or fullname.split(".")[0] in {
            "torch", "mujoco", "robosuite", "openpi", "transformers",
        }:
            raise AssertionError(fullname)
sys.meta_path.insert(0, BlockNative())
from roborsi.cli.icl import main
raise SystemExit(main(["--help"]))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "materialize" in result.stdout


def test_bad_command_is_nonzero(capsys):
    assert main(["unknown"]) != 0
    assert capsys.readouterr().out == ""


def test_record_write_failure_keeps_created_workspace(
    materials, tmp_path, monkeypatch, capsys, caplog
):
    package = build_package(materials, tmp_path / "package", package_id="reference")
    record_parent = tmp_path / "blocked-parent"
    record_parent.write_text("ordinary file")
    workspace = tmp_path / "work"
    assert (
        main(
            [
                "materialize",
                str(package.root),
                str(workspace),
                "--workspace-record",
                str(record_parent / "record.json"),
            ]
        )
        != 0
    )
    assert (workspace / "note.md").read_text() == "reference\n"
    assert "workspace was created" in caplog.text
    assert capsys.readouterr().out == ""


def test_cli_reports_transport_failure_without_traceback(
    tmp_path, capsys, caplog, monkeypatch
):
    import httpx

    original_client = httpx.Client

    def fail(request):
        raise httpx.ConnectError("sensitive transport detail", request=request)

    transport = httpx.MockTransport(fail)
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: original_client(transport=transport, **kwargs)
    )
    source = _json(
        tmp_path / "source.json",
        {
            "type": "http",
            "url": "https://example.invalid/package.zip",
            "archive_sha256": "a" * 64,
        },
    )
    assert (
        main(
            [
                "fetch",
                "--source-json",
                str(source),
                "--cache-dir",
                str(tmp_path / "cache"),
            ]
        )
        != 0
    )
    assert capsys.readouterr().out == ""
    assert "ConnectError" in caplog.text
    assert "sensitive transport detail" not in caplog.text


@pytest.mark.parametrize("relative", ["record.json", "payload/record.json"])
def test_materialize_rejects_record_inside_source_package(
    materials, tmp_path, relative, capsys
):
    package = build_package(materials, tmp_path / "package", package_id="reference")
    before = {
        path.relative_to(package.root).as_posix(): path.read_bytes()
        for path in package.root.rglob("*")
        if path.is_file()
    }
    destination = tmp_path / "work"
    record = package.root / relative

    assert (
        main(
            [
                "materialize",
                str(package.root),
                str(destination),
                "--workspace-record",
                str(record),
            ]
        )
        != 0
    )
    assert not destination.exists()
    assert capsys.readouterr().out == ""
    after = {
        path.relative_to(package.root).as_posix(): path.read_bytes()
        for path in package.root.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert ContextPackage.load(package.root).manifest_sha256 == package.manifest_sha256
