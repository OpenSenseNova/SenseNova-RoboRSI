# Copyright 2026 The RoboRSI Contributors.
# Licensed under the Apache License, Version 2.0.

"""Navigation source registration is independent of the companion adapters."""

import json

import pytest

from benchmarks import package
from benchmarks.package import prepare_context
from roborsi.benchmarks.runtime_source import (
    benchmark_source_digest,
    benchmark_source_files,
)


@pytest.fixture(params=["vlnce", "lhpr_vln"])
def checkout(tmp_path, request):
    """Exercise service source isolation with the actual context exporter."""
    benchmark = request.param
    root = tmp_path / "checkout"
    for name in benchmark_source_files(benchmark):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# declared server source\n")
    build = root / "benchmarks" / benchmark
    build.mkdir(parents=True)
    (build / "Dockerfile").write_text("FROM scratch\n")
    (build / "Dockerfile.dockerignore").write_text("**\n")
    (build / "build-inputs.json").write_text(
        json.dumps({"schema_version": 1, "inputs": []})
    )
    return root, benchmark


def test_export_excludes_agent_and_other_benchmark_sources(checkout, tmp_path):
    root, benchmark = checkout
    other = "lhpr_vln" if benchmark == "vlnce" else "vlnce"
    excluded = (
        "robots/vln_shared/cli.py",
        "robots/vln_shared/campaign.py",
        "robots/vln_shared/exploration.py",
        "robots/vln_shared/execution.py",
        "robots/vln_shared/agents/baseline/factory.py",
        "robots/vln_shared/agents/baseline/prompt.md",
        f"robots/{other}/native.py",
        f"benchmarks/{benchmark}/auth.json",
    )
    for name in excluded:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("private Agent material\n")
    context = prepare_context(benchmark, tmp_path / "context", source_root=root)
    runtime = context / "benchmark-runtime"
    assert not any(
        b"private Agent material" in path.read_bytes()
        for path in context.rglob("*")
        if path.is_file()
    )
    before = benchmark_source_digest(root, benchmark)
    assert benchmark_source_digest(runtime, benchmark) == before
    assert json.loads((context / "source.json").read_text())["sha256"] == before
    for name in excluded:
        (root / name).write_text("changed excluded implementation\n")
    assert benchmark_source_digest(root, benchmark) == before

    for name in (
        f"robots/{benchmark}/native.py",
        "robots/vln_shared/native_worker.py",
        "roborsi/evaluation/benchmark_http.py",
    ):
        path = root / name
        original = path.read_text()
        path.write_text(original + "# service changed\n")
        assert benchmark_source_digest(root, benchmark) != before
        path.write_text(original)


@pytest.mark.parametrize("linked", [False, True])
def test_export_rejects_missing_or_linked_native_source(checkout, tmp_path, linked):
    root, benchmark = checkout
    native = root / "robots" / benchmark / "native.py"
    native.unlink()
    if linked:
        target = tmp_path / "external.py"
        target.write_text("# external source\n")
        native.symlink_to(target)
    with pytest.raises(ValueError, match="source is missing or linked"):
        prepare_context(benchmark, tmp_path / "invalid", source_root=root)
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("benchmark", ["lhpr_vln", "vlnce"])
def test_cli_exports_navigation_service_context(tmp_path, monkeypatch, benchmark):
    """The documented image-build command must accept both navigation names."""
    output = tmp_path / "context"
    monkeypatch.setattr(
        "sys.argv",
        ["benchmarks.package", "--benchmark", benchmark, "--output", str(output)],
    )
    package.main()
    metadata = json.loads((output / "source.json").read_text())
    assert metadata["benchmark"] == benchmark
    assert (output / "benchmark-runtime/robots" / benchmark / "native.py").is_file()
