# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Explicit PRO extension packaging, hashing and runtime identity."""

import shutil
from pathlib import Path

import pytest

from roborsi.benchmarks import runtime_source as source

ROOT = Path(__file__).resolve().parents[4]
EXTRA = {
    "robots/libero/pro_protocol.py",
    "robots/libero/image_tools.py",
    "robots/libero/image_tools.lock.json",
    "robots/libero/image_tool_records.py",
    "robots/libero/image_tool_adapter.py",
    "robots/libero/image_vla_bridge.py",
}


def export(tmp_path, bundle):
    for name in source.benchmark_source_files(bundle):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    return tmp_path.resolve()


def test_extension_is_explicit_and_has_no_agent_source():
    default = set(source.benchmark_source_files("libero_pro"))
    extended = set(source.benchmark_source_files("libero_pro_image_tools"))
    assert default.isdisjoint(EXTRA)
    assert extended - default == EXTRA
    assert default <= extended
    assert "robots/libero/roborsi_27b_agent.py" not in extended
    for name in ("libero", "libero_lilo", "robocasa365"):
        assert set(source.benchmark_source_files(name)).isdisjoint(EXTRA)


def test_extension_edits_do_not_change_default_digest(tmp_path):
    root = export(tmp_path, "libero_pro_image_tools")
    default = source.benchmark_source_digest(root, "libero_pro")
    extended = source.benchmark_source_digest(root, "libero_pro_image_tools")
    assert default != extended
    with (root / "robots/libero/image_tools.py").open("a") as f:
        f.write("\n# extension update\n")
    assert source.benchmark_source_digest(root, "libero_pro") == default
    assert source.benchmark_source_digest(root, "libero_pro_image_tools") != extended


@pytest.mark.parametrize("bundle", ["libero_pro", "libero_pro_image_tools"])
def test_export_verifies_only_selected_allowlist(tmp_path, monkeypatch, bundle):
    root = export(tmp_path, bundle)
    monkeypatch.setattr(source, "distributions", lambda: [])
    source.verify_benchmark_runtime(root, bundle)
    if bundle == "libero_pro":
        # The optional files must not silently creep into default packages.
        (root / "robots/libero/image_tools.py").write_text("# unexpected")
        with pytest.raises(ValueError, match="undeclared"):
            source.verify_benchmark_runtime(root, bundle)
    else:
        (root / "robots/libero/pro_protocol.py").unlink()
        with pytest.raises(ValueError, match="missing"):
            source.verify_benchmark_runtime(root, bundle)


@pytest.mark.parametrize("bundle", ["libero_pro", "libero_pro_image_tools"])
def test_runtime_info_hashes_selected_bundle(monkeypatch, bundle):
    from robots.libero.three_benchmark_service import ThreeLiberoBenchmark

    monkeypatch.setenv("ROBORSI_LIBERO_BENCHMARK", "libero_pro")
    monkeypatch.setenv("ROBORSI_LIBERO_SOURCE_BUNDLE", bundle)
    monkeypatch.setenv("ROBORSI_LIBERO_PRO_COMMIT", "a" * 40)
    monkeypatch.delenv("ROBORSI_PRO_EVAL_CONFIG", raising=False)
    monkeypatch.delenv("ROBORSI_IMAGE_OBSERVATION_TOOLS", raising=False)
    service = ThreeLiberoBenchmark()
    info = service.runtime_info()
    assert (
        info.adapter_version
        == "roborsi-source-sha256:" + source.benchmark_source_digest(ROOT, bundle)
    )
