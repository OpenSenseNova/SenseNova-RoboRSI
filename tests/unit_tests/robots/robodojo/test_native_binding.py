# Copyright 2026 The RoboRSI Contributors.
# Licensed under the Apache License, Version 2.0.

"""The simulator imports and opens exactly the advertised native resources."""

import sys
from types import ModuleType

import pytest


def test_native_imports_are_bound_to_verified_checkout(tmp_path, monkeypatch):
    from robots.robodojo.native import activate_native_source

    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in ("env", "src", "task", "utils"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    activate_native_source(tmp_path)
    assert sys.path[0] == str(tmp_path.resolve())


def test_already_imported_other_native_checkout_is_rejected(tmp_path, monkeypatch):
    from robots.robodojo.native import activate_native_source

    foreign = ModuleType("env")
    foreign.__file__ = str(tmp_path.parent / "other/env/__init__.py")
    monkeypatch.setitem(sys.modules, "env", foreign)
    with pytest.raises(ValueError, match="another checkout"):
        activate_native_source(tmp_path)


def test_native_assets_must_be_the_selected_snapshot(tmp_path):
    from robots.robodojo.benchmark_service import verify_assets_binding

    native = tmp_path / "native"
    native.mkdir()
    (native / "Assets").mkdir()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    with pytest.raises(ValueError, match="Assets"):
        verify_assets_binding(native, snapshot)
    verify_assets_binding(native, native / "Assets")


def test_native_transport_uses_the_same_checkout(tmp_path, monkeypatch):
    from robots.robodojo.native import activate_native_source

    (tmp_path / "XPolicyLab/client_server").mkdir(parents=True)
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in ("env", "src", "task", "utils", "client_server"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    activate_native_source(tmp_path)
    assert sys.path[:2] == [str(tmp_path), str(tmp_path / "XPolicyLab")]


def test_native_kit_cache_is_per_worker_and_rejects_symlink(tmp_path):
    from robots.robodojo.native import native_kit_arguments

    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    assert native_kit_arguments(first) == [
        "--kit_args=--portable-root " + str(first / "kit-storage")
    ]
    (second / "kit-storage").symlink_to(first / "kit-storage", target_is_directory=True)
    with pytest.raises(ValueError, match="owned path"):
        native_kit_arguments(second)


def test_native_gpu_ordinal_is_explicit_and_validated(monkeypatch):
    from robots.robodojo.native import native_gpu_index

    monkeypatch.delenv("ROBORSI_ROBODOJO_GPU_INDEX", raising=False)
    assert native_gpu_index() == 0
    monkeypatch.setenv("ROBORSI_ROBODOJO_GPU_INDEX", "1")
    assert native_gpu_index() == 1
    monkeypatch.setenv("ROBORSI_ROBODOJO_GPU_INDEX", "-1")
    with pytest.raises(ValueError, match="nonnegative integer"):
        native_gpu_index()
