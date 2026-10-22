# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

import pytest

from roborsi.benchmarks.registry import directory_sha256


def test_libero_pro_assets_are_content_addressed(tmp_path, monkeypatch):
    from benchmarks.libero_pro import assets

    source = tmp_path / "source"
    staged = tmp_path / "staged"
    for root in (source, staged):
        for kind, suffix in (("bddl_files", "bddl"), ("init_files", "init")):
            for suite in assets.REQUIRED_SUITES:
                directory = root / kind / suite
                directory.mkdir(parents=True)
                (directory / f"task.{suffix}").write_text(f"{kind}:{suite}")
    extra = source / "bddl_files/unrelated_internal_suite"
    extra.mkdir()
    (extra / "unused.bddl").write_text("must not enter the public snapshot")
    digest = directory_sha256(staged)
    monkeypatch.setattr(assets, "ASSET_SHA256", digest)

    output = tmp_path / "output"
    descriptor = assets.prepare_asset(source, output)

    assert descriptor["name"] == "libero_pro"
    assert descriptor["sha256"] == digest
    assert directory_sha256(output / descriptor["path"]) == digest
    context = tmp_path / "context"
    context.mkdir()
    (context / "Dockerfile").write_text("FROM scratch\n")
    target = assets.stage_image_asset(output / descriptor["path"], context)
    assert directory_sha256(target) == digest
    with pytest.raises(FileExistsError, match="already exists"):
        assets.stage_image_asset(output / descriptor["path"], context)


def test_libero_pro_assets_reject_unlocked_content(tmp_path):
    from benchmarks.libero_pro.assets import REQUIRED_SUITES, prepare_asset

    source = tmp_path / "source"
    for kind in ("bddl_files", "init_files"):
        for suite in REQUIRED_SUITES:
            (source / kind / suite).mkdir(parents=True)

    with pytest.raises(ValueError, match="evaluation assets differ"):
        prepare_asset(source, tmp_path / "output")


def test_libero_pro_assets_require_every_public_suite(tmp_path):
    from benchmarks.libero_pro.assets import prepare_asset

    source = tmp_path / "source"
    (source / "bddl_files").mkdir(parents=True)
    (source / "init_files").mkdir()
    with pytest.raises(FileNotFoundError, match="libero_spatial_task"):
        prepare_asset(source, tmp_path / "output")
