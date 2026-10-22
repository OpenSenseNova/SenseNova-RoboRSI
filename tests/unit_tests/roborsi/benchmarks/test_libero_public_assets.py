# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

import json
from pathlib import Path

from scripts.libero_three import download_assets


def test_public_downloader_disables_auth_and_materializes_hub_links(
    tmp_path: Path, monkeypatch
):
    cache = tmp_path / "cache"
    cache.mkdir()
    blob = cache / "blob"
    blob.write_text("public payload")
    icl = cache / "icl"
    for path in download_assets.ICL_PATHS.values():
        package = icl / path / "package"
        package.mkdir(parents=True)
        (package / "manifest.json").symlink_to(blob)
    pro = cache / "pro"
    pro.mkdir()
    tokenizer = cache / "tokenizer.json"
    tokenizer.symlink_to(blob)
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(icl if kwargs["repo_id"] == download_assets.ICL_REPOSITORY else pro)

    def hf_hub_download(**kwargs):
        calls.append(kwargs)
        return str(tokenizer)

    def prepare_agent(roots, tokenizer_file, destination, *, repository):
        assert repository.is_dir()
        assert not tokenizer_file.is_symlink()
        assert all(
            not path.is_symlink() for root in roots.values() for path in root.rglob("*")
        )
        destination.mkdir()
        (destination / "asset.json").write_text("{}")
        return destination

    def prepare_pro(source, destination):
        assert source == pro
        snapshot = destination / "snapshots" / "digest"
        snapshot.mkdir(parents=True)
        return {"path": "snapshots/digest", "sha256": "digest"}

    monkeypatch.setattr(download_assets, "snapshot_download", snapshot_download)
    monkeypatch.setattr(download_assets, "hf_hub_download", hf_hub_download)
    monkeypatch.setattr(download_assets, "prepare_agent_assets", prepare_agent)
    monkeypatch.setattr(download_assets, "prepare_asset", prepare_pro)

    output = download_assets.download(
        tmp_path / "output", repository=Path(__file__).resolve().parents[4]
    )

    assert output.is_dir()
    assert calls and all(call["token"] is False for call in calls)
    assert json.loads((output / "sources.json").read_text())["schema_version"] == 1
