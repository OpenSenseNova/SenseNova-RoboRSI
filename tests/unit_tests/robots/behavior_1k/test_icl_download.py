# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Shared frozen HF packages feed the existing Agent without resealing."""

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from roborsi.agents.config import AgentConfig
from roborsi.components.icl.build import build_package
from roborsi.components.icl.package import ContextPackage
from roborsi.embodied_agent import EmbodiedAgent
from robots.behavior_1k.agents.baseline.download import ICL_REPOSITORY, download_icl
from robots.behavior_1k.agents.baseline.factory import make_agent
from robots.behavior_1k.agents.baseline.icl import attach_preset

REVISION = "a" * 40
PREFIX = "behavior/v1.0.0/tasks/sorting_vegetables"
ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def hub(tmp_path, monkeypatch):
    import huggingface_hub
    import huggingface_hub.utils

    workspace = tmp_path / "source"
    workspace.mkdir()
    (workspace / "TASK.md").write_text("Learned sorting_vegetables official 21D notes")
    (workspace / "SOURCES.md").write_text("200 public training demonstrations")
    (workspace / "helper.py").write_text("raise RuntimeError('do not execute')\n")
    (workspace / ".hidden").write_bytes(b"preserve original bytes\r\n")
    package = build_package(
        workspace,
        tmp_path / "remote",
        package_id="behavior-sorting_vegetables-icl",
        metadata={
            "benchmark": "behavior_1k",
            "task_name": "sorting_vegetables",
            "stage": "icl",
            "action_profile": "r1pro_eef21_v1",
        },
    )
    files = {
        f"{PREFIX}/{p.relative_to(package.root).as_posix()}": p.read_bytes()
        for p in package.root.rglob("*")
        if p.is_file()
    }
    files["behavior/v1.0.0/provenance/source.txt"] = b"not a workspace input"
    files["behavior/v1.0.0/tasks/other/manifest.json"] = b"other task"
    info = SimpleNamespace(
        sha=REVISION,
        siblings=[SimpleNamespace(rfilename=n, size=len(b)) for n, b in files.items()],
    )
    calls = []

    def repo_info(repo_id, **kwargs):
        assert repo_id == ICL_REPOSITORY
        assert kwargs == {
            "repo_type": "dataset",
            "revision": REVISION,
            "files_metadata": True,
        }
        return info

    def url(repo_id, filename, **kwargs):
        assert repo_id == ICL_REPOSITORY
        assert kwargs == {"repo_type": "dataset", "revision": REVISION}
        return "https://hf.example/" + filename

    def request(req):
        name = req.url.path.removeprefix("/")
        calls.append(name)
        return httpx.Response(200, content=files[name])

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kw: real_client(transport=httpx.MockTransport(request), **kw),
    )
    monkeypatch.setattr(huggingface_hub.utils, "build_hf_headers", lambda: {})
    monkeypatch.setattr(
        huggingface_hub, "HfApi", lambda: SimpleNamespace(repo_info=repo_info)
    )
    monkeypatch.setattr(huggingface_hub, "hf_hub_url", url)
    return SimpleNamespace(info=info, calls=calls, files=files, package=package)


def fetch(tmp_path, **kwargs):
    return download_icl(
        task="sorting_vegetables",
        revision=REVISION,
        cache_dir=tmp_path / "cache",
        **kwargs,
    )


def test_frozen_package_and_receipt_feed_the_existing_agent(tmp_path, hub, monkeypatch):
    package = fetch(tmp_path)
    assert package.manifest_sha256 == hub.package.manifest_sha256
    assert set(hub.calls) == {
        name for name in hub.files if name.startswith(PREFIX + "/")
    }
    for name in hub.calls:
        assert (
            package.root / name.removeprefix(PREFIX + "/")
        ).read_bytes() == hub.files[name]
    receipt = json.loads((package.root.parent / "receipt.json").read_text())
    assert receipt["source"]["repo_id"] == ICL_REPOSITORY
    assert receipt["source"]["revision"] == REVISION
    assert receipt["source"]["subpath"] == PREFIX
    count = len(hub.calls)
    assert fetch(tmp_path).manifest_sha256 == package.manifest_sha256
    assert len(hub.calls) == count
    config_path = attach_preset(
        package.root, ROOT / "robots/behavior_1k/agent.yaml", tmp_path / "agent"
    )
    config = AgentConfig.load(config_path)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-fixture")
    case = tmp_path / "case"
    case.mkdir()
    agent = make_agent(case, config=config)
    try:
        assert isinstance(agent, EmbodiedAgent)
        assert agent.planner == "codex_cli" and agent.codex_cli.tools == "mcp"
        assert (
            case / "workspace/.hidden"
        ).read_bytes() == b"preserve original bytes\r\n"
        assert not (case / "workspace/provenance").exists()
        ContextPackage.load(config.resources["icl"])
    finally:
        agent.close()


def test_corrupt_payload_is_not_published(tmp_path, hub):
    hub.files[f"{PREFIX}/payload/TASK.md"] = b"corrupted"
    with pytest.raises(ValueError, match="inventory differs"):
        fetch(tmp_path)
    assert not list((tmp_path / "cache").rglob("receipt.json"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("task_name", "other"),
        ("benchmark", "other"),
        ("stage", "evaluation"),
        ("action_profile", "joint23"),
    ],
)
def test_package_identity_is_checked(tmp_path, hub, field, value):
    name = f"{PREFIX}/manifest.json"
    manifest = json.loads(hub.files[name])
    manifest["metadata"][field] = value
    hub.files[name] = json.dumps(manifest).encode()
    with pytest.raises(ValueError, match="task, action profile or learning stage"):
        fetch(tmp_path)


@pytest.mark.parametrize(
    "revision,version", [("main", "v1.0.0"), (REVISION, "../escape")]
)
def test_unpinned_or_escaping_selection_rejected(tmp_path, hub, revision, version):
    with pytest.raises(ValueError):
        download_icl(
            task="sorting_vegetables",
            revision=revision,
            version=version,
            cache_dir=tmp_path / "cache",
        )
    assert hub.calls == []


def test_size_limit_checked_before_download(tmp_path, hub):
    with pytest.raises(ValueError, match="byte limit"):
        fetch(tmp_path, max_bytes=1)
    assert hub.calls == []
