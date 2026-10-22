# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pinned model assets download only declared files and verify reusable caches."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import traceback
from pathlib import Path

import huggingface_hub
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/evaluation/downloads.py"
REVISION = "1234567890abcdef1234567890abcdef12345678"
PREFIX = "robocasa365/v1.0.0"


def digest(content):
    return hashlib.sha256(content).hexdigest()


@pytest.fixture
def downloader():
    assert SCRIPT.is_file(), "the pinned asset downloader is missing"
    spec = importlib.util.spec_from_file_location("model_eval_downloads", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def package(tmp_path, monkeypatch):
    files = {
        "knowledge_packages.json": b'{"packages": []}',
        "tasks/open_drawer/payload/example.py": b"print('example')\n",
    }
    manifest = "".join(f"{digest(data)}  {name}\n" for name, data in files.items())
    remote = {**files, "MANIFEST.sha256": manifest.encode()}
    calls = []

    def fetch(**kwargs):
        # These arguments determine the actual remote bytes and authentication.
        assert set(kwargs) == {
            "repo_id",
            "repo_type",
            "revision",
            "filename",
            "local_dir",
        }
        assert kwargs["repo_id"] == "OpenSenseNova/test-assets"
        assert kwargs["repo_type"] == "dataset"
        assert kwargs["revision"] == REVISION
        assert kwargs["filename"].startswith(PREFIX + "/")
        name = kwargs["filename"][len(PREFIX) + 1 :]
        calls.append(name)
        destination = Path(kwargs["local_dir"]) / kwargs["filename"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(remote[name])
        return str(destination)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fetch)
    return {
        "args": {
            "repo_id": "OpenSenseNova/test-assets",
            "revision": REVISION,
            "prefix": PREFIX,
            "manifest_sha256": digest(remote["MANIFEST.sha256"]),
            "destination": tmp_path / "assets",
        },
        "remote": remote,
        "files": files,
        "calls": calls,
        "fetch": fetch,
    }


def test_downloads_only_manifest_files_and_does_not_persist_credentials(
    downloader, package, monkeypatch
):
    monkeypatch.setenv("HF_TOKEN", "hf-secret-value")
    monkeypatch.setenv("OPENAI_API_KEY", "model-secret-value")
    package["remote"]["unrelated.bin"] = b"unrelated"
    result = downloader.download_snapshot(**package["args"])

    assert result == package["args"]["destination"]
    for name, data in package["files"].items():
        assert (result / name).read_bytes() == data
    assert sorted(package["calls"]) == sorted(["MANIFEST.sha256", *package["files"]])
    assert not (result / "unrelated.bin").exists()
    marker = (result / ".download.json").read_text()
    assert REVISION in marker
    assert "secret-value" not in marker


def test_valid_cache_requires_no_network(downloader, package, monkeypatch):
    result = downloader.download_snapshot(**package["args"])

    def offline(**kwargs):
        pytest.fail("a verified cache must not require the network")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
    assert downloader.download_snapshot(**package["args"]) == result


@pytest.mark.parametrize("field,value", [("revision", "main"), ("manifest_sha256", "")])
def test_requires_pinned_revision_and_checksum(downloader, package, field, value):
    package["args"][field] = value
    with pytest.raises(ValueError):
        downloader.download_snapshot(**package["args"])
    assert not package["calls"]


def test_manifest_checksum_is_verified_before_payload_download(downloader, package):
    package["remote"]["MANIFEST.sha256"] += b"modified\n"
    with pytest.raises(ValueError, match="(?i)checksum|sha.?256"):
        downloader.download_snapshot(**package["args"])
    assert package["calls"] == ["MANIFEST.sha256"]
    assert not package["args"]["destination"].exists()


def test_payload_checksum_failure_leaves_no_partial_destination(downloader, package):
    package["remote"]["knowledge_packages.json"] = b"bad payload"
    with pytest.raises(ValueError, match="(?i)checksum|sha.?256"):
        downloader.download_snapshot(**package["args"])
    assert not package["args"]["destination"].exists()
    assert list(package["args"]["destination"].parent.iterdir()) == []


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/absolute",
        "C:/escape",
        "a/../../escape",
        "a\\escape",
        ".download.json",
        ".download.json/nested",
        "CON/example.py",
    ],
)
def test_rejects_unsafe_manifest_paths(downloader, package, name):
    content = f"{digest(b'bad')}  {name}\n".encode()
    package["remote"]["MANIFEST.sha256"] = content
    package["args"]["manifest_sha256"] = digest(content)
    with pytest.raises(ValueError, match="(?i)path|reserved"):
        downloader.download_snapshot(**package["args"])
    assert package["calls"] == ["MANIFEST.sha256"]
    assert not package["args"]["destination"].exists()


def test_rejects_duplicate_manifest_paths(downloader, package):
    content = package["remote"]["MANIFEST.sha256"] * 2
    package["remote"]["MANIFEST.sha256"] = content
    package["args"]["manifest_sha256"] = digest(content)
    with pytest.raises(ValueError, match="(?i)duplicate"):
        downloader.download_snapshot(**package["args"])


@pytest.mark.parametrize("mutation", ["same_size", "extra", "missing", "bad_marker"])
def test_rechecks_cache_and_preserves_modified_files(downloader, package, mutation):
    result = downloader.download_snapshot(**package["args"])
    payload = result / "knowledge_packages.json"
    if mutation == "same_size":
        payload.write_bytes(b"x" * payload.stat().st_size)
    elif mutation == "extra":
        (result / "unlisted.txt").write_text("user file")
    elif mutation == "missing":
        payload.unlink()
    else:
        (result / ".download.json").write_text("{}")
    before = {
        str(p.relative_to(result)): p.read_bytes()
        for p in result.rglob("*")
        if p.is_file()
    }
    with pytest.raises(ValueError, match="(?i)cache|existing|destination"):
        downloader.download_snapshot(**package["args"])
    after = {
        str(p.relative_to(result)): p.read_bytes()
        for p in result.rglob("*")
        if p.is_file()
    }
    assert after == before


def test_does_not_reuse_cache_for_a_different_source(downloader, package):
    downloader.download_snapshot(**package["args"])
    package["args"]["revision"] = "a" * 40
    with pytest.raises(ValueError, match="(?i)cache|source|destination"):
        downloader.download_snapshot(**package["args"])


def test_preserves_existing_directory_without_marker(downloader, package):
    destination = package["args"]["destination"]
    destination.mkdir()
    (destination / "mine.txt").write_text("my data")
    with pytest.raises(ValueError, match="(?i)cache|existing|destination"):
        downloader.download_snapshot(**package["args"])
    assert (destination / "mine.txt").read_text() == "my data"
    assert not package["calls"]


def test_interrupted_download_can_retry_and_preserves_unrelated_temp_data(
    downloader, package, monkeypatch
):
    sibling = package["args"]["destination"].parent / ".assets-other-run"
    sibling.mkdir()
    (sibling / "mine").write_text("keep")

    def interrupted(**kwargs):
        if kwargs["filename"].endswith("knowledge_packages.json"):
            raise RuntimeError("failed with hf-secret-value")
        return package["fetch"](**kwargs)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", interrupted)
    with pytest.raises(RuntimeError) as error:
        downloader.download_snapshot(**package["args"])
    rendered = "".join(traceback.format_exception(error.value))
    assert "hf-secret-value" not in rendered
    assert list(sibling.parent.iterdir()) == [sibling]
    assert (sibling / "mine").read_text() == "keep"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", package["fetch"])
    result = downloader.download_snapshot(**package["args"])
    assert (result / "knowledge_packages.json").read_bytes() == b'{"packages": []}'


def test_does_not_replace_a_destination_created_during_download(
    downloader, package, monkeypatch
):
    destination = package["args"]["destination"]

    def competing_download(**kwargs):
        if not destination.exists():
            destination.mkdir()
            (destination / "mine").write_text("keep")
        return package["fetch"](**kwargs)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", competing_download)
    with pytest.raises(ValueError, match="(?i)cache|existing|destination"):
        downloader.download_snapshot(**package["args"])
    assert (destination / "mine").read_text() == "keep"
    assert list(destination.parent.iterdir()) == [destination]


def test_cache_marker_cannot_escape_destination(downloader, package):
    result = downloader.download_snapshot(**package["args"])
    marker_path = result / ".download.json"
    marker = json.loads(marker_path.read_text())
    marker["files"]["../outside"] = {"size": 0, "sha256": digest(b"")}
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="(?i)cache|path"):
        downloader.download_snapshot(**package["args"])


def test_cache_manifest_cannot_be_rewritten_along_with_its_marker(downloader, package):
    result = downloader.download_snapshot(**package["args"])
    manifest = result / "MANIFEST.sha256"
    modified = manifest.read_bytes().replace(
        b"knowledge_packages", b"attacker_packages"
    )
    manifest.write_bytes(modified)
    marker_path = result / ".download.json"
    marker = json.loads(marker_path.read_text())
    marker["files"]["MANIFEST.sha256"]["sha256"] = digest(modified)
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="(?i)cache|checksum"):
        downloader.download_snapshot(**package["args"])


def test_rejects_symlink_in_cached_payload(downloader, package):
    result = downloader.download_snapshot(**package["args"])
    payload = result / "knowledge_packages.json"
    outside = result.parent / "outside.json"
    outside.write_bytes(payload.read_bytes())
    payload.unlink()
    try:
        payload.symlink_to(outside)
    except OSError:
        pytest.skip("This Windows account cannot create symlinks")
    with pytest.raises(ValueError, match="(?i)cache|link"):
        downloader.download_snapshot(**package["args"])
    assert outside.read_bytes() == b'{"packages": []}'


def test_reuses_verified_cache_created_by_a_concurrent_download(
    downloader, package, monkeypatch
):
    def concurrent_download(**kwargs):
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", package["fetch"])
        downloader.download_snapshot(**package["args"])
        return package["fetch"](**kwargs)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", concurrent_download)
    result = downloader.download_snapshot(**package["args"])
    assert (result / "knowledge_packages.json").read_bytes() == b'{"packages": []}'
    assert list(result.parent.iterdir()) == [result]
