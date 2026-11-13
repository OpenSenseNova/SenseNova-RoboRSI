# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Offline contracts for fixed package sources and transactional caching."""

from __future__ import annotations

import hashlib
import io
import json
import stat
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from roborsi.components.icl.package import ContextPackage
from roborsi.components.icl.sources import (
    HfPackageSource,
    HttpArchiveSource,
    LocalPackageSource,
    fetch_package,
)


@pytest.fixture
def package_tree(tmp_path):
    root = tmp_path / "source"
    (root / "payload/notes").mkdir(parents=True)
    content = b"reference\n"
    (root / "payload/notes/task.md").write_bytes(content)
    document = {
        "schema_version": 1,
        "package_type": "rpent.icl.context",
        "package_id": "reference",
        "files": {"notes/task.md": hashlib.sha256(content).hexdigest()},
        "memory": [{"title": "Task", "path": "notes/task.md"}],
        "metadata": {"custom": "preserved"},
    }
    (root / "manifest.json").write_text(json.dumps(document), encoding="utf-8")
    return root


def _archive(root, extra=()):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.writestr(path.relative_to(root).as_posix(), path.read_bytes())
        for name, value in extra:
            archive.writestr(name, value)
    return stream.getvalue()


def _http_source(content, **kwargs):
    return HttpArchiveSource(
        "https://packages.example/reference.zip",
        hashlib.sha256(content).hexdigest(),
        **kwargs,
    )


def _mock_http(monkeypatch, handler):
    real_client = httpx.Client
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: real_client(transport=transport, **kwargs)
    )


def _assert_unpublished(cache):
    assert not list(cache.rglob("receipt.json"))
    assert not list(cache.rglob("manifest.json"))


def test_local_cache_is_independent_and_offline_revalidates(package_tree, tmp_path):
    source = LocalPackageSource(package_tree)
    cache = tmp_path / "cache"
    package = fetch_package(source, cache)
    assert package.root.parent.parent == cache
    assert set(package.root.iterdir()) == {
        package.root / "manifest.json",
        package.root / "payload",
    }
    assert package.memory[0][1].read_text() == "reference\n"
    assert package.metadata == {"custom": "preserved"}
    receipt = json.loads((package.root.parent / "receipt.json").read_text())
    assert receipt["manifest_sha256"] == package.manifest_sha256
    assert fetch_package(source, cache, offline=True).root == package.root
    (package.root / "payload/notes/task.md").write_text("tampered")
    with pytest.raises(ValueError):
        fetch_package(source, cache, offline=True)
    assert (package_tree / "payload/notes/task.md").read_text() == "reference\n"


def test_local_manifest_change_cannot_reuse_previous_offline_version(
    package_tree, tmp_path
):
    source = LocalPackageSource(package_tree)
    cache = tmp_path / "cache"
    first = fetch_package(source, cache)
    manifest = package_tree / "manifest.json"
    document = json.loads(manifest.read_text())
    document["metadata"]["custom"] = "new"
    manifest.write_text(json.dumps(document))
    with pytest.raises(FileNotFoundError):
        fetch_package(source, cache, offline=True)
    second = fetch_package(source, cache)
    assert first.manifest_sha256 != second.manifest_sha256
    assert first.root != second.root


def test_local_source_has_same_byte_bound_as_downloaded_payload(package_tree, tmp_path):
    with pytest.raises(ValueError, match="limit"):
        fetch_package(
            LocalPackageSource(package_tree), tmp_path / "cache", max_unpacked_bytes=1
        )
    _assert_unpublished(tmp_path / "cache")


def test_http_hit_is_verified_without_network_and_receipt_omits_credentials(
    package_tree, tmp_path, monkeypatch
):
    content = _archive(package_tree)
    source = HttpArchiveSource(
        "https://packages.example/reference.zip?token=TOP_SECRET",
        hashlib.sha256(content).hexdigest(),
    )
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, content=content)

    _mock_http(monkeypatch, handler)
    cache = tmp_path / "cache"
    package = fetch_package(source, cache)
    assert fetch_package(source, cache).root == package.root
    assert fetch_package(source, cache, offline=True).root == package.root
    assert len(calls) == 1
    receipt = (package.root.parent / "receipt.json").read_text()
    assert "TOP_SECRET" not in receipt
    assert "token=" not in receipt
    assert package.manifest_sha256 in receipt


@pytest.mark.parametrize(
    "damage", ["receipt", "manifest", "payload", "receipt_identity"]
)
def test_cache_corruption_is_not_repaired_or_bypassed(
    package_tree, tmp_path, monkeypatch, damage
):
    content = _archive(package_tree)
    calls = []
    _mock_http(
        monkeypatch,
        lambda request: (calls.append(request), httpx.Response(200, content=content))[
            1
        ],
    )
    source = _http_source(content)
    cache = tmp_path / "cache"
    package = fetch_package(source, cache)
    if damage == "receipt":
        (package.root.parent / "receipt.json").write_text("{")
    elif damage == "receipt_identity":
        receipt_path = package.root.parent / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["source"] = {"kind": "other"}
        receipt_path.write_text(json.dumps(receipt))
    elif damage == "manifest":
        manifest = package.root / "manifest.json"
        manifest.write_text(manifest.read_text() + " ")
    else:
        (package.root / "payload/extra.txt").write_text("undeclared")
    for offline in (False, True):
        with pytest.raises(ValueError):
            fetch_package(source, cache, offline=offline)
    assert len(calls) == 1


def test_offline_missing_source_never_borrows_another_cache(
    package_tree, tmp_path, monkeypatch
):
    content = _archive(package_tree)
    _mock_http(monkeypatch, lambda request: httpx.Response(200, content=content))
    cache = tmp_path / "cache"
    fetch_package(_http_source(content), cache)
    other = HttpArchiveSource(
        "https://elsewhere.example/same.zip", hashlib.sha256(content).hexdigest()
    )
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: pytest.fail("offline accessed network")
    )
    with pytest.raises(FileNotFoundError):
        fetch_package(other, cache, offline=True)


@pytest.mark.parametrize("field", ["archive", "manifest"])
def test_archive_and_manifest_hashes_are_independent(
    package_tree, tmp_path, monkeypatch, field
):
    content = _archive(package_tree)
    _mock_http(monkeypatch, lambda request: httpx.Response(200, content=content))
    source = (
        HttpArchiveSource("https://packages.example/data.zip", "0" * 64)
        if field == "archive"
        else _http_source(content, manifest_sha256=hashlib.sha256(content).hexdigest())
    )
    with pytest.raises(ValueError, match="SHA-256|hash"):
        fetch_package(source, tmp_path / "cache")
    _assert_unpublished(tmp_path / "cache")


def test_matching_explicit_manifest_hash_is_accepted(
    package_tree, tmp_path, monkeypatch
):
    content = _archive(package_tree)
    _mock_http(monkeypatch, lambda request: httpx.Response(200, content=content))
    expected = hashlib.sha256((package_tree / "manifest.json").read_bytes()).hexdigest()
    package = fetch_package(
        _http_source(content, manifest_sha256=expected), tmp_path / "cache"
    )
    assert package.manifest_sha256 == expected


@pytest.mark.parametrize("which", ["download", "unpacked"])
def test_real_byte_limits_ignore_false_content_length(
    package_tree, tmp_path, monkeypatch, which
):
    content = _archive(package_tree)
    _mock_http(
        monkeypatch,
        lambda request: httpx.Response(
            200, content=content, headers={"content-length": "1"}
        ),
    )
    kwargs = {"max_download_bytes" if which == "download" else "max_unpacked_bytes": 10}
    with pytest.raises(ValueError, match="limit"):
        fetch_package(_http_source(content), tmp_path / "cache", **kwargs)
    _assert_unpublished(tmp_path / "cache")


def test_interrupted_stream_does_not_publish_and_sanitizes_error(
    package_tree, tmp_path, monkeypatch
):
    content = _archive(package_tree)

    class Interrupted(httpx.SyncByteStream):
        def __iter__(self):
            yield content[:8]
            raise httpx.ReadError("Authorization: Bearer TOP_SECRET")

    _mock_http(monkeypatch, lambda request: httpx.Response(200, stream=Interrupted()))
    with pytest.raises(RuntimeError) as caught:
        fetch_package(_http_source(content), tmp_path / "cache")
    assert "ReadError" in str(caught.value)
    assert "TOP_SECRET" not in str(caught.value)
    assert caught.value.__suppress_context__
    _assert_unpublished(tmp_path / "cache")


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/absolute",
        "C:/drive",
        "payload/../escape",
        "payload\\escape",
        "payload/a/./b",
        "payload//double",
        "payload/NUL",
        "payload/name:stream",
        "payload/trailing.",
        "payload/trailing ",
        "outer/manifest.json",
    ],
)
def test_zip_rejects_unsafe_or_unexpected_paths(
    package_tree, tmp_path, monkeypatch, name
):
    content = _archive(package_tree, [(name, b"bad")])
    _mock_http(monkeypatch, lambda request: httpx.Response(200, content=content))
    with pytest.raises(ValueError):
        fetch_package(_http_source(content), tmp_path / "cache")
    _assert_unpublished(tmp_path / "cache")
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize(
    "mode", [stat.S_IFLNK | 0o777, stat.S_IFIFO | 0o644, stat.S_IFCHR | 0o644]
)
def test_zip_rejects_links_and_special_files(package_tree, tmp_path, monkeypatch, mode):
    entry = zipfile.ZipInfo("payload/unsafe")
    entry.create_system = 3
    entry.external_attr = mode << 16
    content = _archive(package_tree, [(entry, b"target")])
    _mock_http(monkeypatch, lambda request: httpx.Response(200, content=content))
    with pytest.raises(ValueError):
        fetch_package(_http_source(content), tmp_path / "cache")
    _assert_unpublished(tmp_path / "cache")


@pytest.mark.parametrize("name", ["payload/notes/task.md", "payload/NOTES/task.md"])
def test_zip_rejects_duplicate_and_case_conflicting_entries(
    package_tree, tmp_path, monkeypatch, name
):
    if name == "payload/notes/task.md":
        with pytest.warns(UserWarning, match="Duplicate name"):
            content = _archive(package_tree, [(name, b"replacement")])
    else:
        content = _archive(package_tree, [(name, b"replacement")])
    _mock_http(monkeypatch, lambda request: httpx.Response(200, content=content))
    with pytest.raises(ValueError):
        fetch_package(_http_source(content), tmp_path / "cache")
    _assert_unpublished(tmp_path / "cache")


def test_two_concurrent_publishers_reuse_one_complete_cache(
    package_tree, tmp_path, monkeypatch
):
    content = _archive(package_tree)
    _mock_http(monkeypatch, lambda request: httpx.Response(200, content=content))
    source = _http_source(content)
    cache = tmp_path / "cache"
    barrier = threading.Barrier(2, timeout=10)
    original = Path.rename

    def synchronized_rename(path, target):
        if Path(target).parent == cache and not Path(target).name.startswith("."):
            barrier.wait()
        return original(path, target)

    monkeypatch.setattr(Path, "rename", synchronized_rename)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: fetch_package(source, cache), range(2)))
    assert results[0].root == results[1].root
    assert results[0].manifest_sha256 == results[1].manifest_sha256
    assert len(list(cache.iterdir())) == 1
    assert (
        ContextPackage.load(results[0].root).memory[0][1].read_text() == "reference\n"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_download_bytes": 0},
        {"max_unpacked_bytes": -1},
        {"max_download_bytes": True},
        {"max_unpacked_bytes": 1.5},
    ],
)
def test_invalid_limits_fail_before_network(tmp_path, monkeypatch, kwargs):
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: pytest.fail("invalid input accessed network")
    )
    with pytest.raises(ValueError):
        fetch_package(
            HttpArchiveSource("https://example.com/a.zip", "0" * 64),
            tmp_path / "cache",
            **kwargs,
        )
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize(
    "source_factory",
    [
        lambda: HttpArchiveSource("file:///etc/passwd", "0" * 64),
        lambda: HttpArchiveSource("https://user:secret@example.com/a", "0" * 64),
        lambda: HttpArchiveSource("https://example.com/a", "A" * 64),
        lambda: HttpArchiveSource("https://example.com/a", "0" * 63),
        lambda: HttpArchiveSource("https://example.com/a", "0" * 64, "bad"),
        lambda: HfPackageSource("owner/repo", "main", "package"),
        lambda: HfPackageSource("owner/repo", "a" * 39, "package"),
        lambda: HfPackageSource("owner/repo", "a" * 40, "../package"),
        lambda: HfPackageSource("owner/repo", "a" * 40, "/package"),
        lambda: HfPackageSource("owner/repo", "a" * 40, "C:/package"),
        lambda: HfPackageSource("owner/repo", "a" * 40, "package", "unsupported"),
    ],
)
def test_source_validation_fails_before_network(tmp_path, monkeypatch, source_factory):
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: pytest.fail("invalid input accessed network")
    )
    with pytest.raises(ValueError):
        fetch_package(source_factory(), tmp_path / "cache")
    assert not (tmp_path / "cache").exists()


def _mock_hf(monkeypatch, root, *, returned_revision="a" * 40, size_override=None):
    import huggingface_hub
    import huggingface_hub.utils

    downloads = []
    metadata_calls = []
    files = {
        "task/manifest.json": (root / "manifest.json").read_bytes(),
        "task/payload/notes/task.md": (root / "payload/notes/task.md").read_bytes(),
        "task/.cache/huggingface/downloads": b"private downloader metadata",
        "task-other/manifest.json": b"other task",
        "unrelated.bin": b"large unrelated file",
    }

    class FakeApi:
        def repo_info(self, repo_id, *, revision, repo_type, files_metadata):
            assert (repo_id, revision, repo_type, files_metadata) == (
                "owner/repo",
                "a" * 40,
                "dataset",
                True,
            )
            metadata_calls.append(revision)
            return SimpleNamespace(
                sha=returned_revision,
                siblings=[
                    SimpleNamespace(
                        rfilename=name,
                        size=size_override if size_override is not None else len(data),
                    )
                    for name, data in files.items()
                ],
            )

    def url(repo_id, filename, *, revision, repo_type):
        assert (
            repo_id == "owner/repo" and revision == "a" * 40 and repo_type == "dataset"
        )
        return "https://hf.example/" + filename

    def handler(request):
        filename = request.url.path.removeprefix("/")
        downloads.append(filename)
        assert request.headers["authorization"] == "Bearer TOP_SECRET"
        return httpx.Response(
            200, content=files[filename], headers={"content-length": "1"}
        )

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "hf_hub_url", url)
    monkeypatch.setattr(
        huggingface_hub.utils,
        "build_hf_headers",
        lambda: {"authorization": "Bearer TOP_SECRET"},
    )
    _mock_http(monkeypatch, handler)
    return downloads, metadata_calls


def test_hf_verifies_revision_fetches_only_exact_subtree_and_supports_offline(
    package_tree, tmp_path, monkeypatch
):
    downloads, metadata_calls = _mock_hf(monkeypatch, package_tree)
    source = HfPackageSource("owner/repo", "a" * 40, "task")
    cache = tmp_path / "cache"
    package = fetch_package(source, cache)
    assert sorted(downloads) == ["task/manifest.json", "task/payload/notes/task.md"]
    assert metadata_calls == ["a" * 40]
    assert fetch_package(source, cache, offline=True).root == package.root
    assert metadata_calls == ["a" * 40]
    receipt = (package.root.parent / "receipt.json").read_text()
    assert "a" * 40 in receipt
    assert "TOP_SECRET" not in receipt
    assert "authorization" not in receipt
    assert "hf.example" not in receipt


def test_hf_rejects_wrong_resolved_revision_before_file_download(
    package_tree, tmp_path, monkeypatch
):
    downloads, _ = _mock_hf(monkeypatch, package_tree, returned_revision="b" * 40)
    with pytest.raises(ValueError, match="revision"):
        fetch_package(
            HfPackageSource("owner/repo", "a" * 40, "task"), tmp_path / "cache"
        )
    assert downloads == []
    _assert_unpublished(tmp_path / "cache")


def test_hf_limits_actual_bytes_even_if_metadata_lies(
    package_tree, tmp_path, monkeypatch
):
    _mock_hf(monkeypatch, package_tree, size_override=1)
    with pytest.raises(ValueError, match="limit"):
        fetch_package(
            HfPackageSource("owner/repo", "a" * 40, "task"),
            tmp_path / "cache",
            max_download_bytes=10,
        )
    _assert_unpublished(tmp_path / "cache")


def test_hf_offline_miss_does_not_query_metadata(tmp_path, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "HfApi", lambda: pytest.fail("offline queried HF")
    )
    with pytest.raises(FileNotFoundError):
        fetch_package(
            HfPackageSource("owner/repo", "a" * 40, "task"),
            tmp_path / "cache",
            offline=True,
        )


def test_cache_entry_symlink_is_rejected(package_tree, tmp_path):
    cache = tmp_path / "cache"
    source = LocalPackageSource(package_tree)
    package = fetch_package(source, cache)
    entry = package.root.parent
    relocated = tmp_path / "relocated"
    entry.rename(relocated)
    try:
        entry.symlink_to(relocated, target_is_directory=True)
    except OSError:
        pytest.skip("symbolic links are unavailable")
    with pytest.raises(ValueError):
        fetch_package(source, cache, offline=True)
    assert (relocated / "package/payload/notes/task.md").read_text() == "reference\n"


def test_hf_unpacked_limit_stops_stream_before_reading_next_chunk(
    package_tree, tmp_path, monkeypatch
):
    client_class = httpx.Client
    _mock_hf(monkeypatch, package_tree, size_override=1)

    class BoundedStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b"x" * (64 * 1024)
            raise AssertionError("read beyond unpacked byte limit")

    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, stream=BoundedStream())
    )
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: client_class(transport=transport, **kwargs)
    )
    with pytest.raises(ValueError, match="limit"):
        fetch_package(
            HfPackageSource("owner/repo", "a" * 40, "task"),
            tmp_path / "cache",
            max_unpacked_bytes=10,
        )
    _assert_unpublished(tmp_path / "cache")


def test_local_cache_cannot_be_created_inside_source(package_tree):
    with pytest.raises(ValueError):
        fetch_package(LocalPackageSource(package_tree), package_tree / "cache")
    assert {path.name for path in package_tree.iterdir()} == {
        "manifest.json",
        "payload",
    }
    assert ContextPackage.load(package_tree).memory[0][1].read_text() == "reference\n"


def test_cached_manifest_replaced_with_hardlink_is_rejected(package_tree, tmp_path):
    source = LocalPackageSource(package_tree)
    package = fetch_package(source, tmp_path / "cache")
    manifest = package.root / "manifest.json"
    other = tmp_path / "other-manifest.json"
    other.write_bytes(manifest.read_bytes())
    manifest.unlink()
    try:
        manifest.hardlink_to(other)
    except OSError:
        pytest.skip("hardlinks are unavailable")
    with pytest.raises(ValueError):
        fetch_package(source, tmp_path / "cache", offline=True)
