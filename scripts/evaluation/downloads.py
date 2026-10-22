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

"""Download a pinned Hugging Face dataset package using its checksum manifest."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import stat
import tempfile
from pathlib import Path, PureWindowsPath

MARKER = ".download.json"


def _safe_path(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or ":" in value
        or any(ord(char) < 32 for char in value)
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or any(part.endswith((".", " ")) for part in value.split("/"))
        or any(PureWindowsPath(part).is_reserved() for part in value.split("/"))
    ):
        raise ValueError("Unsafe path in dataset package metadata")
    return value


def _file_checksum(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def _manifest_files(path: Path, manifest_filename: str) -> dict[str, str]:
    entries = {}
    casefold_names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](.+)", line)
        if not match:
            raise ValueError("Invalid SHA-256 manifest line")
        checksum, filename = match.groups()
        filename = _safe_path(filename)
        if any(
            name.casefold() in {MARKER.casefold(), manifest_filename.casefold()}
            for name in (filename, *(p.as_posix() for p in Path(filename).parents))
        ):
            raise ValueError("Manifest contains a reserved path")
        if filename.casefold() in casefold_names:
            raise ValueError("Duplicate path in dataset manifest")
        casefold_names.add(filename.casefold())
        entries[filename] = checksum.lower()
    if not entries:
        raise ValueError("Dataset manifest contains no files")
    for filename in entries:
        for parent in Path(filename).parents:
            if parent.as_posix().casefold() in casefold_names:
                raise ValueError("Conflicting file and directory paths in manifest")
    return entries


def _validate_cache(destination: Path, source: dict[str, str]) -> Path:
    """Recheck every byte against the pinned manifest before reusing a cache."""
    try:
        if destination.is_symlink() or not destination.is_dir():
            raise ValueError("not a regular directory")
        actual_files = set()
        for path in destination.rglob("*"):
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode):
                raise ValueError("cache contains a link or special file")
            actual_files.add(path.relative_to(destination).as_posix())
        marker = json.loads((destination / MARKER).read_text(encoding="utf-8"))
        if marker.get("version") != 1 or marker.get("source") != source:
            raise ValueError("source metadata does not match")
        manifest_filename = source["manifest_filename"]
        manifest = destination / manifest_filename
        if _file_checksum(manifest) != source["manifest_sha256"]:
            raise ValueError("manifest checksum does not match")
        expected = _manifest_files(manifest, manifest_filename)
        expected[manifest_filename] = source["manifest_sha256"]
        files = marker["files"]
        if not isinstance(files, dict) or set(files) != set(expected):
            raise ValueError("cache file metadata does not match the manifest")
        if actual_files != {*expected, MARKER}:
            raise ValueError("cache has missing or unlisted files")
        for filename, checksum in expected.items():
            path = destination / filename
            record = files[filename]
            if (
                record["sha256"] != checksum
                or path.stat().st_size != record["size"]
                or _file_checksum(path) != checksum
            ):
                raise ValueError("payload checksum does not match")
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        raise ValueError(
            f"Existing destination is not a verified cache: {destination}. "
            "Choose a new destination or move the existing directory before retrying."
        ) from None
    return destination


def _fetch_file(*, source: dict[str, str], filename: str, download_root: Path) -> Path:
    # Use HF_TOKEN or the user's existing HF login. Model API keys are unrelated.
    from huggingface_hub import hf_hub_download

    remote_filename = f"{source['prefix']}/{filename}"
    try:
        downloaded = Path(
            hf_hub_download(
                repo_id=source["repo_id"],
                repo_type="dataset",
                revision=source["revision"],
                filename=remote_filename,
                local_dir=str(download_root),
            )
        )
    except Exception:
        # Upstream exception messages may contain request credentials or URLs.
        raise RuntimeError(
            "Dataset download failed. Check network access and Hugging Face "
            "authentication (HF_TOKEN or your existing login), then retry."
        ) from None
    if (
        downloaded != download_root / remote_filename
        or downloaded.is_symlink()
        or not downloaded.is_file()
        or not downloaded.resolve().is_relative_to(download_root.resolve())
    ):
        raise ValueError("Dataset download returned an unsafe file path")
    return downloaded


def download_snapshot(
    *,
    repo_id: str,
    revision: str,
    prefix: str,
    destination: Path,
    manifest_sha256: str,
    manifest_filename: str = "MANIFEST.sha256",
) -> Path:
    """Download only files named by a pinned SHA-256 manifest into destination.

    Existing destinations must be complete, verified caches from the same source.
    Downloads and staging live in a temporary sibling directory; errors preserve
    existing user data and clean up only this call's temporary directory.
    """
    if not re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?", repo_id):
        raise ValueError("Expected a Hugging Face dataset repository ID")
    if not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
        raise ValueError("Dataset revision must be a pinned 40-character commit SHA")
    if not re.fullmatch(r"[a-fA-F0-9]{64}", manifest_sha256):
        raise ValueError("Manifest SHA-256 must contain 64 hexadecimal characters")
    source = {
        "repo_id": repo_id,
        "revision": revision.lower(),
        "prefix": _safe_path(prefix),
        "manifest_filename": _safe_path(manifest_filename),
        "manifest_sha256": manifest_sha256.lower(),
    }
    if manifest_filename.casefold() == MARKER.casefold():
        raise ValueError("Manifest filename uses a reserved path")
    destination = Path(destination).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        return _validate_cache(destination, source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-download-", dir=destination.parent
    ) as temporary:
        temporary_root = Path(temporary)
        download_root = temporary_root / "hub"
        staging = temporary_root / "content"
        staging.mkdir()
        manifest = _fetch_file(
            source=source, filename=manifest_filename, download_root=download_root
        )
        if _file_checksum(manifest) != source["manifest_sha256"]:
            raise ValueError("Dataset manifest SHA-256 checksum mismatch")
        expected = _manifest_files(manifest, manifest_filename)
        expected[manifest_filename] = source["manifest_sha256"]
        records = {}
        for filename, checksum in expected.items():
            downloaded = (
                manifest
                if filename == manifest_filename
                else _fetch_file(
                    source=source, filename=filename, download_root=download_root
                )
            )
            if _file_checksum(downloaded) != checksum:
                raise ValueError("Dataset payload SHA-256 checksum mismatch")
            target = staging / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(downloaded, target)
            records[filename] = {"size": target.stat().st_size, "sha256": checksum}
        (staging / MARKER).write_text(
            json.dumps({"version": 1, "source": source, "files": records}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        if destination.exists() or destination.is_symlink():
            return _validate_cache(destination, source)
        try:
            staging.rename(destination)
        except OSError:
            # Another download may have completed while this one was staging.
            if destination.exists() or destination.is_symlink():
                return _validate_cache(destination, source)
            raise
    return destination
