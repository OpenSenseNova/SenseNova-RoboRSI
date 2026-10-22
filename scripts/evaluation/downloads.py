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

"""Download benchmark packages and model assets, without loading inference models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path, PureWindowsPath
from uuid import uuid4

MARKER = ".download.json"
OPENPI_TOKENIZER = "gs://big_vision/paligemma_tokenizer.model"


def _openpi_cache_path(source: str) -> Path:
    if not source.startswith("gs://"):
        return Path(source).expanduser().resolve()
    if not re.fullmatch(r"gs://[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+", source) or any(
        part in {"", ".", ".."} for part in source[5:].split("/")
    ):
        raise ValueError("Invalid OpenPI GCS source")
    if not os.environ.get("OPENPI_DATA_HOME"):
        raise ValueError("Set OPENPI_DATA_HOME to the prepared OpenPI download cache")
    return Path(os.environ["OPENPI_DATA_HOME"]).expanduser() / source[5:]


def _validate_openpi_checkpoint(checkpoint: Path) -> Path:
    if not (checkpoint / "assets").is_dir() or not (
        (checkpoint / "params").is_dir() or (checkpoint / "model.safetensors").is_file()
    ):
        raise ValueError(
            "OpenPI checkpoint is incomplete; use evaluate_model.py --prepare-only "
            "to download assets and params/model.safetensors first"
        )
    return checkpoint


def prepared_openpi_checkpoint(source: str) -> Path:
    """Resolve local assets for serving; never contact a download service."""
    checkpoint = _validate_openpi_checkpoint(_openpi_cache_path(source))
    tokenizer = _openpi_cache_path(OPENPI_TOKENIZER)
    if not tokenizer.is_file() or not tokenizer.stat().st_size:
        raise ValueError(
            "OpenPI tokenizer is missing; use evaluate_model.py --prepare-only to download it"
        )
    return checkpoint


def download_openpi(checkpoint: str) -> Path:
    """Run only inside the isolated OpenPI environment; no policy/GPU imports."""
    from openpi.shared import download

    path = _validate_openpi_checkpoint(
        Path(download.maybe_download(checkpoint, gs={"token": "anon"}))
    )
    tokenizer = Path(download.maybe_download(OPENPI_TOKENIZER, gs={"token": "anon"}))
    if not tokenizer.is_file() or not tokenizer.stat().st_size:
        raise ValueError("OpenPI tokenizer download is incomplete")
    return path


def prepare_openpi(image: str, cache_dir: Path, checkpoint: str) -> None:
    """Download weights/tokenizer in a disposable CPU container, preserving cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    mounts = ["--mount", f"type=bind,src={cache_dir.resolve()},dst=/cache"]
    if not checkpoint.startswith("gs://"):
        mounts += ["--mount", f"type=bind,src={checkpoint},dst=/checkpoint,readonly"]
        checkpoint = "/checkpoint"
    container = "roborsi-vla-download-" + uuid4().hex[:12]
    print(f"Preparing VLA checkpoint: {checkpoint}", file=sys.stderr)
    try:
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--platform",
                "linux/amd64",
                "--name",
                container,
                *mounts,
                "--entrypoint",
                "/opt/openpi/.venv/bin/python",
                image,
                "/app/scripts/evaluation/downloads.py",
                "openpi",
                "--checkpoint",
                checkpoint,
            ],
            text=True,
            check=True,
        )
    finally:
        try:
            subprocess.run(
                ["docker", "rm", "--force", container],
                text=True,
                check=False,
                capture_output=True,
            )
        except OSError:
            print(f"Could not stop download container: {container}", file=sys.stderr)


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="kind", required=True)
    openpi = subcommands.add_parser(
        "openpi", help="Download OpenPI weights and tokenizer"
    )
    openpi.add_argument("--checkpoint", required=True)
    args = parser.parse_args(argv)
    checkpoint = download_openpi(args.checkpoint)
    print(json.dumps({"checkpoint": str(checkpoint)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
