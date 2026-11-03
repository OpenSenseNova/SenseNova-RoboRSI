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

"""Content identities and immutable, portable exploration snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from roborsi.evaluation.result import write_json_atomic


def digest(value: Any) -> str:
    """Hash canonical finite JSON."""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_digest(path: Path) -> str:
    """Hash an input without loading large scene files into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON document."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    """Atomically save a document in its owning directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, value)


def inventory(root: Path) -> dict[str, str]:
    """Reject symlinks and inventory regular files in a resource tree."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"resource must be a real directory: {root}")
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError(f"unsupported resource: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = file_digest(path)
    return result


def verify_snapshot(root: Path) -> str:
    """Verify every frozen file and return the snapshot's identity."""
    manifest = read_json(root / "snapshot.json")
    files = inventory(root / "workspace")
    if manifest != {"schema_version": 1, "files": files, "sha256": digest(files)}:
        raise ValueError("exploration snapshot changed")
    if not (root / "workspace/EXPLORATION.md").read_text().strip():
        raise ValueError("exploration must include nonempty EXPLORATION.md")
    return manifest["sha256"]


def freeze_snapshot(workspace: Path, destination: Path) -> str:
    """Copy completed exploration once; never share writable episode workspaces."""
    inventory(workspace)
    if not (workspace / "EXPLORATION.md").read_text().strip():
        raise ValueError("exploration must include nonempty EXPLORATION.md")
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        shutil.copytree(
            workspace,
            temporary / "workspace",
            ignore=shutil.ignore_patterns(
                "__pycache__",
                "*.pyc",
                "*.pyo",
                ".git",
                ".venv",
                ".pytest_cache",
                ".ruff_cache",
            ),
        )
        for path in sorted(
            (temporary / "workspace").rglob("*"),
            key=lambda p: len(p.parts),
            reverse=True,
        ):
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
        files = inventory(temporary / "workspace")
        write_json(
            temporary / "snapshot.json",
            {"schema_version": 1, "files": files, "sha256": digest(files)},
        )
        identity = verify_snapshot(temporary)
        os.rename(temporary, destination)
        return identity
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
