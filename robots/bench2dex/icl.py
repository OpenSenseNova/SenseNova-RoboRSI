# Copyright 2026 The RoboRSI Contributors.
# Licensed under the Apache License, Version 2.0.

"""Historical ICL package identity, isolated from native Benchmark contracts."""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

from roborsi.evaluation.messages import Task

from .contracts import ROOT, TASKS, file_sha256

PROTOCOL = json.loads((ROOT / "icl_protocol.json").read_text(encoding="utf-8"))
MODEL = "gpt-6-astra"
SETTING = "bench2dex_task_icl_v3_embodied_cli_v3"
ICL_CATALOG = json.loads((ROOT / "icl_catalog.json").read_text(encoding="utf-8"))


def package_entry(task_id: str) -> dict:
    """Return the content-addressed package paired with one native task."""
    return next(
        row for row in ICL_CATALOG["packages"] if str(row["task_id"]) == task_id
    )


def verify_package(path: Path, task: Task) -> dict:
    """Verify every frozen resource and its instruction/robot identity."""
    from .reference.knowledge import verify_package as verify

    entry = package_entry(task.task_id)
    manifest = verify(path, entry["manifest_sha256"], int(task.task_id), PROTOCOL)
    if (
        manifest["robot_key"] != TASKS[task.task_id]["robot_key"]
        or manifest["instruction_sha256"]
        != hashlib.sha256(task.instruction.encode()).hexdigest()
    ):
        raise ValueError("ICL package instruction/robot mismatch")
    return manifest


def extract_verified(archive: Path, target: Path, row: dict) -> None:
    """Verify the release digest before extracting bounded regular files only."""
    if (
        archive.stat().st_size != row["archive_bytes"]
        or file_sha256(archive) != row["sha256"]
    ):
        raise ValueError("ICL archive differs from the release catalog")
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if (
            len(members) > 10000
            or sum(m.file_size for m in members) != row["package_bytes"]
        ):
            raise ValueError("unexpected expanded ICL size")
        seen = set()
        for member in members:
            name = member.filename
            parts = PurePosixPath(name).parts
            mode = member.external_attr >> 16
            if (
                not parts
                or name.startswith("/")
                or "\\" in name
                or ":" in name
                or any(p in {"", ".", ".."} for p in name.split("/"))
                or name in seen
                or member.is_dir()
                or stat.S_IFMT(mode) not in {0, stat.S_IFREG}
                or member.flag_bits & 1
                or not (
                    name == "manifest.json" or len(parts) > 1 and parts[0] == "payload"
                )
            ):
                raise ValueError("unsafe or unexpected ICL archive member")
            seen.add(name)
        if "manifest.json" not in seen:
            raise ValueError("missing ICL manifest")
        target.mkdir()
        for member in members:
            path = target / member.filename
            path.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, path.open("xb") as output:
                shutil.copyfileobj(source, output)
