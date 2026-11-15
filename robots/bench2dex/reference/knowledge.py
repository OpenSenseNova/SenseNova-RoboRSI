# Copyright 2026 The Bench2Dex Evaluation Contributors.
# Licensing of this copied reference code is pending; see LICENSE_NOTICE.md.

"""Content-addressed task knowledge. Only sealed payloads enter test namespaces."""

import json
import shutil
from pathlib import Path

from ..icl import MODEL
from ..icl_inventory import content_hashes
from .core import write_json

SETTING = "task_icl_v3"


def file_hash(path):
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tree_hashes(root):
    """Reject links/devices/sockets; never follow model-controlled paths."""
    return content_hashes(root)


def contract(config):
    return {
        k: config[k]
        for k in (
            "model",
            "codex_version",
            "reasoning_effort",
            "observation_profile",
            "deployment_profile",
            "upstream_commit",
            "isaaclab_commit",
        )
    }


def portable_config(config):
    """Replay compares package content, not differing host snapshot paths."""
    if config.get("evaluation_setting") != SETTING:
        return config
    return {
        **config,
        "task_packages": {
            k: {"sha256": v["sha256"]} for k, v in config["task_packages"].items()
        },
    }


def validate_config(config):
    if (
        config.get("evaluation_setting") != SETTING
        or config["reasoning_effort"] != "max"
        or config["generalization_profile"] != "inv_cov"
        or config["generalization_split"] != "unseen"
    ):
        raise ValueError("ICL v3 requires Astra/max, Full/unseen and explicit setting")
    refs = config.get("task_packages")
    if not isinstance(refs, dict) or not refs:
        raise ValueError("ICL needs frozen task packages")
    for task, ref in refs.items():
        if (
            not task.isdigit()
            or str(int(task)) != task
            or not isinstance(ref, dict)
            or set(ref) != {"path", "sha256"}
            or not isinstance(ref["path"], str)
            or not ref["path"]
            or not isinstance(ref["sha256"], str)
            or len(ref["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in ref["sha256"])
        ):
            raise ValueError("Invalid task package reference")


def verify_package(path, expected_hash, task_id, config):
    if path.is_symlink():
        raise ValueError("Package directory may not be a symlink")
    # Scan the entire container, not just its declared files.
    container = tree_hashes(path)
    if container.get("manifest.json") != expected_hash:
        raise ValueError("Task package manifest hash mismatch")
    manifest = json.loads((path / "manifest.json").read_text())
    if (
        manifest.get("setting") != SETTING
        or manifest.get("task_id") != task_id
        or manifest.get("contract") != contract(config)
        or manifest.get("preparation_model") != MODEL
        or manifest.get("ready") is not True
    ):
        raise ValueError("Task package identity/setting mismatch")
    payload = {
        name.removeprefix("payload/"): value
        for name, value in container.items()
        if name.startswith("payload/")
    }
    if not (path / "payload").is_dir() or payload != manifest.get("files"):
        raise ValueError("Task knowledge changed after freezing")
    if set(container) != {
        "manifest.json",
        *("payload/" + p for p in manifest["files"]),
    }:
        raise ValueError("Undeclared package files")
    if not all(manifest["files"].get(k) for k in ("TASK.md", "SOURCES.md")):
        raise ValueError("Missing task knowledge entrypoint or source record")
    return manifest


def snapshot_packages(config, task_ids, destination):
    """Host snapshot before any tests, not a mutable cross-episode workspace."""
    if config.get("evaluation_setting") != SETTING:
        return config
    refs = {}
    for task_id in task_ids:
        ref = config["task_packages"].get(str(task_id))
        if ref is None:
            raise ValueError(f"Missing frozen knowledge for task {task_id}")
        source = Path(ref["path"])
        verify_package(source, ref["sha256"], task_id, config)
        target = destination / f"task_{task_id:02d}"
        shutil.copytree(source, target)
        verify_package(target, ref["sha256"], task_id, config)
        refs[str(task_id)] = {"path": str(target.resolve()), "sha256": ref["sha256"]}
    return {**config, "task_packages": refs}


def attach_package(package, expected_hash, task_id, config, inputs, private):
    manifest = verify_package(package, expected_hash, task_id, config)
    target = inputs / "task_knowledge"
    shutil.copytree(package / "payload", target)
    if tree_hashes(target) != manifest["files"]:
        raise ValueError("Task knowledge copy differs")
    write_json(
        private / "task_knowledge.json", {"sha256": expected_hash, "manifest": manifest}
    )
    return manifest
