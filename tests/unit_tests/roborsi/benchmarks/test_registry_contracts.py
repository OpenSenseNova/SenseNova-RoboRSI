# Copyright 2026 The RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The RoboRSI registry locks declared tasks and asset archives before startup."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from roborsi.benchmarks.registry import (
    BenchmarkRegistry,
    directory_sha256,
    snapshot_directory,
)


def _manifest(root: Path) -> tuple[Path, Path]:
    benchmark = root / "demo"
    asset = benchmark / "assets" / "tasks.bin"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"task fixture")
    document = {
        "schema_version": 1,
        "benchmark": "demo",
        "benchmark_version": "1.0",
        "benchmark_commit": "a" * 40,
        "image_repository": "example.org/demo",
        "image_digest": "sha256:" + "b" * 64,
        "assets_version": "v1",
        "assets": [
            {
                "name": "tasks",
                "version": "v1",
                "path": "assets/tasks.bin",
                "sha256": sha256(asset.read_bytes()).hexdigest(),
            }
        ],
        "adapter_version": "1.0",
        "adapter_command": ["python", "-m", "benchmark.adapter"],
        "service_port": 8765,
        "resources": {
            "gpus": 0,
            "cpus": 1,
            "memory_mb": 512,
            "startup_timeout_s": 30,
        },
        "capabilities": {
            "action_schema": {"type": "object"},
            "observation_channels": ["front"],
        },
        "evaluation": {
            "success_metric": "native_success",
            "score_metric": "native_reward",
            "aggregation": "case_mean",
            "error_denominator": "expected",
        },
        "tasks": [
            {
                "task_id": "place_block",
                "seeds": [0, 1],
                "native": {"suite": "demo_suite"},
                "instruction": "Place the block.",
                "max_steps": 100,
            }
        ],
    }
    path = benchmark / "manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path, asset


def test_registry_discovers_and_locks_task_without_native_imports(
    tmp_path: Path,
) -> None:
    path, asset = _manifest(tmp_path)
    registry = BenchmarkRegistry([tmp_path])

    assert registry.discover() == ("demo",)
    lock = registry.lock("demo")
    assert lock.image_reference == "example.org/demo@sha256:" + "b" * 64
    assert [task.seed for task in lock.list_tasks()] == [0, 1]
    task = lock.task("place_block", 1)
    assert task.identity.as_dict() == {
        "benchmark": "demo",
        "task_id": "place_block",
        "seed": 1,
    }
    assert task.native == {"suite": "demo_suite"}
    assert task.capabilities["observation_channels"] == ["front"]
    assert lock.task_message("place_block", 1).native == {"suite": "demo_suite"}
    assert lock.task_message("place_block", 1).max_steps == 100
    assert lock.provenance("agent-v2").docker_image_digest == "sha256:" + "b" * 64
    assert lock.provenance("agent-v2").agent_version == "agent-v2"

    detached = lock.as_dict()
    detached["tasks"].clear()
    assert lock.task("place_block", 0).identity.seed == 0
    with pytest.raises(KeyError, match="undeclared"):
        lock.task("place_block", 99)

    asset.write_bytes(b"modified")
    with pytest.raises(ValueError, match="asset digest mismatch"):
        lock.verify()

    asset.write_bytes(b"task fixture")
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest changed"):
        lock.verify()


def test_registry_rejects_missing_assets_and_bad_manifest(tmp_path: Path) -> None:
    path, asset = _manifest(tmp_path)
    registry = BenchmarkRegistry([tmp_path])
    asset.unlink()
    with pytest.raises(FileNotFoundError, match="asset is missing"):
        registry.lock("demo")

    asset.write_bytes(b"task fixture")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["image_digest"] = "latest"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="image_digest"):
        registry.lock("demo")


def test_registry_rejects_duplicate_tasks_and_asset_path_escape(tmp_path: Path) -> None:
    path, _ = _manifest(tmp_path)
    registry = BenchmarkRegistry([tmp_path])
    document = json.loads(path.read_text(encoding="utf-8"))
    document["tasks"].append(document["tasks"][0])
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate benchmark task_id"):
        registry.lock("demo")

    document["tasks"].pop()
    document["assets"][0]["path"] = "../../outside.bin"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes benchmark"):
        registry.lock("demo")


def test_registry_rejects_invalid_action_schema_and_nonstandard_json(
    tmp_path: Path,
) -> None:
    path, _ = _manifest(tmp_path)
    registry = BenchmarkRegistry([tmp_path])
    document = json.loads(path.read_text(encoding="utf-8"))
    document["capabilities"]["action_schema"] = {"type": "unknown-type"}
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid action_schema"):
        registry.lock("demo")

    document["capabilities"]["action_schema"] = {"type": "object"}
    document["resources"]["cpus"] = float("nan")
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid benchmark manifest JSON"):
        registry.lock("demo")


def test_directory_assets_are_content_locked_and_reject_symlinks(
    tmp_path: Path,
) -> None:
    manifest, _ = _manifest(tmp_path)
    tree = manifest.parent / "assets" / "models"
    (tree / "weights").mkdir(parents=True)
    (tree / "empty").mkdir()
    (tree / "weights" / "policy.bin").write_bytes(b"weights-v1")
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["assets"] = [
        {
            "name": "models",
            "version": "v1",
            "kind": "directory",
            "path": "assets/models",
            "sha256": directory_sha256(tree),
        }
    ]
    manifest.write_text(json.dumps(document), encoding="utf-8")
    lock = BenchmarkRegistry([tmp_path]).lock("demo")
    lock.verify()

    (tree / "weights" / "policy.bin").write_bytes(b"weights-v2")
    with pytest.raises(ValueError, match="asset digest mismatch"):
        lock.verify()
    (tree / "weights" / "policy.bin").write_bytes(b"weights-v1")
    (tree / "new-empty").mkdir()
    with pytest.raises(ValueError, match="asset digest mismatch"):
        lock.verify()
    (tree / "new-empty").rmdir()
    (tree / "weights" / "shortcut").symlink_to("policy.bin")
    with pytest.raises(ValueError, match="symlink"):
        lock.verify()
    (tree / "weights" / "shortcut").unlink()
    alias = manifest.parent / "asset-alias"
    alias.symlink_to(tree, target_is_directory=True)
    document["assets"][0]["path"] = "asset-alias"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="symlink"):
        BenchmarkRegistry([tmp_path]).lock("demo")


def test_directory_snapshot_is_copied_once_and_keeps_original_content(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.bin").write_bytes(b"weights-v1")
    snapshots = tmp_path / "snapshots"

    snapshot = snapshot_directory(source, snapshots)

    assert snapshot.name == directory_sha256(source)
    assert snapshot_directory(source, snapshots) == snapshot
    assert (snapshot / "model.bin").read_bytes() == b"weights-v1"
    source.joinpath("model.bin").write_bytes(b"weights-v2")
    assert (snapshot / "model.bin").read_bytes() == b"weights-v1"
    assert snapshot_directory(source, snapshots) != snapshot
    with pytest.raises(ValueError, match="inside the source"):
        snapshot_directory(source, source / "nested")
    alias = tmp_path / "source-alias"
    alias.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="source cannot be a symlink"):
        snapshot_directory(alias, snapshots)


def test_instruction_is_resolved_for_the_selected_seed(tmp_path):
    path, _ = _manifest(tmp_path)
    document = json.loads(path.read_text())
    instructions = {"0": "Pick the red object", "1": "Pick the green object"}
    document["tasks"][0]["instructions_by_seed"] = instructions
    path.write_text(json.dumps(document))
    lock = BenchmarkRegistry([tmp_path]).lock("demo")
    for seed in (0, 1):
        assert (
            lock.task_message("place_block", seed).instruction
            == instructions[str(seed)]
        )
    for invalid in (
        {"0": "Only one"},
        {**instructions, "2": "Not declared"},
        {"00": "Bad seed", "1": "One"},
    ):
        document["tasks"][0]["instructions_by_seed"] = invalid
        path.write_text(json.dumps(document))
        with pytest.raises(ValueError):
            BenchmarkRegistry([tmp_path]).lock("demo")


def test_native_parameters_are_resolved_for_each_seed(tmp_path):
    path, _ = _manifest(tmp_path)
    document = json.loads(path.read_text())
    values = {"0": {"layout_sha256": "a" * 64}, "1": {"layout_sha256": "b" * 64}}
    document["tasks"][0]["native_by_seed"] = values
    path.write_text(json.dumps(document))
    lock = BenchmarkRegistry([tmp_path]).lock("demo")
    assert lock.task_message("place_block", 1).native == values["1"]
    for invalid in ({"0": {}}, {**values, "2": {}}, {"00": {}, "1": {}}):
        document["tasks"][0]["native_by_seed"] = invalid
        path.write_text(json.dumps(document))
        with pytest.raises(ValueError):
            BenchmarkRegistry([tmp_path]).lock("demo")
