# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Build and validate task-scoped rich BEHAVIOR demonstration bundles.

The official LeRobot export stores several episodes in one parquet/video shard.
Mounting those shards directly would make it possible for an agent to inspect
another task.  This module creates a read-only task view containing only the
selected task's metadata, annotations, RGB shards, and convenient per-episode
JSONL records. Depth video shards are deliberately excluded from the public
icl bundle.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from robots.behavior_1k.official import EEF21, JOINT23

DATASET_REPO = "behavior-1k/2026-challenge-demos"
# Version 2 removes depth video paths from the public icl bundle.
RICH_SCHEMA_VERSION = 2

CAMERAS = {
    "head": "zed_link_camera_0",
    "left_wrist": "left_realsense_link_camera_0",
    "right_wrist": "right_realsense_link_camera_0",
}
# Public icl exposes RGB only. The downloaded source may also contain
# depth_linear videos, but their paths are never copied into the task bundle or
# accepted by the icl frame helper.
MODALITIES = ("rgb",)
SHA256 = re.compile(r"[0-9a-f]{64}")


def _validate_action_profile(profile):
    if profile not in (EEF21, JOINT23):
        raise ValueError(f"Unknown demonstration action_profile: {profile!r}")


def _pyarrow():
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - exercised in deployment env
        raise RuntimeError(
            "Rich BEHAVIOR icl requires the official data extra (pyarrow)"
        ) from exc
    return parquet


def _relative(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or ":" in value
        or PurePosixPath(value).is_absolute()
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError(f"Unsafe rich-material path: {value!r}")
    return value


def _is_depth_video(relative: str) -> bool:
    """Return whether a bundle path names a depth-linear video shard."""
    return relative.startswith("videos/") and "depth_linear" in relative


def _file(root: Path, relative: str) -> Path:
    relative = _relative(relative)
    path = root
    for part in PurePosixPath(relative).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Rich source file is missing or unsafe: {path}")
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Rich source file is missing or unsafe: {path}")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        # A task bundle may live on another filesystem.  Falling back to a
        # byte-for-byte copy preserves the same isolation semantics.
        shutil.copy2(source, target)


def _source_revision(source: Path, explicit: str | None) -> str:
    candidates = (
        source.parent / f"{source.name}.selection.json",
        source.parent / "selection.json",
        source / "selection.json",
    )
    detected = None
    for candidate in candidates:
        if not candidate.is_file() or candidate.is_symlink():
            continue
        try:
            value = json.loads(candidate.read_text(encoding="utf-8")).get("revision")
        except (OSError, ValueError, TypeError, AttributeError):
            continue
        if isinstance(value, str) and value:
            detected = value
            break
    if explicit:
        if detected and explicit != detected:
            raise ValueError(
                f"Configured rich_data_revision {explicit!r} differs from source manifest {detected!r}"
            )
        return explicit
    return detected or "local-unpinned"


def _task_id(source: Path, task_name: str) -> int:
    parquet = _pyarrow()
    path = _file(source, "meta/tasks.parquet")
    rows = parquet.read_table(path).to_pylist()
    matches = [
        row
        for row in rows
        if isinstance(row, dict)
        and (row.get("task") or row.get("task_name")) == task_name
    ]
    if (
        len(matches) != 1
        or type(matches[0].get("task_index")) is not int
        or matches[0]["task_index"] < 0
    ):
        raise ValueError(
            f"Task is missing or duplicated in rich dataset metadata: {task_name}"
        )
    return int(matches[0]["task_index"])


def _episode_metadata(source: Path, task_id: int) -> tuple[list[Path], list[dict]]:
    parquet = _pyarrow()
    directory = source / "meta" / "episodes" / f"chunk-{task_id:03d}"
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"Missing episode metadata for task id {task_id}")
    files = sorted(path for path in directory.glob("file-*.parquet") if path.is_file())
    if not files:
        raise ValueError(f"Missing episode metadata for task id {task_id}")
    rows: list[dict] = []
    for path in files:
        if path.is_symlink():
            raise ValueError(f"Symlink in rich source: {path}")
        rows.extend(parquet.read_table(path).to_pylist())
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Rich episode metadata rows must be objects")
    rows = [row for row in rows if row.get("task_index") == task_id]
    if not rows:
        raise ValueError(f"No episodes found for task id {task_id}")
    episode_ids = [row.get("episode_index") for row in rows]
    if any(type(value) is not int or value < 0 for value in episode_ids) or len(
        set(episode_ids)
    ) != len(episode_ids):
        raise ValueError(
            f"Rich episode metadata has duplicate or invalid episode_index for task {task_id}"
        )
    rows.sort(
        key=lambda row: (
            row.get("demo_index_within_task", -1),
            row.get("episode_index", -1),
        )
    )
    return files, rows


def _episode_selection(
    cfg: dict, rows: list[dict], task_id: int
) -> list[tuple[dict, dict]]:
    by_instance: dict[int, dict] = {}
    for row in rows:
        instance = row.get("task_instance_id")
        if type(instance) is not int or instance < 0:
            raise ValueError("Rich episode metadata has an invalid task_instance_id")
        if instance in by_instance:
            raise ValueError(f"Duplicate rich task instance metadata: {instance}")
        by_instance[instance] = row
    selected: list[tuple[dict, dict]] = []
    seen: set[int] = set()
    for episode in cfg.get("train_episodes", []):
        if not isinstance(episode, dict):
            raise ValueError("Rich training episodes must be objects")
        index = episode.get("instance_index")
        if type(index) is not int or index < 0:
            raise ValueError(
                "Rich training instance_index must be a nonnegative integer"
            )
        row = by_instance.get(index)
        if row is None:
            raise ValueError(
                f"Rich source has no training demonstration for task id {task_id}, "
                f"instance {index}"
            )
        row_tasks = row.get("tasks", [])
        if row_tasks and cfg.get("task_name") not in row_tasks:
            raise ValueError("Rich episode metadata task differs from configuration")
        if index in seen:
            raise ValueError(f"Duplicate rich training instance: {index}")
        if episode.get("mode") != "train":
            raise ValueError("Rich material episodes must use mode=train")
        if episode.get("task_name") != cfg.get("task_name"):
            raise ValueError("Rich material episode task differs from configuration")
        if not isinstance(episode.get("id"), str) or not re.fullmatch(
            r"[A-Za-z0-9_-]+", episode["id"]
        ):
            raise ValueError("Rich material episode id is invalid")
        seen.add(index)
        selected.append((episode, row))
    if not selected:
        raise ValueError("At least one rich training episode is required")
    return selected


def _video_path(row: dict, modality: str, camera: str) -> str:
    source_camera = CAMERAS[camera]
    base = f"videos/observation.{modality}.{source_camera}"
    chunk = row.get(base + "/chunk_index")
    file_index = row.get(base + "/file_index")
    if (
        type(chunk) is not int
        or type(file_index) is not int
        or chunk < 0
        or file_index < 0
    ):
        raise ValueError(f"Episode is missing {modality}/{camera} video metadata")
    return f"{base}/chunk-{chunk:03d}/file-{file_index:03d}.mp4"


def _all_owners(source: Path, metadata_files: list[Path]) -> dict[str, set[int]]:
    """Map every physical data/video shard to the task ids that reference it."""
    parquet = _pyarrow()
    # Do not assume that the task chunk is the only metadata available.  A
    # future dataset layout may pack multiple task ids into one physical
    # shard; scanning every local episode index lets us reject that safely.
    all_metadata = sorted(
        path
        for path in (source / "meta" / "episodes").glob("chunk-*/file-*.parquet")
        if path.is_file() and not path.is_symlink()
    )
    if not all_metadata:
        all_metadata = metadata_files
    owners: dict[str, set[int]] = defaultdict(set)
    for metadata in all_metadata:
        for row in parquet.read_table(metadata).to_pylist():
            if not isinstance(row, dict):
                raise ValueError("Rich episode metadata rows must be objects")
            task_id = row.get("task_index")
            if type(task_id) is not int:
                raise ValueError("Rich episode metadata has an invalid task_index")
            data_chunk, data_file = (
                row.get("data/chunk_index"),
                row.get("data/file_index"),
            )
            if type(data_chunk) is int and type(data_file) is int:
                owners[f"data/chunk-{data_chunk:03d}/file-{data_file:03d}.parquet"].add(
                    task_id
                )
            for modality in MODALITIES:
                for camera in CAMERAS:
                    try:
                        owners[_video_path(row, modality, camera)].add(task_id)
                    except ValueError:
                        # The caller will produce the more specific missing-modality error.
                        pass
    return owners


def _json_value(value: Any) -> Any:
    """Convert Arrow/numpy scalar values into JSON-safe Python values."""
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    return value


def _record(row: dict) -> dict:
    pose = {
        camera: _json_value(row[f"observation.robot2cam_pose.{source_camera}"])
        for camera, source_camera in CAMERAS.items()
    }
    return {
        "index": _json_value(row.get("index")),
        "episode_index": _json_value(row.get("episode_index")),
        "frame_index": _json_value(row.get("frame_index")),
        "timestamp": _json_value(row.get("timestamp")),
        "action": _json_value(row.get("action")),
        "reward": _json_value(row.get("next.reward")),
        "terminated": _json_value(row.get("next.terminated")),
        "truncated": _json_value(row.get("next.truncated")),
        "proprioception": _json_value(row.get("observation.state")),
        "camera_poses": pose,
    }


def _write_episode_records(
    stage: Path, source: Path, episode_id: str, row: dict
) -> str:
    parquet = _pyarrow()
    data_chunk, data_file = row.get("data/chunk_index"), row.get("data/file_index")
    if type(data_chunk) is not int or type(data_file) is not int:
        raise ValueError("Episode is missing numerical data shard metadata")
    path = _file(source, f"data/chunk-{data_chunk:03d}/file-{data_file:03d}.parquet")
    table = parquet.read_table(path)
    all_rows = table.to_pylist()
    if any(not isinstance(item, dict) for item in all_rows):
        raise ValueError("Rich numerical records must be objects")
    episode_index = row.get("episode_index")
    start, end = row.get("dataset_from_index"), row.get("dataset_to_index")
    selected = [item for item in all_rows if item.get("episode_index") == episode_index]
    if type(start) is int and type(end) is int:
        selected = [item for item in selected if start <= item.get("index", -1) < end]
    selected.sort(key=lambda item: (item.get("index", -1), item.get("frame_index", -1)))
    if len(selected) != int(row.get("length", len(selected))):
        raise ValueError(
            f"Numerical records for {episode_id} do not match episode length "
            f"({len(selected)} != {row.get('length')})"
        )
    relative = f"records/{episode_id}.jsonl.gz"
    target = stage / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(target, "wt", encoding="utf-8", newline="\n") as stream:
        for item in selected:
            stream.write(
                json.dumps(_record(item), ensure_ascii=False, allow_nan=False) + "\n"
            )
    return relative


def _copy_source_files(stage: Path, source: Path, relatives: set[str]) -> None:
    for relative in sorted(relatives):
        _link_or_copy(_file(source, relative), stage / relative)


def _inventory(root: Path) -> list[dict]:
    files: list[dict] = []
    for directory, folders, names in os.walk(root, followlinks=False):
        for name in sorted(folders + names):
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError(f"Symlinks are forbidden in rich materials: {path}")
            if path.is_file():
                files.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
            elif not path.is_dir():
                raise ValueError(f"Special file in rich materials: {path}")
    return sorted(files, key=lambda item: item["path"])


def _source_info(source: Path) -> dict:
    try:
        value = json.loads(_file(source, "meta/info.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Rich source has invalid meta/info.json") from exc
    if not isinstance(value, dict):
        raise ValueError("Rich source has invalid meta/info.json")
    if value.get("robot_type") != "R1Pro":
        raise ValueError("Rich demonstrations must be from the BEHAVIOR R1Pro dataset")
    features = value.get("features", {})
    if not isinstance(features, dict):
        raise ValueError("Rich source has an invalid feature schema")
    required = {
        "action",
        "next.reward",
        "next.terminated",
        "next.truncated",
        "observation.state",
    }
    required.update(f"observation.robot2cam_pose.{name}" for name in CAMERAS.values())
    required.update(
        f"observation.{modality}.{name}"
        for modality in MODALITIES
        for name in CAMERAS.values()
    )
    missing = sorted(required - set(features))
    if missing:
        raise ValueError(
            "Rich source is missing required records: " + ", ".join(missing)
        )

    def shape(name):
        feature = features.get(name)
        return feature.get("shape") if isinstance(feature, dict) else None

    if (
        shape("action") != [23]
        or shape("observation.state") != [61]
        or any(
            shape(f"observation.robot2cam_pose.{name}") != [7]
            for name in CAMERAS.values()
        )
        or not isinstance(value.get("fps"), (int, float))
        or not math.isfinite(float(value["fps"]))
        or value["fps"] <= 0
    ):
        raise ValueError("Rich source has an unexpected R1Pro feature schema")
    return value


def validate_rich_materials(
    root: Path,
    *,
    task_name: str | None = None,
    expected_episodes: list[dict] | None = None,
    source_revision: str | None = None,
    action_profile: str | None = None,
) -> dict:
    """Validate a staged bundle and reject unlisted files or path escapes."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Rich training_set must be a regular directory")
    index_path = root / "index.json"
    if index_path.is_symlink() or not index_path.is_file():
        raise ValueError("Rich training_set is missing index.json")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError("Rich training index is not valid JSON") from exc
    required = {
        "schema_version",
        "bundle_type",
        "benchmark",
        "task_name",
        "task_index",
        "source",
        "episodes",
        "files",
        "modalities",
    }
    converted = isinstance(index, dict) and index.get("schema_version") == 3
    if converted:
        required.add("action_conversion")
    if not isinstance(index, dict) or set(index) != required:
        raise ValueError("Invalid rich training index fields")
    if (
        index["schema_version"] not in (RICH_SCHEMA_VERSION, 3)
        or index["bundle_type"] != "behavior_1k_official_task"
    ):
        raise ValueError("Unsupported rich training index schema")
    if action_profile is not None:
        _validate_action_profile(action_profile)
        if (action_profile == EEF21) != converted:
            raise ValueError(
                "Rich bundle action_profile differs from experiment; use a new output"
            )
    if (
        index["benchmark"] != "behavior_1k"
        or not isinstance(index["task_name"], str)
        or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", index["task_name"])
    ):
        raise ValueError("Rich training index has an invalid task")
    if task_name is not None and index["task_name"] != task_name:
        raise ValueError("Rich training index task differs from configuration")
    if type(index["task_index"]) is not int or index["task_index"] < 0:
        raise ValueError("Invalid rich task_index")
    source_descriptor = index["source"]
    if (
        not isinstance(source_descriptor, dict)
        or set(source_descriptor) != {"repo_id", "revision"}
        or source_descriptor["repo_id"] != DATASET_REPO
        or not isinstance(source_descriptor["revision"], str)
        or not source_descriptor["revision"]
    ):
        raise ValueError("Rich source descriptor is invalid")
    if source_revision is not None and source_descriptor["revision"] != source_revision:
        raise ValueError("Rich training source revision differs from configuration")
    modalities = index["modalities"]
    if (
        not isinstance(modalities, dict)
        or set(modalities) != {"rgb", "depth", "records", "annotations"}
        or modalities["rgb"] != list(CAMERAS)
        or modalities["depth"] != []
        or modalities["annotations"] is not True
        or modalities["records"]
        != [
            "action",
            "proprioception",
            "camera_poses",
            "reward",
            "terminated",
            "truncated",
            "timestamp",
            "frame_index",
        ]
    ):
        raise ValueError("Rich modality descriptor is invalid")
    if not isinstance(index["episodes"], list) or not index["episodes"]:
        raise ValueError("Rich training index must contain episodes")
    if expected_episodes is not None:
        expected = {
            episode.get("id"): episode.get("instance_index")
            for episode in expected_episodes
            if isinstance(episode, dict)
        }
        actual_selection = {
            episode.get("id"): episode.get("instance_index")
            for episode in index["episodes"]
            if isinstance(episode, dict)
        }
        if actual_selection != expected:
            raise ValueError("Rich training index episodes differ from configuration")
    if not isinstance(index["files"], list):
        raise ValueError("Rich training index files must be a list")
    listed = {}
    for item in index["files"]:
        if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
            raise ValueError("Invalid rich file manifest entry")
        relative = _relative(item["path"])
        if relative == "index.json" or relative in listed:
            raise ValueError("Rich file manifest contains an invalid or duplicate path")
        if _is_depth_video(relative):
            raise ValueError(
                "Rich training bundle must not include depth_linear videos"
            )
        if (
            type(item["bytes"]) is not int
            or item["bytes"] < 0
            or not isinstance(item["sha256"], str)
            or not SHA256.fullmatch(item["sha256"])
        ):
            raise ValueError("Invalid rich file manifest hash")
        path = _file(root, relative)
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError("Rich file hash mismatch: " + relative)
        listed[relative] = item
    actual = {item["path"] for item in _inventory(root) if item["path"] != "index.json"}
    if actual != set(listed):
        raise ValueError("Rich training_set contains unlisted files")
    episode_ids = set()
    for episode in index["episodes"]:
        if not isinstance(episode, dict):
            raise ValueError("Invalid rich episode entry")
        expected_episode_fields = {
            "id",
            "instance_index",
            "task_index",
            "task_instance_id",
            "episode_index",
            "length",
            "records",
            "annotation",
            "annotation_source",
            "videos",
            "intervals",
        }
        if set(episode) != expected_episode_fields:
            raise ValueError("Invalid rich episode fields")
        for field in ("id", "records", "annotation", "videos", "intervals"):
            if field not in episode:
                raise ValueError("Rich episode is missing " + field)
        episode_id = episode["id"]
        if (
            not isinstance(episode_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]+", episode_id)
            or episode_id in episode_ids
        ):
            raise ValueError("Rich episode ids must be unique")
        if (
            type(episode["instance_index"]) is not int
            or episode["instance_index"] < 0
            or type(episode["task_index"]) is not int
            or episode["task_index"] != index["task_index"]
            or type(episode["task_instance_id"]) is not int
            or type(episode["episode_index"]) is not int
            or type(episode["length"]) is not int
            or episode["length"] <= 0
            or not isinstance(episode["annotation_source"], str)
        ):
            raise ValueError("Invalid rich episode metadata")
        _relative(episode["annotation_source"])
        if not episode["annotation_source"].startswith(
            f"annotations/task-{index['task_index']:04d}/"
        ):
            raise ValueError("Rich annotation source belongs to another task")
        episode_ids.add(episode_id)
        for relative in (episode["records"], episode["annotation"]):
            _file(root, relative)
            if relative not in listed:
                raise ValueError("Rich episode references an unlisted file")
        videos = episode["videos"]
        if not isinstance(videos, dict) or set(videos) != set(CAMERAS):
            raise ValueError("Rich episode must contain all three cameras")
        for camera, modalities in videos.items():
            if not isinstance(modalities, dict) or set(modalities) != set(MODALITIES):
                raise ValueError("Rich episode must contain only RGB for every camera")
            for relative in modalities.values():
                if not isinstance(relative, str):
                    raise ValueError("Rich video path must be a string")
                _file(root, relative)
                if relative not in listed:
                    raise ValueError("Rich episode references an unlisted video")
        intervals = episode["intervals"]
        if not isinstance(intervals, dict) or set(intervals) != set(CAMERAS):
            raise ValueError("Rich episode must contain intervals for all cameras")
        for camera in CAMERAS:
            if not isinstance(intervals[camera], dict) or set(intervals[camera]) != set(
                MODALITIES
            ):
                raise ValueError("Rich episode intervals must contain only RGB")
            for modality in MODALITIES:
                interval = intervals[camera][modality]
                if (
                    not isinstance(interval, dict)
                    or set(interval)
                    != {
                        "start_frame",
                        "start_timestamp",
                        "end_timestamp",
                        "fps",
                        "frames",
                    }
                    or type(interval["start_frame"]) is not int
                    or interval["start_frame"] < 0
                    or not all(
                        isinstance(interval[key], (int, float))
                        and not isinstance(interval[key], bool)
                        for key in ("start_timestamp", "end_timestamp", "fps")
                    )
                    or not all(
                        math.isfinite(float(interval[key]))
                        for key in ("start_timestamp", "end_timestamp", "fps")
                    )
                    or interval["fps"] <= 0
                    or interval["end_timestamp"] <= interval["start_timestamp"]
                    or type(interval["frames"]) is not int
                    or interval["frames"] != episode["length"]
                ):
                    raise ValueError("Invalid rich video interval")
    if converted:
        from .official_materials import validate_conversion

        validate_conversion(root, index)
    return index


def prepare_rich_materials(cfg: dict, *, source_root: Path, output_root: Path) -> dict:
    """Create an immutable task-only rich bundle from a downloaded dataset root."""
    action_profile = cfg.get("benchmark_options", {}).get("action_profile", JOINT23)
    _validate_action_profile(action_profile)
    task_name = cfg.get("task_name")
    if not task_name:
        tasks = [
            episode.get("task_name")
            for episode in cfg.get("train_episodes", [])
            if isinstance(episode, dict) and isinstance(episode.get("task_name"), str)
        ]
        if len(set(tasks)) != 1:
            raise ValueError("Rich preparation requires one task in train_episodes")
        task_name = tasks[0]
    source_candidate = Path(source_root).expanduser()
    output_candidate = Path(output_root).expanduser()
    if source_candidate.is_symlink():
        raise ValueError("rich_data_root must be a regular directory")
    if output_candidate.is_symlink():
        raise ValueError("Rich training_set must not be a symlink")
    source = source_candidate.resolve()
    output = output_candidate.resolve()
    if (
        source == output
        or source.is_relative_to(output)
        or output.is_relative_to(source)
    ):
        raise ValueError("Rich source and output directories must be disjoint")
    if not source.is_dir() or source.is_symlink():
        raise ValueError("rich_data_root must be a regular directory")
    if output.exists():
        expected_revision = _source_revision(source, cfg.get("rich_data_revision"))
        index = validate_rich_materials(
            output,
            task_name=task_name,
            expected_episodes=cfg.get("train_episodes"),
            source_revision=expected_revision,
            action_profile=action_profile,
        )
        if action_profile == EEF21:
            from .r1pro_fk import R1ProFK

            if index["action_conversion"] != R1ProFK.from_config(cfg).descriptor():
                raise ValueError("official FK assets changed; use a new training_set")
        return index

    info = _source_info(source)
    revision = _source_revision(source, cfg.get("rich_data_revision"))
    task_id = _task_id(source, task_name)
    metadata_files, rows = _episode_metadata(source, task_id)
    selected = _episode_selection({**cfg, "task_name": task_name}, rows, task_id)
    owners = _all_owners(source, metadata_files)

    relatives: set[str] = {"meta/info.json"}
    relatives.update(path.relative_to(source).as_posix() for path in metadata_files)
    episodes: list[dict] = []
    record_jobs: list[tuple[str, dict]] = []
    for episode, row in selected:
        episode_id = episode["id"]
        data_chunk, data_file = row.get("data/chunk_index"), row.get("data/file_index")
        data_relative = f"data/chunk-{data_chunk:03d}/file-{data_file:03d}.parquet"
        if owners.get(data_relative) != {task_id}:
            raise ValueError(
                "A numerical shard is shared with another task: " + data_relative
            )
        relatives.add(data_relative)
        annotation_source = row.get("annotation_path")
        if not isinstance(annotation_source, str):
            raise ValueError("Episode is missing annotation_path")
        if not annotation_source.startswith(f"annotations/task-{task_id:04d}/"):
            raise ValueError("Episode annotation belongs to another task")
        # Validate the source path here, but expose only the normalized
        # task-local name below; the original task directory is not needed by
        # the model and can reveal unrelated source layout.
        _file(source, annotation_source)
        modalities: dict[str, dict[str, str]] = {}
        intervals: dict[str, dict[str, int | float]] = {}
        for camera in CAMERAS:
            modalities[camera] = {}
            intervals[camera] = {}
            for modality in MODALITIES:
                relative = _video_path(row, modality, camera)
                if owners.get(relative) != {task_id}:
                    raise ValueError(
                        "A visual shard is shared with another task: " + relative
                    )
                relatives.add(relative)
                modalities[camera][modality] = relative
                source_camera = CAMERAS[camera]
                base = f"videos/observation.{modality}.{source_camera}"
                start = row.get(base + "/from_timestamp")
                end = row.get(base + "/to_timestamp")
                length = row.get("length")
                if (
                    not isinstance(start, (int, float))
                    or isinstance(start, bool)
                    or not isinstance(end, (int, float))
                    or isinstance(end, bool)
                    or type(length) is not int
                    or not math.isfinite(float(start))
                    or not math.isfinite(float(end))
                    or end <= start
                    or abs((float(end) - float(start)) * float(info["fps"]) - length)
                    > 0.1
                ):
                    raise ValueError(f"Invalid {modality}/{camera} episode interval")
                intervals[camera][modality] = {
                    "start_frame": int(round(float(start) * info["fps"])),
                    "start_timestamp": float(start),
                    "end_timestamp": float(end),
                    "fps": float(info["fps"]),
                    "frames": int(row["length"]),
                }
        record_jobs.append((episode_id, row))
        episodes.append(
            {
                "id": episode_id,
                "instance_index": episode["instance_index"],
                "task_index": task_id,
                "task_instance_id": int(row["task_instance_id"]),
                "episode_index": int(row["episode_index"]),
                "length": int(row["length"]),
                "records": f"records/{episode_id}.jsonl.gz",
                "annotation": f"annotations/{episode_id}.json",
                "annotation_source": annotation_source,
                "videos": modalities,
                "intervals": intervals,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".rich-materials-", dir=output.parent))
    try:
        _copy_source_files(stage, source, relatives)
        for episode, item in zip(episodes, record_jobs, strict=True):
            source_annotation = _file(source, item[1]["annotation_path"])
            target_annotation = stage / episode["annotation"]
            target_annotation.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_annotation, target_annotation)
            _write_episode_records(stage, source, episode["id"], item[1])
        _write_json(
            stage / "meta/task.json",
            {
                "task_name": task_name,
                "task_index": task_id,
                "source": {"repo_id": DATASET_REPO, "revision": revision},
                "feature_schema": info.get("features", {}),
            },
        )
        conversion = None
        if action_profile == EEF21:
            from .official_materials import convert_stage
            from .r1pro_fk import R1ProFK

            converter = R1ProFK.from_config(cfg)
            convert_stage(stage, converter)
            conversion = converter.descriptor()
        index = {
            "schema_version": 3 if conversion else RICH_SCHEMA_VERSION,
            **({"action_conversion": conversion} if conversion else {}),
            "bundle_type": "behavior_1k_official_task",
            "benchmark": "behavior_1k",
            "task_name": task_name,
            "task_index": task_id,
            "source": {
                "repo_id": DATASET_REPO,
                "revision": revision,
            },
            "modalities": {
                "rgb": list(CAMERAS),
                # Kept as an explicit empty descriptor so consumers can see
                # that depth was intentionally excluded from icl.
                "depth": [],
                "records": [
                    "action",
                    "proprioception",
                    "camera_poses",
                    "reward",
                    "terminated",
                    "truncated",
                    "timestamp",
                    "frame_index",
                ],
                "annotations": True,
            },
            "episodes": episodes,
            "files": [],
        }
        _write_json(stage / "index.json", index)
        files = _inventory(stage)
        files = [item for item in files if item["path"] != "index.json"]
        index["files"] = files
        _write_json(stage / "index.json", index)
        validate_rich_materials(
            stage, task_name=task_name, action_profile=action_profile
        )
        stage.rename(output)
        return index
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def material_index(path: Path) -> dict:
    """Load a validated rich index for prompt rendering."""
    return validate_rich_materials(
        Path(path).parent if Path(path).name == "index.json" else path
    )
