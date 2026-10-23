# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
"""Camera records exposed to the pinned image tool functions."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image


def _metric_depth(depth: Any, camera_meta: dict) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    near = camera_meta.get("depth_near")
    far = camera_meta.get("depth_far")
    if near is not None and far is not None:
        d = near / (1.0 - d * (1.0 - near / far))
    return d


def _world_from_depth(depth_metric: np.ndarray, camera_meta: dict) -> np.ndarray:
    k_matrix = np.array(camera_meta["intrinsic_K"], dtype=np.float64)
    extrinsic = np.array(camera_meta["extrinsic_cam2world"], dtype=np.float64)
    fx, fy = k_matrix[0, 0], k_matrix[1, 1]
    cx, cy = k_matrix[0, 2], k_matrix[1, 2]
    height, width = depth_metric.shape
    rr, cc = np.mgrid[0:height, 0:width]
    z = depth_metric.astype(np.float64)
    camera_points = np.stack(
        [(cc - cx) * z / fx, (rr - cy) * z / fy, z, np.ones_like(z)],
        axis=-1,
    )
    return (camera_points @ extrinsic.T)[..., :3]


class CameraRecords:
    """Disk-backed snapshots shared by viewing, calibration and back-projection."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.latest = -1
        self.env_steps = -1

    def capture(self, env, values, env_steps, *, terminated=False, truncated=False):
        if self.env_steps == env_steps:
            return
        index = self.latest + 1
        directory = self.directory / f"{index:06d}"
        directory.mkdir(parents=True, exist_ok=False)
        for name, value in env.camera_artifacts().items():
            path = directory / name
            if name.endswith(".png"):
                Image.fromarray(value).save(path)
            elif name.endswith(".npz"):
                np.savez_compressed(path, value=value)
            else:
                path.write_text(json.dumps(value))
        metadata = {
            "step": index,
            "native_env_steps": env_steps,
            "terminated": terminated,
            "truncated": truncated,
            "state": values.get("robot_state_feedback", {}),
            "states": np.asarray(values.get("states", [])).tolist(),
            "task_language": values.get("task_descriptions"),
        }
        (directory / "record.json").write_text(json.dumps(metadata))
        self.latest = index
        self.env_steps = env_steps

    def get(self, step=-1):
        if type(step) is not int or step < -1:
            raise ValueError("step must be -1 or a recorded nonnegative state index")
        index = self.latest if step == -1 else step
        if index < 0 or index > self.latest:
            raise ValueError(f"state step {step} not available")
        directory = self.directory / f"{index:06d}"
        return SimpleNamespace(
            step_idx=index, artifacts={p.name for p in directory.iterdir()}
        )

    def load(self, name, *, step=-1):
        record = self.get(step)
        path = self.directory / f"{record.step_idx:06d}" / name
        if name.endswith(".npz"):
            with np.load(path, allow_pickle=False) as data:
                return data["value"]
        if name.endswith(".png"):
            with Image.open(path) as image:
                return np.asarray(image).copy()
        return json.loads(path.read_text())

    def view(self, step=-1):
        record = self.get(step)
        metadata = self.load("record.json", step=record.step_idx)
        images = {
            name: self.load(name + ".png", step=record.step_idx)
            for name in ("agentview", "agentview_high", "wrist_high")
        }
        return metadata, images
