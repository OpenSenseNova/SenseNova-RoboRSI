# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Pinned task catalog and public official 21D action contract."""

from __future__ import annotations

import json
from pathlib import Path

from roborsi.evaluation.messages import Task

NATIVE_COMMIT = "b1979916ec1549b10a4e65e630bc6504a9af1b00"
DATA_REVISION = "4f50b44796641a4d526a19d9aeadc8aa51e2f2c2"
ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 21,
            "maxItems": 21,
            "description": "base velocity(3), torso joints(4), left base-frame xyz+rotvec(6), gripper(1), right base-frame xyz+rotvec(6), gripper(1)",
        }
    },
    "required": ["action"],
    "additionalProperties": False,
}
CHANNELS = sorted(
    [
        "head_image",
        "left_wrist_image",
        "right_wrist_image",
        "head_depth_image",
        "left_wrist_depth_image",
        "right_wrist_depth_image",
    ]
)
CAPABILITIES = {
    "action_schema": ACTION_SCHEMA,
    "observation_channels": CHANNELS + ["proprio"],
}
EVALUATION_RULES = {
    "success_metric": "native_success",
    "score_metric": "native_q_score_final",
    "aggregation": "case_mean",
    "error_denominator": "valid",
}


def validate_task_name(task_name: str) -> None:
    """Keep the published pipeline within its pinned 100-task selection."""
    selection = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/behavior_1k/official_tasks.json"
    )
    if task_name not in json.loads(selection.read_text())["tasks"]:
        raise ValueError("task is outside the official 100-task selection")


def task_from_metadata(metadata: dict) -> Task:
    """Resolve instructions and limits from an actual native reset."""
    validate_task_name(metadata["task_name"])
    if metadata["mode"] != "public_test" or metadata["instance_id"] != 301:
        raise ValueError("official scope requires public_test instance 301")
    if metadata["action_dim"] != 21 or metadata["action_profile"] != "r1pro_eef21_v1":
        raise ValueError("native action contract differs")
    return Task(
        benchmark="behavior_1k",
        task_id=metadata["task_name"],
        seed=0,
        instruction=metadata["task_instruction"],
        max_steps=metadata["max_steps"],
        native={
            "mode": "public_test",
            "instance_id": 301,
            "eval_seed": 0,
            "action_profile": "r1pro_eef21_v1",
            "commit": NATIVE_COMMIT,
        },
        capabilities=CAPABILITIES,
    )


def read_catalog(path: Path) -> list[Task]:
    """Reject altered scope, duplicated identities and non-native catalogs."""
    data = json.loads(path.read_text())
    if data["schema_version"] != 1 or data["native_commit"] != NATIVE_COMMIT:
        raise ValueError("catalog native revision differs")
    tasks = [Task.from_dict(item) for item in data["tasks"]]
    if not tasks or len({(t.task_id, t.seed) for t in tasks}) != len(tasks):
        raise ValueError("catalog needs unique task identities")
    for task in tasks:
        expected = task_from_metadata(
            {
                "mode": "public_test",
                "instance_id": 301,
                "action_dim": 21,
                "action_profile": "r1pro_eef21_v1",
                "task_name": task.task_id,
                "task_instruction": task.instruction,
                "max_steps": task.max_steps,
            }
        )
        if task != expected:
            raise ValueError("catalog task contract differs")
    return tasks
