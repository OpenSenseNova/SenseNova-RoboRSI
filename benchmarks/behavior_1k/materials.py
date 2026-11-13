# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Prepare task-scoped official 21D demonstrations for offline ICL."""

import argparse
from pathlib import Path

from benchmarks.behavior_1k.official_demonstrations import (
    _episode_metadata,
    _task_id,
    prepare_rich_materials,
)
from robots.behavior_1k.catalog import DATA_REVISION
from robots.behavior_1k.official import EEF21


def training_episodes(source: Path, task: str) -> list[dict]:
    """Select the actual public demonstration instances, whose IDs have gaps."""
    _, rows = _episode_metadata(source, _task_id(source, task))
    instances = [row.get("task_instance_id") for row in rows]
    if (
        len(instances) != 200
        or any(type(value) is not int or not 1 <= value <= 300 for value in instances)
        or len(set(instances)) != 200
    ):
        raise ValueError("expected 200 distinct public training instances in 1–300")
    return [
        {
            "id": f"practice-{i}",
            "task_name": task,
            "mode": "train",
            "instance_index": instance,
        }
        for i, instance in enumerate(instances, start=1)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = {
        "task_name": args.task,
        "rich_data_revision": DATA_REVISION,
        "benchmark_options": {
            "action_profile": EEF21,
            "repo_root": str(args.native_root),
            "data_root": str(args.data_root),
        },
        "train_episodes": training_episodes(args.source, args.task),
    }
    prepare_rich_materials(cfg, source_root=args.source, output_root=args.output)


if __name__ == "__main__":
    main()
