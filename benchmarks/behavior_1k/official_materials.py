# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Convert a staged official bundle without mutating its source data."""

import copy
import gzip
import json

from robots.behavior_1k.contracts import validate_action
from robots.behavior_1k.official import ACTION_LAYOUT, EEF21


def convert_stage(stage, converter):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from .official_demonstrations import _write_json

    for path in sorted((stage / "records").glob("*.jsonl.gz")):
        temporary = path.with_suffix(".converted.gz")
        with gzip.open(path, "rt", encoding="utf-8") as source:
            with gzip.open(temporary, "wt", encoding="utf-8") as target:
                for line in source:
                    row = json.loads(line)
                    row["action"] = converter.convert(row["action"])
                    target.write(json.dumps(row, allow_nan=False) + "\n")
        temporary.replace(path)
    # Parquet shards are also public. Convert them as well so the agent cannot
    # accidentally learn a different action representation via another reader.
    for path in sorted((stage / "data").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(path)
        actions = [converter.convert(action) for action in table["action"].to_pylist()]
        table = table.set_column(
            table.schema.get_field_index("action"),
            "action",
            pa.array(actions, type=pa.list_(pa.float32(), 21)),
        )
        # Drop stale LeRobot/pandas feature metadata describing action[23].
        table = table.replace_schema_metadata(None)
        temporary = path.with_suffix(".converted.parquet")
        pq.write_table(table, temporary)
        temporary.replace(path)  # original may be hardlinked; never overwrite its inode
    # Joint-action statistics cannot be mapped through nonlinear FK by merely
    # converting their means/min/max. Remove those stale 23D statistics.
    for path in sorted((stage / "meta/episodes").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(path)
        stale = [
            name for name in table.column_names if name.startswith("stats/action/")
        ]
        if stale:
            table = table.drop(stale).replace_schema_metadata(None)
            temporary = path.with_suffix(".converted.parquet")
            pq.write_table(table, temporary)
            temporary.replace(path)
    info_path = stage / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"]["action"] = {
        "dtype": "float32",
        "shape": [21],
        "names": [
            f"{name}_{i}"
            for name, start, stop in ACTION_LAYOUT
            for i in range(stop - start)
        ],
    }
    info["action_conversion"] = converter.descriptor()
    _write_json(info_path, info)
    task_path = stage / "meta/task.json"
    task = json.loads(task_path.read_text())
    task["feature_schema"] = copy.deepcopy(info["features"])
    task["action_conversion"] = converter.descriptor()
    _write_json(task_path, task)


def validate_conversion(root, index):
    conversion = index.get("action_conversion")
    if not isinstance(conversion, dict) or conversion.get("action_profile") != EEF21:
        raise ValueError("official bundle is missing its FK conversion descriptor")
    if conversion.get("method") != "omnigibson.utils.control_utils.FKSolver":
        raise ValueError("official labels must use official target FK")
    for name in ("meta/info.json", "meta/task.json"):
        metadata = json.loads((root / name).read_text())
        features = metadata.get("features", metadata.get("feature_schema", {}))
        if features.get("action", {}).get("shape") != [21]:
            raise ValueError("official feature schema must declare action[21]")
        if metadata.get("action_conversion") != conversion:
            raise ValueError("official conversion metadata differs from index")
    for episode in index["episodes"]:
        count = 0
        with gzip.open(root / episode["records"], "rt", encoding="utf-8") as stream:
            for line in stream:
                validate_action(json.loads(line)["action"], action_dim=21)
                count += 1
        if count != episode["length"]:
            raise ValueError("official record count differs from episode length")
    from .official_demonstrations import _pyarrow

    for item in index["files"]:
        if item["path"].startswith("data/") and item["path"].endswith(".parquet"):
            table = _pyarrow().read_table(root / item["path"], columns=["action"])
            for action in table["action"].to_pylist():
                validate_action(action, action_dim=21)
