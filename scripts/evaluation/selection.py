# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""User Python selectors operate on already validated catalog cells."""

from __future__ import annotations

import copy
import runpy
import sys
from pathlib import Path

from scripts.evaluation.configuration import BenchmarkConfig


def select_cells(config: BenchmarkConfig, cases: list) -> list:
    """Apply a trusted user function without admitting new task/seed pairs."""
    reference = config.evaluation.get("selector")
    if reference is None:
        return cases
    filename, _, function_name = reference.rpartition(":")
    available = {(case.task_id, case.seed): case for case in cases}
    original_path = sys.path.copy()
    try:
        sys.path.insert(0, str(Path(filename).parent))
        namespace = runpy.run_path(filename)
        function = namespace.get(function_name)
        if not callable(function):
            raise ValueError("evaluation.selector must name a callable")
        choices = function(
            tuple({"task_id": case.task_id, "seed": case.seed} for case in cases),
            benchmark=config.benchmark,
            parameters=copy.deepcopy(config.evaluation.get("parameters", {})),
        )
    finally:
        sys.path[:] = original_path
    if not isinstance(choices, (list, tuple)) or not choices:
        raise ValueError(
            "evaluation.selector must return a nonempty list of catalog cells"
        )
    selected, seen = [], set()
    for choice in choices:
        if (
            not isinstance(choice, dict)
            or set(choice) != {"task_id", "seed"}
            or not isinstance(choice["task_id"], str)
            or type(choice["seed"]) is not int
        ):
            raise ValueError(
                "evaluation.selector must return catalog task_id/seed mappings"
            )
        key = (choice["task_id"], choice["seed"])
        if key not in available or key in seen:
            raise ValueError(
                "evaluation.selector returned a duplicate or unknown catalog cell"
            )
        selected.append(available[key])
        seen.add(key)
    return selected
