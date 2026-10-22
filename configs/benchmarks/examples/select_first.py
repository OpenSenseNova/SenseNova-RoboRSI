# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Example: select all declared seeds for the first N catalog task IDs."""


def select_cases(cases, *, benchmark, parameters):
    count = parameters.get("task_count", 1)
    if type(count) is not int or count <= 0:
        raise ValueError("task_count must be a positive integer")
    tasks = list(dict.fromkeys(case["task_id"] for case in cases))[:count]
    return [dict(case) for case in cases if case["task_id"] in tasks]
