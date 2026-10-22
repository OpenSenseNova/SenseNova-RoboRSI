# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

import pytest

from roborsi.evaluation.messages import Task
from robots.libero.three_benchmark_agent import context_package_key


def _task(benchmark: str, task_id: str) -> Task:
    return Task(
        benchmark=benchmark,
        task_id=task_id,
        seed=0,
        instruction="test",
        max_steps=1,
        native={},
        capabilities={},
    )


@pytest.mark.parametrize(
    ("benchmark", "task_id", "expected"),
    (
        (
            "libero_pro",
            "libero_spatial_task-t00-i00",
            "libero_spatial_task-t00-i03",
        ),
        (
            "libero_pro",
            "libero_10_swap-t09-i09",
            "libero_10_swap-t09-i03",
        ),
        ("libero_long_plus_plus", "t11-r09", "t11-i01"),
        ("libero_ultra_long", "t08-r00", "t08-i01"),
        ("libero_ultra_long", "t08-i01", "t08-i01"),
    ),
)
def test_all_official_trials_reuse_the_task_level_icl_package(
    benchmark, task_id, expected
):
    assert context_package_key(_task(benchmark, task_id)) == expected


def test_context_package_key_rejects_an_unknown_case_shape():
    with pytest.raises(ValueError, match="invalid ordered LIBERO task id"):
        context_package_key(_task("libero_long_plus_plus", "t00-i01-extra"))
