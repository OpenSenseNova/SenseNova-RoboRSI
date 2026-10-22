# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

from pathlib import Path

import pytest

from scripts.libero_three.run_combined import _DEFAULT_TASKS
from scripts.libero_three.run_combined_batch import _tasks

_REPOSITORY = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize(
    ("benchmark", "profile", "expected"),
    (
        ("libero_pro", "official", 800),
        ("libero_pro", "migration_subset", 80),
        ("libero_long_plus_plus", "official", 120),
        ("libero_long_plus_plus", "migration_subset", 12),
        ("libero_ultra_long", "official", 90),
        ("libero_ultra_long", "migration_subset", 9),
    ),
)
def test_combined_batch_exposes_official_and_subset_profiles(
    benchmark, profile, expected
):
    tasks = _tasks(_REPOSITORY, benchmark, profile)
    assert len(tasks) == expected
    assert len(set(tasks)) == expected


def test_single_case_runner_defaults_to_official_cases():
    assert _DEFAULT_TASKS["official"] == {
        "libero_long_plus_plus": "t00-r00",
        "libero_ultra_long": "t00-r00",
        "libero_pro": "libero_spatial_task-t00-i00",
    }
