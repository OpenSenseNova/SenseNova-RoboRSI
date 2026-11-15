# Copyright 2026 The RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""LIBERO evaluation catalogs freeze public task and budget semantics."""

import pytest

from robots.libero.benchmark_catalog import (
    PRO_SUITES,
    ordered_cases,
    pro_cases,
)


def _pro_instructions() -> dict[str, list[str]]:
    return {
        suite: [f"{suite} task {index}" for index in range(10)] for suite in PRO_SUITES
    }


@pytest.mark.parametrize(
    ("profile", "expected"), (("migration_subset", 80), ("official", 800))
)
def test_pro_catalog_counts_and_uses_explicit_init_states(profile, expected):
    cases = pro_cases(_pro_instructions(), profile=profile)
    assert len(cases) == expected
    assert len({case.task_id for case in cases}) == expected
    assert {case.native["init_state_id"] for case in cases} == (
        {3} if profile == "migration_subset" else set(range(10))
    )
    assert all(case.official_1x_steps == case.max_env_steps == 10_000 for case in cases)


@pytest.mark.parametrize(
    ("benchmark", "tasks", "profile", "expected"),
    (
        ("libero_long_plus_plus", 12, "migration_subset", 12),
        ("libero_long_plus_plus", 12, "official", 120),
        ("libero_ultra_long", 9, "migration_subset", 9),
        ("libero_ultra_long", 9, "official", 90),
    ),
)
def test_ordered_catalog_counts_and_labels_trials_without_seed_abuse(
    benchmark, tasks, profile, expected
):
    cases = ordered_cases(
        benchmark,
        [f"task {index}" for index in range(tasks)],
        [3 + index % 2 for index in range(tasks)],
        profile=profile,
    )
    assert len(cases) == expected
    assert all(case.manifest_task()["seeds"] == [0] for case in cases)
    assert {case.native["observation_protocol"] for case in cases} == {
        "lilo-instance-pose-feedback-v1"
    }
    if profile == "official":
        assert {case.native["trial_id"] for case in cases} == set(range(10))
        assert all(
            case.native["reset_protocol"] == "official_bare_random_reset"
            for case in cases
        )
    else:
        assert {case.native["init_state_id"] for case in cases} == {1}


def test_plus9_multiplies_only_the_native_failure_budget():
    case = ordered_cases(
        "libero_ultra_long",
        [f"task {index}" for index in range(9)],
        [9] * 9,
        profile="migration_subset",
        plus9_on_failure=True,
    )[0]
    assert case.official_1x_steps == 1_800
    assert case.max_env_steps == 18_000
    assert case.native["plus9_on_failure"] is True
