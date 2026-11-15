# Copyright 2026 The SenseNova-RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Deterministic task catalogs for the three public LIBERO evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from robots.libero.lilo_assistance import ASSISTANCE_CONTRACT

PRO_SUITES = tuple(
    f"libero_{family}_{variant}"
    for family in ("spatial", "object", "goal", "10")
    for variant in ("task", "swap")
)
LONG_PLUS_PLUS_SUITE = "libero_long_plus_plus_all"
ULTRA_LONG_SUITE = "ultra_long_all"

BenchmarkName = Literal["libero_pro", "libero_long_plus_plus", "libero_ultra_long"]
Profile = Literal["migration_subset", "official"]


@dataclass(frozen=True, slots=True)
class CatalogCase:
    """One immutable native LIBERO task declaration."""

    benchmark: BenchmarkName
    task_id: str
    instruction: str
    native: dict
    official_1x_steps: int
    max_env_steps: int

    def manifest_task(self) -> dict:
        """Return the RoboRSI manifest row for this single native case."""
        return {
            "task_id": self.task_id,
            "seeds": [0],
            "native": dict(self.native),
            "instruction": self.instruction,
            # The Adapter independently enforces max_env_steps. This upper bound
            # also permits policies that submit one physical action per turn.
            "max_steps": self.max_env_steps,
        }


def _budget(official_1x_steps: int, plus9_on_failure: bool) -> int:
    if type(official_1x_steps) is not int or official_1x_steps < 1:
        raise ValueError("official 1x steps must be a positive integer")
    if type(plus9_on_failure) is not bool:
        raise TypeError("plus9_on_failure must be a boolean")
    return official_1x_steps * (10 if plus9_on_failure else 1)


def pro_cases(
    instructions: dict[str, list[str]],
    *,
    profile: Profile = "official",
    plus9_on_failure: bool = False,
) -> tuple[CatalogCase, ...]:
    """Expand the eight LIBERO-PRO Task/Swap suites.

    The migration subset uses the delivery's fixed held-out state. The official
    profile uses the ten leaderboard states.
    """
    if set(instructions) != set(PRO_SUITES):
        raise ValueError("PRO instructions must contain exactly eight Task/Swap suites")
    init_states = (3,) if profile == "migration_subset" else tuple(range(10))
    cases = []
    for suite in PRO_SUITES:
        if len(instructions[suite]) != 10:
            raise ValueError(f"PRO suite {suite} must contain ten tasks")
        for task_index, instruction in enumerate(instructions[suite]):
            for init_state_id in init_states:
                one_x = 10_000
                cases.append(
                    CatalogCase(
                        benchmark="libero_pro",
                        task_id=f"{suite}-t{task_index:02d}-i{init_state_id:02d}",
                        instruction=instruction,
                        native={
                            "suite": suite,
                            "task_index": task_index,
                            "aggregation_task": f"{suite}-t{task_index:02d}",
                            "reset_protocol": "fixed_init_state",
                            "init_state_id": init_state_id,
                            "official_1x_steps": one_x,
                            "max_env_steps": _budget(one_x, plus9_on_failure),
                            "plus9_on_failure": plus9_on_failure,
                            "score_protocol": "final_goal_success_at_1x",
                        },
                        official_1x_steps=one_x,
                        max_env_steps=_budget(one_x, plus9_on_failure),
                    )
                )
    return tuple(cases)


def ordered_cases(
    benchmark: Literal["libero_long_plus_plus", "libero_ultra_long"],
    instructions: list[str],
    skill_counts: list[int],
    *,
    profile: Profile = "official",
    plus9_on_failure: bool = False,
) -> tuple[CatalogCase, ...]:
    """Expand Long++ or Ultra-Long without treating repeat labels as RNG seeds."""
    expected = 12 if benchmark == "libero_long_plus_plus" else 9
    suite = LONG_PLUS_PLUS_SUITE if expected == 12 else ULTRA_LONG_SUITE
    if len(instructions) != expected or len(skill_counts) != expected:
        raise ValueError(f"{benchmark} requires exactly {expected} configurations")
    if any(type(count) is not int or count < 1 for count in skill_counts):
        raise ValueError("ordered LIBERO tasks need positive skill counts")
    repetitions: tuple[tuple[str, int | None], ...]
    if profile == "migration_subset":
        repetitions = (("i01", 1),)
    else:
        # LiLo's released protocol performs a bare random reset. rXX is an
        # independent repeat label and deliberately is not exposed as a seed.
        repetitions = tuple((f"r{trial:02d}", None) for trial in range(10))
    cases = []
    for task_index, (instruction, skill_count) in enumerate(
        zip(instructions, skill_counts, strict=True)
    ):
        one_x = skill_count * 200
        for label, init_state_id in repetitions:
            native = {
                "suite": suite,
                "task_index": task_index,
                "aggregation_task": f"t{task_index:02d}",
                "reset_protocol": (
                    "fixed_init_state"
                    if init_state_id is not None
                    else "official_bare_random_reset"
                ),
                "official_1x_steps": one_x,
                "max_env_steps": _budget(one_x, plus9_on_failure),
                "plus9_on_failure": plus9_on_failure,
                "score_protocol": "ordered_prefix_at_1x",
                "observation_protocol": ASSISTANCE_CONTRACT,
                "skill_count": skill_count,
            }
            if init_state_id is not None:
                native["init_state_id"] = init_state_id
            else:
                native["trial_id"] = int(label[1:])
            cases.append(
                CatalogCase(
                    benchmark=benchmark,
                    task_id=f"t{task_index:02d}-{label}",
                    instruction=instruction,
                    native=native,
                    official_1x_steps=one_x,
                    max_env_steps=_budget(one_x, plus9_on_failure),
                )
            )
    return tuple(cases)
