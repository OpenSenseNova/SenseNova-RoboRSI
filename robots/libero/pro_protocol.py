# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
"""Explicit PRO protocol extension; default catalog semantics stay upstream."""

from dataclasses import dataclass

from robots.libero.benchmark_catalog import PRO_SUITES as PRO_SUITES
from robots.libero.benchmark_catalog import CatalogCase
from robots.libero.benchmark_catalog import pro_cases as original_pro_cases


@dataclass(frozen=True, slots=True)
class ConfiguredCase(CatalogCase):
    seed: int = 0

    def manifest_task(self):
        row = CatalogCase.manifest_task(self)
        row["seeds"] = [self.seed]
        return row


def load_pro_config(path) -> dict:
    """Read the same explicit protocol in the CLI and Benchmark process."""
    from pathlib import Path

    import yaml

    return validate_pro_config(yaml.safe_load(Path(path).read_text()))


def validate_pro_config(config: dict) -> dict:
    """Validate explicit task selection and physical budget; reject silent defaults."""
    fields = {
        "benchmark",
        "categories",
        "variants",
        "task_indices",
        "init_state_ids",
        "seed",
        "max_env_steps",
    }
    if not isinstance(config, dict) or set(config) != fields:
        raise ValueError("PRO config requires exactly: " + ", ".join(sorted(fields)))
    if config["benchmark"] != "libero_pro":
        raise ValueError("expected libero_pro benchmark")
    selections = {
        "categories": {"spatial", "object", "goal", "10"},
        "variants": {"task", "swap"},
        "task_indices": set(range(10)),
        "init_state_ids": set(range(10)),
    }
    for name, allowed in selections.items():
        values = config[name]
        expected_type = int if name in {"task_indices", "init_state_ids"} else str
        if (
            not isinstance(values, list)
            or not values
            or any(type(v) is not expected_type for v in values)
            or len(set(values)) != len(values)
            or not set(values) <= allowed
        ):
            raise ValueError(f"invalid PRO selection: {name}")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(config["max_env_steps"]) is not int or config["max_env_steps"] < 1:
        raise ValueError("max_env_steps must be a positive integer")
    return config


def validate_pro_80_config(config: dict) -> dict:
    """Validate the supported 80-case PRO protocol independently of its filename."""
    expected = {
        "benchmark": "libero_pro",
        "categories": ["spatial", "object", "goal", "10"],
        "variants": ["task", "swap"],
        "task_indices": list(range(10)),
        "init_state_ids": [0],
        "seed": 0,
        "max_env_steps": 2000,
    }
    validate_pro_config(config)
    if config != expected:
        raise ValueError(
            "explicit PRO configuration requires the fixed 80-case, init-0, seed-0, 2000-step protocol"
        )
    return config


def pro_cases(instructions, *, config, profile="official", plus9_on_failure=False):
    """Select public task definitions, then apply only the explicit run protocol."""
    validate_pro_config(config)
    if profile not in {"official", "migration_subset"}:
        raise ValueError("unknown PRO profile")
    if plus9_on_failure:
        raise ValueError("configured PRO runs prohibit extra-budget recovery")
    originals = {case.task_id: case for case in original_pro_cases(instructions)}
    cases = []
    budget = config["max_env_steps"]
    for category in config["categories"]:
        for variant in config["variants"]:
            suite = f"libero_{category}_{variant}"
            for index in range(10):
                if index not in config["task_indices"]:
                    continue
                for init in config["init_state_ids"]:
                    case = originals[f"{suite}-t{index:02d}-i{init:02d}"]
                    native = {
                        **case.native,
                        "official_1x_steps": budget,
                        "max_env_steps": budget,
                    }
                    cases.append(
                        ConfiguredCase(
                            benchmark=case.benchmark,
                            task_id=case.task_id,
                            instruction=case.instruction,
                            native=native,
                            official_1x_steps=budget,
                            max_env_steps=budget,
                            seed=config["seed"],
                        )
                    )
    return tuple(cases)
