# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

import pytest


def test_explicit_pro_protocol_matches_init0_packages():
    import json
    from pathlib import Path

    from robots.libero.pro_protocol import PRO_SUITES, load_pro_config, pro_cases

    root = Path(__file__).resolve().parents[4]
    protocol = load_pro_config(root / "configs/evaluations/libero_pro_80_2000.yaml")
    cases = pro_cases(
        {suite: ["fixture"] * 10 for suite in PRO_SUITES}, config=protocol
    )
    packages = json.loads(
        (root / "configs/agents/libero/context_maps/libero_pro_init0.json").read_text()
    )["packages"]
    assert len(cases) == 80 and len(set(packages.values())) == 40
    assert set(packages) == {case.task_id for case in cases}
    assert all(
        case.max_env_steps == 2000
        and case.native["init_state_id"] == 0
        and case.seed == 0
        for case in cases
    )
    with pytest.raises(ValueError, match="extra-budget"):
        pro_cases(
            {suite: ["fixture"] * 10 for suite in PRO_SUITES},
            config=protocol,
            plus9_on_failure=True,
        )


def test_named_pro_80_protocol_cannot_drift():
    from pathlib import Path

    from robots.libero.pro_protocol import load_pro_config, validate_pro_80_config

    root = Path(__file__).resolve().parents[4]
    config = load_pro_config(root / "configs/evaluations/libero_pro_80_2000.yaml")
    assert validate_pro_80_config(config) == config
    for field, value in (
        ("max_env_steps", 10000),
        ("seed", 1),
        ("init_state_ids", [3]),
        ("variants", ["task"]),
    ):
        with pytest.raises(ValueError, match="fixed 80-case"):
            validate_pro_80_config({**config, field: value})
