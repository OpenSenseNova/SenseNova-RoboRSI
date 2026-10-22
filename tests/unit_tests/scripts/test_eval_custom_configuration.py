# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Custom configuration changes selection without rebuilding native environments."""

import importlib.util
from pathlib import Path

import pytest
import yaml

from scripts.evaluation import configuration

ROOT = Path(__file__).resolve().parents[3]


def runner(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "custom_eval_runner", ROOT / "scripts/evaluation/evaluate_model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "IS_WINDOWS", False)
    return module


def test_minimal_benchmark_configuration_inherits_native_environment(tmp_path):
    path = tmp_path / "smoke.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "benchmark": "libero_pro",
                "evaluation": {"tasks": ["pick_cup"], "seeds": [0]},
            }
        )
    )
    selected = configuration.load_benchmark("libero_pro", path=path)
    canonical = configuration.load_benchmark("libero_pro")
    assert selected.environment == canonical.environment
    assert selected.agent == canonical.agent
    assert selected.model == canonical.model
    assert selected.evaluation == {"tasks": ["pick_cup"], "seeds": [0]}


def test_three_config_paths_resolve_before_planning(tmp_path, monkeypatch):
    benchmark = tmp_path / "smoke.yaml"
    benchmark.write_text(
        "schema_version: 1\nbenchmark: libero_pro\nevaluation:\n  seeds: [0]\n"
    )
    model = tmp_path / "model.yaml"
    model.write_text(
        "schema_version: 1\nprovider: openai\nbase_url: http://model.test/v1\nmodel_name: own-model\n"
    )
    agent = ROOT / "configs/agents/libero/pro.yaml"
    module = runner(monkeypatch)
    _, args = module.parse_args(
        [
            "--benchmark",
            "libero_pro",
            "--benchmark-config",
            str(benchmark),
            "--agent-config",
            str(agent),
            "--model-config",
            str(model),
            "--dry-run",
        ]
    )
    assert args.configurations["libero_pro"].agent == agent
    assert args.configurations["libero_pro"].evaluation["seeds"] == [0]
    assert args.models["libero_pro"].model_name == "own-model"
    assert args.models["libero_pro"].base_url == "http://model.test/v1"


@pytest.mark.parametrize(
    "evaluation",
    [
        {"seeds": [-1]},
        {"seeds": [True]},
        {"tasks": []},
        {"seeds": [0, 0]},
        {"unknown": 1},
        {"selector": "missing.py:select"},
    ],
)
def test_bad_selection_fails_before_environment_setup(tmp_path, evaluation):
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump(
            {"schema_version": 1, "benchmark": "libero", "evaluation": evaluation}
        )
    )
    with pytest.raises((ValueError, FileNotFoundError)):
        configuration.load_benchmark("libero", path=path)


def test_python_selector_can_only_choose_existing_cells(tmp_path):
    from roborsi.evaluation.batch import BatchCase
    from scripts.evaluation.selection import select_cells

    code = tmp_path / "subset.py"
    code.write_text(
        "def select(cases, *, benchmark, parameters):\n    return [c for c in cases if c['seed'] == parameters['seed']]\n"
    )
    path = tmp_path / "tasks.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "benchmark": "libero",
                "evaluation": {
                    "selector": "subset.py:select",
                    "parameters": {"seed": 1},
                },
            }
        )
    )
    config = configuration.load_benchmark("libero", path=path)
    cases = [BatchCase("libero", "pick", 0), BatchCase("libero", "pick", 1)]
    assert select_cells(config, cases) == [cases[1]]
    code.write_text(
        "def select(cases, **kwargs):\n    return [{'task_id':'unknown', 'seed':99}]\n"
    )
    with pytest.raises(ValueError, match="catalog"):
        select_cells(config, cases)


def test_model_path_can_be_scoped_to_a_benchmark(tmp_path, monkeypatch):
    path = tmp_path / "model.yaml"
    path.write_text("schema_version: 1\nprovider: openai\nmodel_name: selected\n")
    module = runner(monkeypatch)
    _, args = module.parse_args(
        [
            "--benchmark",
            "libero",
            "--benchmark",
            "libero_pro",
            "--prepare-only",
            "--model-config",
            f"libero_pro={path}",
        ]
    )
    assert args.models["libero_pro"].model_name == "selected"
    assert args.configurations["libero"].model != path


def test_different_selected_seed_sets_do_not_expand_cross_product(
    tmp_path, monkeypatch
):
    from tests.unit_tests.scripts.test_evaluate_model import (
        registry as registry_fixture,
    )

    registry = registry_fixture.__wrapped__(tmp_path)
    code = tmp_path / "subset.py"
    code.write_text(
        "def select(cases, **kwargs):\n    return [c for c in cases if c['seed'] == (2 if c['task_id'] == 'atomic_task' else 1)]\n"
    )
    path = tmp_path / "tasks.yaml"
    path.write_text(
        "schema_version: 1\nbenchmark: robocasa365\nevaluation:\n  selector: subset.py:select\n"
    )
    module = runner(monkeypatch)
    _, args = module.parse_args(
        [
            "--benchmark",
            "robocasa365",
            "--benchmark-config",
            str(path),
            "--registry-root",
            str(registry),
            "--base-url",
            "http://model.test/v1",
            "--model-name",
            "own",
            "--dry-run",
        ]
    )
    commands = module.build_commands(args)
    selected = [(c[c.index("--task") + 1], c[c.index("--seed") + 1]) for c in commands]
    assert sorted(selected) == [
        ("robocasa365:atomic_task", "2"),
        ("robocasa365:composite_task", "1"),
    ]


def test_selector_can_import_sibling_python_module(tmp_path):
    from roborsi.evaluation.batch import BatchCase
    from scripts.evaluation.selection import select_cells

    (tmp_path / "my_subset_parameters.py").write_text("SEED = 1\n")
    (tmp_path / "selector.py").write_text(
        "from my_subset_parameters import SEED\ndef select(cases, **kwargs):\n    return [c for c in cases if c['seed'] == SEED]\n"
    )
    path = tmp_path / "suite.yaml"
    path.write_text(
        "schema_version: 1\nbenchmark: libero\nevaluation:\n  selector: selector.py:select\n"
    )
    cases = [BatchCase("libero", "pick", 0), BatchCase("libero", "pick", 1)]
    assert select_cells(configuration.load_benchmark("libero", path=path), cases) == [
        cases[1]
    ]
