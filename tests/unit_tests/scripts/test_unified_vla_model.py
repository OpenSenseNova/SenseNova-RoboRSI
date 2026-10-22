# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""One model file selects both planning and execution models."""

from pathlib import Path

import pytest
import yaml

from scripts.evaluation import configuration, vla

ROOT = Path(__file__).resolve().parents[3]


def model_file(tmp_path, **vla_changes):
    document = yaml.safe_load((ROOT / "configs/models/default.yaml").read_text())
    document["base_url"] = "http://planner.test/v1"
    document["model_name"] = "own-planner"
    document["vla"] = {
        "backend": "openpi",
        "openpi_revision": "a" * 40,
        "policy_config": "custom_libero",
        "checkpoint": "gs://fixture/custom-checkpoint",
        "action_steps": 2,
        **vla_changes,
    }
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(document))
    return path


def test_model_file_resolves_primary_and_vla_together(tmp_path):
    selected = configuration.resolve_model(model_file(tmp_path))
    assert selected.model_name == "own-planner"
    assert selected.vla.policy_config == "custom_libero"
    assert selected.vla.action_steps == 2


def test_vla_local_checkpoint_resolves_relative_to_model_file(tmp_path):
    (tmp_path / "weights").mkdir()
    selected = configuration.resolve_model(model_file(tmp_path, checkpoint="weights"))
    assert selected.vla.checkpoint == str((tmp_path / "weights").resolve())


def test_selected_model_drives_plan_without_agent_model_resource(tmp_path):
    agent = ROOT / "configs/agents/libero/pro_vla.yaml"
    selected = configuration.resolve_model(model_file(tmp_path))
    plan = vla.plan({"libero_pro": agent}, models={"libero_pro": selected})
    assert plan["libero_pro"]["checkpoint"] == "gs://fixture/custom-checkpoint"
    assert plan["libero_pro"]["openpi_revision"] == "a" * 40
    from roborsi.agents.config import AgentConfig

    assert "vla_profile" not in AgentConfig.from_file(agent).resources


def test_enabled_agent_requires_vla_in_selected_model_before_commands(
    tmp_path, monkeypatch
):
    path = tmp_path / "model.yaml"
    path.write_text("schema_version: 1\nprovider: openai\n")
    monkeypatch.setattr(vla, "_run", lambda *a, **k: pytest.fail("ran Docker"))
    with pytest.raises(ValueError, match="vla"):
        vla.prepare_agents(
            {"libero_pro": ROOT / "configs/agents/libero/pro_vla.yaml"},
            tmp_path / "not-created",
            models={"libero_pro": configuration.resolve_model(path)},
        )
    assert not (tmp_path / "not-created").exists()


def test_server_requires_prepared_checkpoint_before_importing_openpi(
    tmp_path, monkeypatch
):
    from robots.libero.vla_server import main

    monkeypatch.setenv("OPENPI_DATA_HOME", str(tmp_path))
    with pytest.raises(ValueError, match="download"):
        main(["--checkpoint", "gs://fixture/missing"])


def test_cli_model_path_drives_vla_preparation_plan(tmp_path, monkeypatch):
    import importlib.util

    from scripts.evaluation import bootstrap

    spec = importlib.util.spec_from_file_location(
        "unified_model_eval", ROOT / "scripts/evaluation/evaluate_model.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "IS_WINDOWS", False)
    _, args = runner.parse_args(
        [
            "--benchmark",
            "libero_pro",
            "--dry-run",
            "--agent-config",
            str(ROOT / "configs/agents/libero/pro_vla.yaml"),
            "--model-config",
            str(model_file(tmp_path)),
        ]
    )
    plan = bootstrap.preparation_plan(
        args.benchmark, tmp_path / "missing", configurations=args.configurations
    )
    assert args.models["libero_pro"].model_name == "own-planner"
    assert plan["vla"]["libero_pro"]["policy_config"] == "custom_libero"
    assert not (tmp_path / "missing").exists()


def test_benchmarks_can_select_independent_vla_models(tmp_path):
    from dataclasses import replace

    selected = configuration.resolve_model(model_file(tmp_path))
    second = replace(selected, vla=replace(selected.vla, action_steps=3))
    agent = ROOT / "configs/agents/libero/pro_vla.yaml"
    plan = vla.plan(
        {"libero_pro": agent, "libero_long_plus_plus": agent},
        models={"libero_pro": selected, "libero_long_plus_plus": second},
    )
    assert plan["libero_pro"]["action_steps"] == 2
    assert plan["libero_long_plus_plus"]["action_steps"] == 3


def test_download_failure_cleans_only_owned_container_and_keeps_cache(
    tmp_path, monkeypatch
):
    import subprocess

    from scripts.evaluation import downloads

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "existing").write_bytes(b"keep")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[1] == "run":
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(downloads.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        downloads.prepare_openpi("image", cache, "gs://fixture/weights")
    container = calls[0][calls[0].index("--name") + 1]
    assert calls[1] == ["docker", "rm", "--force", container]
    assert (cache / "existing").read_bytes() == b"keep"


def test_serving_resolves_prepared_cache_without_download_imports(
    tmp_path, monkeypatch
):
    from scripts.evaluation.downloads import prepared_openpi_checkpoint

    monkeypatch.setenv("OPENPI_DATA_HOME", str(tmp_path))
    checkpoint = tmp_path / "fixture/weights"
    (checkpoint / "assets").mkdir(parents=True)
    (checkpoint / "params").mkdir()
    tokenizer = tmp_path / "big_vision/paligemma_tokenizer.model"
    tokenizer.parent.mkdir()
    with pytest.raises(ValueError, match="tokenizer.*download"):
        prepared_openpi_checkpoint("gs://fixture/weights")
    tokenizer.write_bytes(b"cached-tokenizer")
    assert prepared_openpi_checkpoint("gs://fixture/weights") == checkpoint
