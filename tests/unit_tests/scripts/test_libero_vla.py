# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""VLA opt-in owns downloads and services without changing benchmark execution."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]


def models(*names):
    from scripts.evaluation.configuration import resolve_model

    return {name: resolve_model(ROOT / "configs/models/default.yaml") for name in names}


def default_profile():
    return models("libero")["libero"].vla


def preset(tmp_path, *, enabled=True, **parameters):
    path = tmp_path / "agent.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_root": os.path.commonpath([str(ROOT), str(tmp_path)]),
                "extends": str(ROOT / "configs/agents/libero/standard.yaml"),
                "parameters": {"enable_vla": enabled, **parameters},
            }
        )
    )
    return str(path)


def test_disabled_vla_has_no_download_or_docker_side_effect(tmp_path, monkeypatch):
    from scripts.evaluation import vla

    def forbidden(*args, **kwargs):
        pytest.fail("disabled VLA attempted setup")

    monkeypatch.setattr(vla, "_run", forbidden)
    destination = tmp_path / "not-created"
    assert (
        vla.prepare_agents({"libero": preset(tmp_path, enabled=False)}, destination)
        == {}
    )
    assert not destination.exists()


def test_existing_vla_endpoint_skips_build_and_download(tmp_path, monkeypatch):
    from scripts.evaluation import vla

    monkeypatch.setenv("ROBORSI_LIBERO_VLA_URL", "http://127.0.0.1:32100")
    monkeypatch.setattr(
        vla,
        "_run",
        lambda *args, **kwargs: pytest.fail("external service triggered Docker"),
    )
    prepared = vla.prepare_agents(
        {"libero": preset(tmp_path)}, tmp_path / "cache", models=models("libero")
    )
    assert prepared["libero"].endpoint == "http://127.0.0.1:32100"
    assert not (tmp_path / "cache").exists()


def test_external_mcp_remains_an_independent_opt_in(tmp_path, monkeypatch):
    from scripts.evaluation import vla

    monkeypatch.setenv("ROBORSI_VLA_MCP_URL", "http://localhost:32101/mcp")
    monkeypatch.setattr(
        vla, "_run", lambda *args, **kwargs: pytest.fail("MCP triggered OpenPI setup")
    )
    assert (
        vla.prepare_agents({"libero_pro": preset(tmp_path)}, tmp_path / "cache") == {}
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("openpi_revision", "main"),
        ("action_steps", 0),
        ("request_timeout_s", True),
        ("unknown", 1),
    ],
)
def test_invalid_profile_is_rejected_before_commands(tmp_path, field, value):
    from scripts.evaluation.configuration import resolve_model

    document = yaml.safe_load((ROOT / "configs/models/default.yaml").read_text())
    document["vla"][field] = value
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError):
        resolve_model(path)


def test_preparation_builds_once_and_downloads_without_gpu(tmp_path, monkeypatch):
    from scripts.evaluation import downloads, vla

    calls = []
    monkeypatch.delenv("ROBORSI_LIBERO_VLA_URL", raising=False)
    monkeypatch.delenv("ROBORSI_VLA_MCP_URL", raising=False)

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(downloads.subprocess, "run", run)
    monkeypatch.setattr(vla, "_ensure_image", lambda profile: "pinned-openpi-image")
    path = preset(tmp_path)
    prepared = vla.prepare_agents(
        {"libero": path, "libero_pro": path},
        tmp_path / "cache",
        models=models("libero", "libero_pro"),
    )
    starts = [c for c in calls if c[:2] == ["docker", "run"]]
    assert len(starts) == 1
    assert "/app/scripts/evaluation/downloads.py" in starts[0]
    assert "openpi" in starts[0]
    assert "--gpus" not in starts[0]
    assert "gs://openpi-assets/checkpoints/pi05_libero" in starts[0]
    assert prepared["libero"] is prepared["libero_pro"]


def test_shared_service_is_ready_before_binding_and_removed_on_failure(
    tmp_path, monkeypatch
):
    from roborsi.agents.config import AgentConfig
    from scripts.evaluation import vla

    profile = default_profile()
    deployment = vla.Deployment(profile, "openpi-image", tmp_path / "cache")
    calls, ready = [], []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="127.0.0.1:32123" if command[1] == "port" else "",
            stderr="",
        )

    monkeypatch.setattr(vla, "_run", run)
    monkeypatch.setattr(vla, "_wait", lambda *args: ready.append(args[0]))
    path = preset(tmp_path)
    with pytest.raises(RuntimeError, match="failed evaluation"):
        with vla.services(
            {"libero": deployment, "libero_pro": deployment},
            {"libero": path, "libero_pro": path},
            tmp_path,
        ) as bound:
            assert ready == ["http://127.0.0.1:32123"]
            assert (
                os.environ[
                    AgentConfig.from_file(bound["libero"]).parameters[
                        "vla_endpoint_env"
                    ]
                ]
                == "http://127.0.0.1:32123"
            )
            assert (
                AgentConfig.from_file(bound["libero"]).parameters["vla_model"][
                    "checkpoint"
                ]
                == profile.checkpoint
            )
            raise RuntimeError("failed evaluation")
    starts = [c for c in calls if c[:2] == ["docker", "run"]]
    assert len(starts) == 1
    assert "127.0.0.1::8000" in starts[0]
    assert starts[0][starts[0].index("--gpus") + 1] == "all"
    name = starts[0][starts[0].index("--name") + 1]
    assert calls[-1] == ["docker", "rm", "--force", name]


def test_readiness_failure_still_cleans_owned_service(tmp_path, monkeypatch):
    from scripts.evaluation import vla

    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command, 0, stdout="127.0.0.1:32123", stderr=""
        )

    monkeypatch.setattr(vla, "_run", run)
    monkeypatch.setattr(
        vla, "_wait", lambda *args: (_ for _ in ()).throw(RuntimeError("not ready"))
    )
    deployment = vla.Deployment(default_profile(), "image", tmp_path)
    with pytest.raises(RuntimeError, match="not ready"):
        with vla.services(
            {"libero": deployment}, {"libero": preset(tmp_path)}, tmp_path
        ):
            pytest.fail("evaluation ran before readiness")
    assert calls[-1][:3] == ["docker", "rm", "--force"]
    container = calls[-1][-1]
    assert calls[-2] == ["docker", "logs", "--timestamps", container]
    assert (tmp_path / "logs/services" / f"{container}.log").is_file()


def test_benchmark_override_can_enable_vla_before_any_download(tmp_path, monkeypatch):
    from dataclasses import replace

    from scripts.evaluation import bootstrap
    from scripts.evaluation.configuration import load_benchmark

    config = replace(
        load_benchmark("libero"), agent_overrides={"parameters": {"enable_vla": True}}
    )

    def forbidden(*args, **kwargs):
        pytest.fail("planning ran a command")

    monkeypatch.setattr(bootstrap, "_run", forbidden)
    result = bootstrap.preparation_plan(
        ["libero"], tmp_path / "missing", configurations={"libero": config}
    )
    assert result["vla"]["libero"]["policy_config"] == "pi05_libero"
    assert result["vla"]["libero"]["mode"] == "managed"
    assert not (tmp_path / "missing").exists()


def test_selected_agent_drives_automatic_preparation(tmp_path, monkeypatch):
    from dataclasses import replace

    from roborsi.agents.config import AgentConfig
    from scripts.evaluation import bootstrap
    from scripts.evaluation.configuration import load_benchmark

    config = replace(
        load_benchmark("libero"), agent=Path(preset(tmp_path)), agent_overrides={}
    )
    monkeypatch.setenv("ROBORSI_LIBERO_VLA_URL", "http://localhost:32104")
    monkeypatch.delenv("ROBORSI_VLA_MCP_URL", raising=False)
    monkeypatch.setattr(bootstrap, "_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(bootstrap, "_build_image", lambda *args: "native-image")
    monkeypatch.setattr(
        bootstrap, "_prepare_standard_assets", lambda *args: tmp_path / "assets"
    )
    prepared = bootstrap.prepare_benchmarks(
        ["libero"], tmp_path / "work", configurations={"libero": config}
    )
    assert prepared.vla["libero"].endpoint == "http://localhost:32104"
    assert (
        AgentConfig.from_file(prepared.presets["libero"]).parameters["enable_vla"]
        is True
    )


def test_openpi_commit_changes_image_identity_and_build_input(monkeypatch):
    from dataclasses import replace

    from scripts.evaluation import vla

    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            command, 1 if command[1:3] == ["image", "inspect"] else 0
        )

    monkeypatch.setattr(vla, "_run", run)
    profile = default_profile()
    assert vla._ensure_image(profile) != vla._ensure_image(
        replace(profile, openpi_revision="a" * 40)
    )
    builds = [command for command in commands if command[1] == "build"]
    assert (
        builds[0][builds[0].index("--build-arg") + 1]
        == "OPENPI_REVISION=215abfb217dbac7d5f1273282331b9b1866c0479"
    )
    assert (
        builds[1][builds[1].index("--build-arg") + 1] == "OPENPI_REVISION=" + "a" * 40
    )
