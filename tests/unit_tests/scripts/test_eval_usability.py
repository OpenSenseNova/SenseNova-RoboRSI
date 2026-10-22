# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""End-user contracts for the one-command evaluation workflow."""

import importlib.util
import json
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts.evaluation import bootstrap, configuration

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def runner(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "ux_eval", ROOT / "scripts/evaluation/evaluate_model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "IS_WINDOWS", False)
    return module


def test_model_owns_existing_vla_endpoint(tmp_path):
    document = yaml.safe_load((ROOT / "configs/models/default.yaml").read_text())
    document["vla"]["endpoint"] = "http://localhost:19000"
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(document))
    assert configuration.resolve_model(path).vla.endpoint == "http://localhost:19000"


def test_benchmark_owns_token_variable(tmp_path):
    path = tmp_path / "benchmark.yaml"
    path.write_text(
        "schema_version: 1\nbenchmark: libero\nendpoint: http://localhost:9\ntoken_env: BENCH_TOKEN\n"
    )
    assert configuration.load_benchmark("libero", path=path).token_env == "BENCH_TOKEN"


@pytest.mark.parametrize("external", [False, True])
def test_dry_run_is_offline_and_needs_no_model_or_credentials(
    runner, tmp_path, monkeypatch, capsys, external
):
    import platform

    platform.uname()  # Initialize Python's Windows platform cache before guarding subprocesses.
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.setattr(
        runner, "build_commands", lambda *a: pytest.fail("contacted catalog")
    )
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: pytest.fail("started subprocess")
    )
    args = ["--benchmark", "libero", "--work-dir", str(tmp_path / "new"), "--dry-run"]
    if external:
        args += [
            "--benchmark-url",
            "libero=http://localhost:9",
            "--benchmark-token-env",
            "UNSET_TOKEN",
        ]
    assert runner.main(args) == 0
    assert json.loads(capsys.readouterr().out)["network_requests"] is False
    assert not (tmp_path / "new").exists()


def test_http_prepares_robocasa_icl_without_simulator_build(tmp_path, monkeypatch):
    package = tmp_path / "package"
    package.mkdir()
    (package / "fixture").write_text("fixture")
    config = replace(
        configuration.load_benchmark("robocasa365"), endpoint="http://localhost:9"
    )
    monkeypatch.setattr(bootstrap, "download_snapshot", lambda **k: package)
    monkeypatch.setattr(
        bootstrap, "_build_image", lambda *a: pytest.fail("built simulator")
    )
    monkeypatch.setattr(bootstrap, "_run", lambda *a, **k: pytest.fail("ran Docker"))
    prepared = bootstrap.prepare_agents(
        ["robocasa365"], tmp_path / "work", configurations={"robocasa365": config}
    )
    from roborsi.agents.config import AgentConfig

    assert (
        AgentConfig.from_file(prepared.presets["robocasa365"]).resources["icl_package"]
        == package
    )
    assert prepared.images == {}


def test_streaming_child_redacts_output_and_log(tmp_path, capsys):
    from scripts.evaluation.reporting import run_child

    path = tmp_path / "runner.log"
    result = run_child(
        [
            sys.executable,
            "-u",
            "-c",
            "import sys; print('stage secret'); print('error secret',file=sys.stderr)",
        ],
        cwd=ROOT,
        env=None,
        log_path=path,
        redact=lambda s: s.replace("secret", "[REDACTED]"),
    )
    output = capsys.readouterr()
    assert "stage [REDACTED]" in output.out
    assert "error [REDACTED]" in output.err
    assert "secret" not in path.read_text()
    assert result.returncode == 0


def test_child_log_is_written_before_child_exits(tmp_path):
    from scripts.evaluation.reporting import run_child

    path, gate = tmp_path / "live.log", tmp_path / "continue"
    results = []
    child = """import pathlib, sys, time
print('starting episode', flush=True)
deadline = time.monotonic() + 5
while not pathlib.Path(sys.argv[1]).exists() and time.monotonic() < deadline:
    time.sleep(0.01)
"""
    worker = threading.Thread(
        target=lambda: results.append(
            run_child(
                [sys.executable, "-u", "-c", child, str(gate)],
                cwd=ROOT,
                env=None,
                log_path=path,
                redact=lambda value: value,
            )
        )
    )
    worker.start()
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if path.exists() and "starting episode" in path.read_text():
                break
            time.sleep(0.01)
        assert "starting episode" in path.read_text()
        assert worker.is_alive(), "output was buffered until child exit"
    finally:
        gate.touch()
        worker.join(timeout=10)
    assert len(results) == 1 and results[0].returncode == 0


def test_resume_requires_new_invocation_id(runner):
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--benchmark",
                "libero",
                "--run-id",
                "same",
                "--resume-from",
                "same",
                "--dry-run",
            ]
        )


def test_failed_api_check_precedes_all_downloads(runner, tmp_path, monkeypatch):
    from scripts.evaluation import preflight

    monkeypatch.setattr(
        preflight,
        "check",
        lambda args: (_ for _ in ()).throw(RuntimeError("model API unavailable")),
    )
    monkeypatch.setattr(
        runner, "load_bootstrap", lambda: pytest.fail("started preparing")
    )
    with pytest.raises(SystemExit):
        runner.main(
            [
                "--benchmark",
                "libero",
                "--base-url",
                "http://localhost:9/v1",
                "--model-name",
                "model",
                "--api-key",
                "EMPTY",
                "--work-dir",
                str(tmp_path),
            ]
        )


@pytest.mark.parametrize(
    "override, expected", [({}, "responses"), ({"openai_format": "chat"}, "chat")]
)
def test_preflight_custom_agent_can_omit_model_config(tmp_path, override, expected):
    from scripts.evaluation.preflight import model_format

    agent = tmp_path / "agent.yaml"
    agent.write_text("schema_version: 1\n")
    args = SimpleNamespace(
        models={"libero": SimpleNamespace(openai_format=None)},
        configurations={
            "libero": SimpleNamespace(
                agent=agent, agent_overrides={"runtime": {"llm": override}}
            )
        },
    )
    assert model_format(args, "libero") == expected


def test_duplicate_run_id_does_not_change_previous_results(runner, tmp_path):
    directory = tmp_path / "existing"
    directory.mkdir()
    original = directory / "summary.json"
    original.write_text('{"complete": true}')
    with pytest.raises(SystemExit):
        runner.main(
            [
                "--benchmark",
                "libero",
                "--base-url",
                "http://localhost:9/v1",
                "--model-name",
                "fixture",
                "--api-key",
                "EMPTY",
                "--results-root",
                str(tmp_path),
                "--run-id",
                "existing",
            ]
        )
    assert list(directory.iterdir()) == [original]
    assert original.read_text() == '{"complete": true}'
