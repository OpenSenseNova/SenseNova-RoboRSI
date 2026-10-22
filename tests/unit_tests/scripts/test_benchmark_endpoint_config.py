# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Benchmark YAML can select an already running image's HTTP service."""

import importlib.util
from pathlib import Path

import pytest
import yaml

from scripts.evaluation.configuration import load_benchmark

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def runner(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "endpoint_eval", ROOT / "scripts/evaluation/evaluate_model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "IS_WINDOWS", False)
    return module


def arguments(tmp_path, endpoint="http://localhost:18800"):
    path = tmp_path / "benchmark.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "benchmark": "libero",
                "endpoint": endpoint,
            }
        )
    )
    return [
        "--benchmark",
        "libero",
        "--benchmark-config",
        str(path),
        "--base-url",
        "http://localhost:8000/v1",
        "--model-name",
        "own-model",
        "--dry-run",
    ]


def test_configured_endpoint_uses_external_http_without_preparation(
    tmp_path, runner, monkeypatch
):
    argsv = arguments(tmp_path)
    _, args = runner.parse_args(argsv)
    assert args.endpoints == {"libero": "http://localhost:18800"}
    assert not args.auto_prepare
    monkeypatch.setattr(
        runner,
        "run_evaluation",
        lambda parser, args: pytest.fail("executed offline plan"),
    )
    assert runner.main(argsv) == 0


def test_cli_benchmark_source_overrides_yaml(tmp_path, runner):
    argsv = arguments(tmp_path)
    _, args = runner.parse_args(
        argsv + ["--benchmark-url", "libero=http://localhost:19900"]
    )
    assert args.endpoints == {"libero": "http://localhost:19900"}
    _, args = runner.parse_args(argsv + ["--registry-root", str(tmp_path)])
    assert args.endpoints == {}
    assert args.registry_root == tmp_path


def test_null_endpoint_keeps_automatic_environment(tmp_path, runner):
    _, args = runner.parse_args(arguments(tmp_path, None))
    assert args.auto_prepare
    assert args.endpoints == {}
    assert load_benchmark("libero").endpoint is None


@pytest.mark.parametrize(
    "url", ["file:///tmp/service", "http://user:password@host", "http://host/v1"]
)
def test_bad_endpoint_fails_before_contacting_service(tmp_path, runner, url):
    with pytest.raises(SystemExit):
        runner.parse_args(arguments(tmp_path, url))


def test_configured_http_reuses_token_environment(tmp_path, runner, monkeypatch):
    monkeypatch.setenv("BENCHMARK_TOKEN", "fixture-token")
    _, args = runner.parse_args(
        arguments(tmp_path) + ["--benchmark-token-env", "BENCHMARK_TOKEN"]
    )
    assert args.benchmark_token_env == "BENCHMARK_TOKEN"
