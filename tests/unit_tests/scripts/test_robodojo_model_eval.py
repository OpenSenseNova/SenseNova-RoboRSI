# Copyright 2026 The RoboRSI Contributors.
# Licensed under the Apache License, Version 2.0.

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from roborsi.agents.config import AgentConfig
from scripts.evaluation import bootstrap, evaluate_model, preflight
from scripts.evaluation.configuration import load_benchmark
from scripts.evaluation.robodojo import description_parameters
from tests.unit_tests.roborsi.benchmarks.test_registry_contracts import _manifest
from tests.unit_tests.scripts.test_evaluate_model import evaluation_report


def test_track_http_plan_is_offline_and_never_prepares_icl(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    assert (
        evaluate_model.main(
            [
                "--benchmark",
                "robodojo",
                "--benchmark-url",
                "robodojo=http://127.0.0.1:9",
                "--dry-run",
                "--work-dir",
                str(tmp_path / "absent"),
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["network_requests"] is False
    assert plan["icl"]["robodojo"] == "not used (RoboDojo Agent Track)"
    assert plan["selection"]["robodojo"]["seeds"] == [0]
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize(
    "options",
    [
        [],
        ["--openai-format", "chat"],
        ["--max-tokens", "10"],
    ],
)
def test_one_command_rejects_missing_native_source_or_wrong_track(options):
    source = [] if not options else ["--benchmark-url", "robodojo=http://127.0.0.1:9"]
    with pytest.raises(SystemExit):
        evaluate_model.parse_args(
            ["--benchmark", "robodojo", "--dry-run", *source, *options]
        )


def test_legacy_wiki_bundle_cannot_replace_canonical_public_scoring(tmp_path):
    raw = json.dumps(
        {
            "schema_version": 2,
            "origin": "https://robodojo-benchmark.com/doc/sim-tasks/",
            "tasks": {"cover_blocks": {}},
        }
    ).encode()
    (tmp_path / "MANIFEST.json").write_bytes(raw)
    config = replace(
        load_benchmark("robodojo"),
        agent_overrides={
            "parameters": {
                "official_description_bundle": str(tmp_path),
                "official_description_manifest_sha256": hashlib.sha256(raw).hexdigest(),
            }
        },
    )
    with pytest.raises(ValueError, match="canonical descriptions"):
        description_parameters(config, tmp_path)


def test_automatic_descriptions_are_verified_and_reused_without_icl(
    tmp_path, monkeypatch
):
    from benchmarks.robodojo import official_descriptions

    config = load_benchmark("robodojo")
    root = Path(evaluate_model.ROOT)
    tasks = json.loads(
        (
            root / "robots/robodojo/agents/submission/skillpack/task_catalog.json"
        ).read_text()
    )["tasks"]
    calls = []
    monkeypatch.setattr(
        official_descriptions,
        "download_source",
        lambda: (
            calls.append("official") or "fixture",
            "https://robodojo-benchmark.com/assets/fixture.js",
        ),
    )
    monkeypatch.setattr(
        official_descriptions,
        "specifications",
        lambda _: {
            task: {
                "description": "Current official objective.",
                "scoring": ["Success: 1 point."],
            }
            for task in tasks
        },
    )
    monkeypatch.setattr(
        bootstrap, "download_snapshot", lambda **kw: pytest.fail("ICL download")
    )
    prepared = bootstrap.prepare_agents(["robodojo"], tmp_path)
    selected = AgentConfig.from_file(prepared.presets["robodojo"])
    assert selected.parameters["skill_library_mode"] == "general_only"
    assert selected.parameters["eef_execution_backend_by_task"] == {}
    assert selected.parameters["vla_backend"] == "simate_beta"
    assert selected.parameters["provider_kind"] == "responses"
    assert prepared.vla == {} and calls == ["official"]
    parameters = description_parameters(config, tmp_path)
    assert (
        parameters["official_description_manifest_sha256"]
        == selected.parameters["official_description_manifest_sha256"]
    )
    assert calls == ["official"]
    bundle = Path(parameters["official_description_bundle"])
    (bundle / next(iter(tasks))).with_suffix(".txt").write_text("tampered")
    with pytest.raises(ValueError, match="description text changed"):
        description_parameters(config, tmp_path)


def test_execution_binds_submitted_model_environment_without_generic_overrides(
    tmp_path, monkeypatch
):
    source, asset = _manifest(tmp_path / "template")
    document = json.loads(source.read_text())
    document["benchmark"] = "robodojo"
    document["tasks"][0]["seeds"] = [0]
    root = tmp_path / "registry/robodojo"
    (root / "assets").mkdir(parents=True)
    (root / "assets/tasks.bin").write_bytes(asset.read_bytes())
    (root / "manifest.json").write_text(json.dumps(document))
    config = load_benchmark("robodojo")
    monkeypatch.setattr(preflight, "check", lambda _: None)
    monkeypatch.setattr(
        bootstrap,
        "prepare_agents",
        lambda names, work, **kw: bootstrap.Prepared(
            tuple(names), work, {"robodojo": str(config.agent)}, {}, {}
        ),
    )
    monkeypatch.setenv("ROBORSI_FULL_ICL_BASELINE_ROOT", "/stale-corpus")
    monkeypatch.setenv("ROBORSI_DESCRIPTION_BUNDLE", "/stale-description")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs["env"]))
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(evaluation_report(tmp_path / "registry", "robodojo", "success")),
            "",
        )

    monkeypatch.setattr(evaluate_model, "run_child", run)
    assert (
        evaluate_model.main(
            [
                "--benchmark",
                "robodojo",
                "--registry-root",
                str(tmp_path / "registry"),
                "--base-url",
                "http://model.invalid/v1",
                "--model-name",
                "fixture-model",
                "--api-key",
                "unit-secret",
                "--run-id",
                "track",
                "--work-dir",
                str(tmp_path / "work"),
            ]
        )
        == 0
    )
    assert len(calls) == 2
    for command, env in calls:
        assert not any(
            item.startswith(("runtime.llm.", "agent.max_", "agent.planner_"))
            for item in command
        )
        assert "unit-secret" not in " ".join(command)
        assert env["CODEX_API_KEY"] == "unit-secret"
        assert env["ROBORSI_CODEX_BASE_URL"] == "http://model.invalid/v1"
        assert env["ROBORSI_CODEX_MODEL"] == "fixture-model"
        assert env["ROBORSI_SKILL_LIBRARY_MODE"] == "general_only"
        assert (
            "ROBORSI_FULL_ICL_BASELINE_ROOT" not in env
            and "ROBORSI_DESCRIPTION_BUNDLE" not in env
        )
