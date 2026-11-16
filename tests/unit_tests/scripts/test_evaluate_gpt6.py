# Copyright 2026 The SenseNova-RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Offline user contracts for the GPT-6 suite entry point."""

import json
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from scripts.evaluation import evaluate_gpt6 as suite
from scripts.evaluation import reporting


def arguments(tmp_path, names):
    args = ["--work-dir", str(tmp_path / "work"), "--run-id", "trial"]
    for name in names:
        args += ["--benchmark", name]
    return args


def jobs_for(tmp_path, names):
    args = SimpleNamespace(work_dir=tmp_path, benchmark=names, run_id="trial")
    return suite.build_jobs(
        suite.load_config(suite.DEFAULT_CONFIG), suite.DEFAULT_CONFIG, args
    )[1]


def test_default_plan_covers_every_release_offline(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("MODEL_NAME", "sensenova-roborsi-27b-preview")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://deployed-model.invalid/v1")
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: pytest.fail("started subprocess")
    )
    monkeypatch.setattr(
        reporting, "run_child", lambda *a, **k: pytest.fail("contacted environment")
    )
    assert suite.main(arguments(tmp_path, []) + ["--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["network_requests"] is False
    assert len(plan["jobs"]) == 12
    assert len({job["benchmark"] for job in plan["jobs"]}) == 11
    assert {job["name"] for job in plan["jobs"]} == set(suite.COHORTS)
    assert not (tmp_path / "work").exists()
    for job in plan["jobs"][:5]:
        command = job["command"]
        assert command[command.index("--model-name") + 1] == "gpt-6-astra"
        assert "--agent-config" in command
        assert "deployed-model.invalid" not in " ".join(command)


def test_task_specific_icl_and_navigation_cohorts_remain_separate(tmp_path):
    jobs = {job.name: job for job in jobs_for(tmp_path, None)}
    bench2dex = jobs["bench2dex"].command
    assert "benchmarks.bench2dex.run_suite" in bench2dex
    assert "--plan" in bench2dex and "--icl-source" in bench2dex
    assert "--agent-config" not in bench2dex  # The existing suite binds each task.
    behavior = jobs["behavior_1k"].command
    assert behavior[behavior.index("--task") + 1] == "behavior_1k:sorting_vegetables"
    assert "--seed" in behavior
    r2r, rxr = jobs["vlnce_r2r"].command, jobs["vlnce_rxr"].command
    assert (
        r2r[r2r.index("--benchmark") + 1]
        == rxr[rxr.index("--benchmark") + 1]
        == "vlnce"
    )
    assert (
        r2r[r2r.index("--registry-root") + 1] != rxr[rxr.index("--registry-root") + 1]
    )
    assert r2r[r2r.index("--run-id") + 1] != rxr[rxr.index("--run-id") + 1]


@pytest.mark.parametrize("name", suite.COHORTS[:5])
def test_managed_jobs_use_real_presets_and_an_independent_model_profile(
    tmp_path, monkeypatch, capsys, name
):
    from scripts.evaluation import evaluate_model

    monkeypatch.setenv("MODEL_NAME", "sensenova-roborsi-27b-preview")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://deployed-model.invalid/v1")
    monkeypatch.delenv("GPT6_BASE_URL", raising=False)
    monkeypatch.setenv("GPT6_MODEL_NAME", "ignored-deployed-model")
    monkeypatch.setattr(
        evaluate_model, "run_evaluation", lambda *a: pytest.fail("started evaluation")
    )
    job = jobs_for(tmp_path, [name])[0]
    assert evaluate_model.main([*job.command[2:], "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["models"][name]["model_name"] == "gpt-6-astra"
    assert plan["models"][name]["base_url"] == "https://api.openai.com/v1"
    assert (
        plan["configurations"][name]["agent"]
        == job.command[job.command.index("--agent-config") + 1]
    )


def test_paths_and_cli_versions_are_per_cohort(tmp_path):
    config = suite.load_config(suite.DEFAULT_CONFIG)
    config["benchmarks"]["robolab"].update(
        codex="codex-160/codex", python="agent-env/bin/python"
    )
    config["benchmarks"]["behavior_1k"]["codex"] = "codex-162/codex"
    args = SimpleNamespace(
        work_dir=tmp_path / "work", benchmark=["robolab", "behavior_1k"], run_id="trial"
    )
    _, jobs = suite.build_jobs(config, tmp_path / "config.yaml", args)
    robolab, behavior = jobs
    assert robolab.command[0] == str(tmp_path / "agent-env/bin/python")
    assert robolab.env["PATH"].startswith(str(tmp_path / "codex-160"))
    assert behavior.env["PATH"].startswith(str(tmp_path / "codex-162"))
    assert "--agent-set" in robolab.command
    assert "--agent-set" not in behavior.command  # Retain the MCP approval wrapper.


@pytest.mark.parametrize(
    "entry,update",
    [
        ("robolab", {"registry_root": "registry"}),
        ("bench2dex", {"tasks": ["26"]}),
        ("libero", {"seeds": [True]}),
        ("libero", {"tasks": ["pick", "pick"]}),
        ("libero", {"case_timeout_s": float("nan")}),
    ],
)
def test_invalid_selection_is_rejected_before_execution(tmp_path, entry, update):
    config = suite.load_config(suite.DEFAULT_CONFIG)
    config["benchmarks"][entry].update(update)
    args = SimpleNamespace(work_dir=tmp_path, benchmark=[entry], run_id="trial")
    with pytest.raises(ValueError):
        suite.build_jobs(config, suite.DEFAULT_CONFIG, args)


def test_preflight_failure_blocks_every_model_evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr(suite, "IS_WINDOWS", False)
    monkeypatch.setattr(suite, "check_inputs", lambda job: None)
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        assert "--dry-run" in command
        failed = command[command.index("--benchmark") + 1] == "robolab"
        return subprocess.CompletedProcess(command, int(failed), "", "")

    monkeypatch.setattr(reporting, "run_child", execute)
    assert suite.main(arguments(tmp_path, ["libero", "robolab"])) == 2
    assert len(calls) == 2
    report = json.loads((tmp_path / "work/results/trial/summary.json").read_text())
    assert report["complete"] is False
    assert [job["status"] for job in report["jobs"]] == ["not_run", "preflight_failed"]


@pytest.mark.parametrize(
    "first_status,expected_code", [("task_failure", 0), ("agent_error", 1)]
)
def test_native_failures_keep_scores_separate_and_continue(
    tmp_path, monkeypatch, first_status, expected_code
):
    monkeypatch.setattr(suite, "IS_WINDOWS", False)
    monkeypatch.setattr(suite, "check_inputs", lambda job: None)
    calls = []

    def execute(command, **kwargs):
        if "--dry-run" in command:
            return subprocess.CompletedProcess(command, 0, "{}", "")
        benchmark = command[command.index("--benchmark") + 1]
        calls.append(benchmark)
        report = {
            "run_id": command[command.index("--run-id") + 1],
            "cases": [
                {
                    "result": {
                        "status": first_status if benchmark == "robolab" else "success"
                    }
                }
            ],
            "benchmarks": {
                benchmark: {"aggregate_score": 0.25 if benchmark == "robolab" else 0.8}
            },
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(report), "")

    monkeypatch.setattr(reporting, "run_child", execute)
    assert suite.main(arguments(tmp_path, ["robolab", "lhpr_vln"])) == expected_code
    assert calls == ["robolab", "lhpr_vln"]
    report = json.loads((tmp_path / "work/results/trial/summary.json").read_text())
    assert (
        report["jobs"][1]["report"]["benchmarks"]["lhpr_vln"]["aggregate_score"] == 0.8
    )
    assert "success_rate" not in report and "aggregate_score" not in report
    assert report["complete"] == (expected_code == 0)


def test_failed_job_does_not_prevent_later_job_or_leak_credentials(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(suite, "IS_WINDOWS", False)
    monkeypatch.setattr(suite, "check_inputs", lambda job: None)
    monkeypatch.setenv("OPENAI_API_KEY", "private-fixture-key")
    monkeypatch.setenv("HF_TOKEN", "private-fixture-hf-token")
    calls = []

    def execute(command, **kwargs):
        if "--dry-run" in command:
            return subprocess.CompletedProcess(command, 0, "{}", "")
        calls.append(command)
        raise OSError("private-fixture-key private-fixture-hf-token")

    monkeypatch.setattr(reporting, "run_child", execute)
    args = arguments(tmp_path, ["robolab", "lhpr_vln"])
    assert suite.main(args) == 1
    assert len(calls) == 2
    directory = tmp_path / "work/results/trial"
    saved = (directory / "summary.json").read_text()
    assert "private-fixture" not in saved
    assert "[REDACTED]" in saved
    assert suite.main(args) == 2
    assert (directory / "summary.json").read_text() == saved


def test_zero_exit_without_a_complete_report_is_not_success(tmp_path):
    job = jobs_for(tmp_path, ["libero"])[0]
    job.report.parent.mkdir(parents=True)
    job.report.write_text(
        json.dumps({"complete": True, "total": 1, "runtime_errors": 1})
    )
    with pytest.raises(ValueError, match="runtime errors"):
        suite.read_report(job, subprocess.CompletedProcess(job.command, 0))
    job.report.write_text(json.dumps({"complete": True, "total": 0}))
    with pytest.raises(ValueError, match="no cases"):
        suite.read_report(job, subprocess.CompletedProcess(job.command, 0))


def test_wrong_cli_pin_fails_before_starting_the_agent(tmp_path, monkeypatch):
    job = jobs_for(tmp_path, ["robolab"])[0]
    job.settings["codex"] = str(tmp_path / "codex")
    job.env["OPENAI_API_KEY"] = "fixture-key"
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="codex-cli 0.154.0")
    )
    with pytest.raises(ValueError, match="requires codex-cli 0.160.0"):
        suite.check_inputs(job)


def test_swapped_r2r_rxr_registry_is_rejected_before_evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr(suite, "IS_WINDOWS", False)
    monkeypatch.setattr(suite, "check_inputs", lambda job: None)
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        assert "--dry-run" in command
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {"cases": [{"benchmark": "vlnce", "task_id": "r2r-7", "seed": 0}]}
            ),
            "",
        )

    monkeypatch.setattr(reporting, "run_child", execute)
    assert suite.main(arguments(tmp_path, ["vlnce_rxr"])) == 2
    assert len(calls) == 1
    report = json.loads((tmp_path / "work/results/trial/summary.json").read_text())
    assert "RXR registry" in report["jobs"][0]["error"]


@pytest.mark.parametrize("cohort,prefix", [("vlnce_r2r", "r2r"), ("vlnce_rxr", "rxr")])
def test_navigation_preflight_accepts_its_own_dataset(tmp_path, cohort, prefix):
    job = jobs_for(tmp_path, [cohort])[0]
    completed = SimpleNamespace(
        stdout=json.dumps({"cases": [{"benchmark": "vlnce", "task_id": prefix + "-7"}]})
    )
    suite.validate_preflight(job, completed)


def test_interruption_is_recorded_without_running_remaining_cohorts(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(suite, "IS_WINDOWS", False)
    monkeypatch.setattr(suite, "check_inputs", lambda job: None)

    def execute(command, **kwargs):
        if "--dry-run" in command:
            return subprocess.CompletedProcess(command, 0, "{}", "")
        raise KeyboardInterrupt

    monkeypatch.setattr(reporting, "run_child", execute)
    assert suite.main(arguments(tmp_path, ["robolab", "lhpr_vln"])) == 130
    report = json.loads((tmp_path / "work/results/trial/summary.json").read_text())
    assert [job["status"] for job in report["jobs"]] == ["interrupted", "not_run"]


@pytest.mark.parametrize("run_id", ["../escape", "trial.v1"])
def test_incomplete_suite_config_and_unsafe_run_ids_fail_offline(
    tmp_path, capsys, run_id
):
    config = suite.load_config(suite.DEFAULT_CONFIG)
    del config["benchmarks"]["vlnce_rxr"]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    assert suite.main(["--config", str(path), "--dry-run"]) == 2
    assert "12 released cohorts" in capsys.readouterr().err
    assert suite.main(["--run-id", run_id, "--dry-run"]) == 2
    assert not (tmp_path / "work").exists()
