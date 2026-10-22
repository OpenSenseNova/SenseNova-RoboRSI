# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The model-evaluation entry point preserves benchmark contracts and credentials."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import threading
from contextlib import ExitStack, contextmanager
from pathlib import Path

import pytest

from roborsi.benchmarks.registry import BenchmarkRegistry
from roborsi.evaluation.batch import BatchCase, BatchReport, CaseRecord, summarize_cases
from roborsi.evaluation.benchmark_http import BenchmarkApplication, BenchmarkHttpServer
from roborsi.evaluation.messages import (
    BenchmarkRuntimeInfo,
    EvaluationResult,
    Provenance,
    ServiceError,
)
from roborsi.utils.rpc import RpcError
from tests.unit_tests.roborsi.benchmarks.test_registry_contracts import _manifest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/evaluation/evaluate_model.py"
BENCHMARKS = (
    "libero",
    "libero_pro",
    "libero_long_plus_plus",
    "libero_ultra_long",
    "robocasa365",
)


@pytest.fixture
def registry(tmp_path):
    root = tmp_path / "registry"
    source, asset = _manifest(tmp_path / "template")
    for benchmark in BENCHMARKS:
        document = json.loads(source.read_text())
        document["benchmark"] = benchmark
        template = document["tasks"][0]
        document["tasks"] = (
            [
                {
                    **template,
                    "task_id": "atomic_task",
                    "seeds": [1, 2],
                    "native": {"timeout_s": 1800},
                },
                {
                    **template,
                    "task_id": "composite_task",
                    "seeds": [1],
                    "native": {"timeout_s": 3600},
                },
            ]
            if benchmark == "robocasa365"
            else [{**template, "seeds": [0]}]
        )
        destination = root / benchmark
        (destination / "assets").mkdir(parents=True)
        (destination / "assets/tasks.bin").write_bytes(asset.read_bytes())
        (destination / "manifest.json").write_text(json.dumps(document))
    return root


@pytest.fixture
def runner(monkeypatch):
    assert SCRIPT.is_file(), "the one-command model-evaluation entry point is missing"
    spec = importlib.util.spec_from_file_location("evaluate_model_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "IS_WINDOWS", False, raising=False)
    return module


def arguments(registry, tmp_path):
    return [
        "--registry-root",
        str(registry),
        "--base-url",
        "http://127.0.0.1:9/v1",
        "--model-name",
        "vision-model",
        "--run-id",
        "trial",
        "--results-root",
        str(tmp_path / "results"),
    ]


def evaluation_report(registry, benchmark, status, message="service failed"):
    lock = BenchmarkRegistry([registry]).lock(benchmark)
    identity = lock.list_tasks()[0]
    result = EvaluationResult(
        task=lock.task_message(identity.task_id, identity.seed),
        episode_id="fixture",
        status=status,
        success=(status == "success")
        if status in {"success", "task_failure"}
        else None,
        score=1.0 if status == "success" else 0.0 if status == "task_failure" else None,
        provenance=Provenance.from_dict(lock.provenance("fixture-agent").as_dict()),
        error=ServiceError(code=status, message=message)
        if status not in {"success", "task_failure"}
        else None,
    )
    records = (
        CaseRecord(
            BatchCase(benchmark, identity.task_id, identity.seed), result, "fixture"
        ),
    )
    return BatchReport(
        "fixture",
        records,
        {benchmark: summarize_cases(records, lock.as_dict()["evaluation"])},
    ).as_dict()


@contextmanager
def catalog_server(registry, benchmark, tmp_path):
    lock = BenchmarkRegistry([registry]).lock(benchmark)

    class Catalog:
        name = benchmark

        def tasks(self):
            return [
                lock.task_message(item.task_id, item.seed) for item in lock.list_tasks()
            ]

        def assets(self):
            return []

        def runtime_info(self):
            return BenchmarkRuntimeInfo(
                benchmark=benchmark,
                benchmark_version="fixture-1",
                benchmark_commit="a" * 40,
                adapter_version="fixture-1",
                evaluation=lock.as_dict()["evaluation"],
                platform={"machine": "test"},
            )

        def create_adapter(self, *args):
            raise AssertionError("dry-run must never start an environment")

    application = BenchmarkApplication(Catalog(), tmp_path / f"server-{benchmark}")
    server = BenchmarkHttpServer(("127.0.0.1", 0), application)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        application.close()
        server.shutdown()
        server.server_close()
        thread.join(3)


def test_dry_run_preflights_every_suite_from_any_directory_without_credentials(
    registry, tmp_path
):
    environment = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    with ExitStack() as stack:
        argv = arguments(registry, tmp_path)[2:]
        for benchmark in BENCHMARKS:
            endpoint = stack.enter_context(
                catalog_server(registry, benchmark, tmp_path)
            )
            argv += ["--benchmark-url", f"{benchmark}={endpoint}"]
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), *argv, "--dry-run"],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
        )
    assert completed.returncode == 0, completed.stderr
    plans = [
        json.loads(line)
        for line in completed.stdout.splitlines()
        if line.startswith('{"schema_version"')
    ]
    assert len(plans) == 6
    cells = [cell for plan in plans for cell in plan["cases"]]
    assert {(cell["benchmark"], cell["seed"]) for cell in cells} == {
        ("libero", 0),
        ("libero_pro", 0),
        ("libero_long_plus_plus", 0),
        ("libero_ultra_long", 0),
        ("robocasa365", 1),
        ("robocasa365", 2),
    }
    assert not (tmp_path / "results").exists()


def test_execution_passes_secret_only_in_environment_and_respects_robocasa_budgets(
    runner,
    registry,
    tmp_path,
    monkeypatch,
    capsys,
):
    calls = []
    secret = "test-credential-never-in-command"

    def execute(command, **kwargs):
        calls.append((command, kwargs))
        report = evaluation_report(registry, "libero", "success")
        return subprocess.CompletedProcess(command, 0, json.dumps(report) + "\n", "")

    monkeypatch.setattr(runner.subprocess, "run", execute)
    assert runner.main([*arguments(registry, tmp_path), "--api-key", secret]) == 0
    assert len(calls) == 12
    assert all("--dry-run" in command for command, _ in calls[:6])
    assert all("--dry-run" not in command for command, _ in calls[6:])
    for command, kwargs in calls:
        assert secret not in " ".join(command)
        assert kwargs["env"]["OPENAI_API_KEY"] == secret
        assert Path(kwargs["cwd"]) == ROOT
        assert 'agent.reasoning_effort="none"' in command
        assert 'runtime.llm.openai_format="chat"' in command
        assert "--seed" not in command
    atomic = next(
        command for command, _ in calls[6:] if "trial-robocasa365-atomic" in command
    )
    composite = next(
        command for command, _ in calls[6:] if "trial-robocasa365-composite" in command
    )
    assert "agent.planner_timeout_s=1800" in atomic
    assert "robocasa365:atomic_task" in atomic
    assert "robocasa365:composite_task" not in atomic
    assert "agent.planner_timeout_s=3600" in composite
    assert secret not in capsys.readouterr().out


@pytest.mark.parametrize(
    "status,expected",
    [
        ("task_failure", 0),
        ("agent_error", 1),
        ("infrastructure_error", 1),
        ("cancelled", 1),
    ],
)
def test_runtime_failures_are_nonzero_but_native_task_failure_is_a_valid_score(
    runner,
    registry,
    tmp_path,
    monkeypatch,
    status,
    expected,
):
    def execute(command, **kwargs):
        report = evaluation_report(registry, "libero", status)
        return subprocess.CompletedProcess(command, 0, json.dumps(report), "")

    monkeypatch.setattr(runner.subprocess, "run", execute)
    assert (
        runner.main(
            [
                *arguments(registry, tmp_path),
                "--benchmark",
                "libero",
                "--api-key",
                "EMPTY",
            ]
        )
        == expected
    )


@pytest.mark.parametrize(
    "extra",
    [
        ["--base-url", "http://user:secret@localhost/v1"],
        ["--base-url", "http://localhost/v1?api_key=secret"],
        ["--base-url", "not-a-url"],
        ["--max-concurrent", "0"],
        ["--timeout-s", "nan"],
        ["--case-timeout-s", "inf"],
        ["--max-turns", "0"],
        ["--max-tokens", "-1"],
        ["--keep-turns", "0"],
        ["--task", "bad"],
        ["--task", "unselected:task"],
        ["--task", "libero:missing"],
        ["--seed", "999"],
        ["--run-id", "../unsafe"],
        ["--benchmark", "libero", "--benchmark", "libero"],
    ],
)
def test_invalid_inputs_fail_before_launching_any_evaluation(
    runner, registry, tmp_path, monkeypatch, extra
):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid input launched evaluation")

    monkeypatch.setattr(runner.subprocess, "run", unexpected)
    with pytest.raises(SystemExit) as exc:
        runner.main([*arguments(registry, tmp_path), "--dry-run", *extra])
    assert exc.value.code == 2


def test_selected_tasks_and_seeds_do_not_add_unrequested_cells(
    runner, registry, tmp_path, monkeypatch
):
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "{}", "")

    monkeypatch.setattr(runner.subprocess, "run", execute)
    assert (
        runner.main(
            [
                *arguments(registry, tmp_path),
                "--benchmark",
                "robocasa365",
                "--task",
                "robocasa365:atomic_task",
                "--seed",
                "2",
                "--dry-run",
            ]
        )
        == 0
    )
    assert len(calls) == 1
    assert "trial-robocasa365-atomic" in calls[0]
    assert calls[0][-2:] == ["--seed", "2"] or calls[0][-3:-1] == ["--seed", "2"]


def test_http_concurrency_is_rejected_without_contacting_service(runner):
    with pytest.raises(SystemExit) as exc:
        runner.main(
            [
                "--base-url",
                "http://localhost:9/v1",
                "--model-name",
                "vision",
                "--benchmark",
                "libero",
                "--benchmark-url",
                "libero=http://localhost:9",
                "--max-concurrent",
                "2",
                "--dry-run",
            ]
        )
    assert exc.value.code == 2


@pytest.mark.parametrize("explicit", [False, True])
def test_custom_presets_keep_their_protocol_unless_explicitly_overridden(
    runner,
    registry,
    tmp_path,
    monkeypatch,
    explicit,
):
    preset = tmp_path / "custom.yaml"
    preset.write_text(
        "schema_version: 1\nagent:\n  planner_timeout_s: 43200\n  max_turns: 10000\nruntime:\n  llm:\n    provider: openai\n    model: custom\n    openai_format: responses\n"
    )
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "{}", "")

    monkeypatch.setattr(runner.subprocess, "run", execute)
    argv = [
        *arguments(registry, tmp_path),
        "--benchmark",
        "libero_long_plus_plus",
        "--agent-config",
        f"libero_long_plus_plus={preset}",
        "--dry-run",
    ]
    if explicit:
        argv += [
            "--max-turns",
            "42",
            "--max-tokens",
            "4096",
            "--keep-turns",
            "2",
            "--openai-format",
            "responses",
        ]
    assert runner.main(argv) == 0
    command = calls[0]
    assert str(preset) in command
    assert not any(value.startswith("agent.planner_timeout_s=") for value in command)
    assert not any(value.startswith("agent.reasoning_effort=") for value in command)
    assert "43200" in command
    for key, value in [
        ("agent.max_turns", 42),
        ("agent.max_tokens", 4096),
        ("runtime.context.keep_turns", 2),
        ("runtime.llm.openai_format", "responses"),
    ]:
        assert (f"{key}={json.dumps(value)}" in command) == explicit


@pytest.mark.parametrize("dry_run", [True, False])
def test_windows_rejects_unsupported_execution_before_loading_a_catalog(
    runner, registry, tmp_path, monkeypatch, capsys, dry_run
):
    monkeypatch.setattr(runner, "IS_WINDOWS", True)
    monkeypatch.setattr(
        runner,
        "build_commands",
        lambda args: pytest.fail("unsupported execution loaded a catalog"),
    )
    argv = [*arguments(registry, tmp_path), "--api-key", "EMPTY"]
    if dry_run:
        argv.append("--dry-run")
    with pytest.raises(SystemExit) as exc:
        runner.main(argv)
    assert exc.value.code == 2
    assert "Linux" in capsys.readouterr().err


def test_catalog_rpc_errors_are_reported_without_a_traceback_or_credential(
    runner, registry, tmp_path, monkeypatch, capsys
):
    secret = "a-private-api-key"

    def fail(args):
        raise RpcError("catalog", f"catalog unavailable {secret}")

    monkeypatch.setattr(runner, "build_commands", fail)
    with pytest.raises(SystemExit) as exc:
        runner.main([*arguments(registry, tmp_path), "--api-key", secret, "--dry-run"])
    assert exc.value.code == 2
    output = capsys.readouterr().err
    assert "catalog unavailable" in output
    assert secret not in output


@pytest.mark.parametrize("benchmark,expected", [("robocasa365", 0), ("libero", 1)])
def test_only_robocasa_scores_planner_timeout_as_a_valid_task_failure(
    runner, registry, tmp_path, monkeypatch, benchmark, expected
):
    def execute(command, **kwargs):
        report = evaluation_report(
            registry,
            benchmark,
            "agent_error",
            "planner timed out after its protocol budget",
        )
        return subprocess.CompletedProcess(command, 0, json.dumps(report), "")

    monkeypatch.setattr(runner.subprocess, "run", execute)
    assert (
        runner.main(
            [
                *arguments(registry, tmp_path),
                "--benchmark",
                benchmark,
                "--api-key",
                "EMPTY",
            ]
        )
        == expected
    )
