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
            argv += [
                "--benchmark",
                benchmark,
                "--benchmark-url",
                f"{benchmark}={endpoint}",
            ]
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
    selected = [value for name in BENCHMARKS for value in ("--benchmark", name)]
    assert (
        runner.main([*arguments(registry, tmp_path), *selected, "--api-key", secret])
        == 0
    )
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
        runner.main(
            [
                *arguments(registry, tmp_path),
                "--benchmark",
                "libero",
                "--dry-run",
                *extra,
            ]
        )
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
    argv = [
        *arguments(registry, tmp_path),
        "--benchmark",
        "libero",
        "--api-key",
        "EMPTY",
    ]
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
        runner.main(
            [
                *arguments(registry, tmp_path),
                "--benchmark",
                "libero",
                "--api-key",
                secret,
                "--dry-run",
            ]
        )
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


def test_automatic_preparation_is_default_and_download_only_needs_no_model(
    runner, tmp_path
):
    _, args = runner.parse_args(
        ["--benchmark", "libero_pro", "--prepare-only", "--work-dir", str(tmp_path)]
    )
    assert args.auto_prepare
    assert args.benchmark == ["libero_pro"]
    assert args.work_dir == tmp_path.resolve()
    assert args.results_root == tmp_path.resolve() / "results"


def test_automatic_dry_run_does_not_download_or_start_containers(
    runner, tmp_path, monkeypatch, capsys
):
    calls = []

    class Bootstrap:
        @staticmethod
        def preparation_plan(benchmarks, work_dir):
            calls.append((benchmarks, work_dir))
            return {"benchmarks": benchmarks, "download": "planned only"}

        @staticmethod
        def prepare_benchmarks(*args):
            pytest.fail("dry-run downloaded assets")

    monkeypatch.setattr(runner, "load_bootstrap", lambda: Bootstrap)
    assert (
        runner.main(
            [
                "--base-url",
                "http://localhost:9/v1",
                "--model-name",
                "vision",
                "--benchmark",
                "robocasa365",
                "--work-dir",
                str(tmp_path),
                "--dry-run",
            ]
        )
        == 0
    )
    assert calls == [(["robocasa365"], tmp_path.resolve())]
    assert "planned only" in capsys.readouterr().out


def test_automatic_environment_is_used_and_cleaned_up(runner, tmp_path, monkeypatch):
    from types import SimpleNamespace

    events = []
    prepared = SimpleNamespace(
        presets={"robocasa365": str(tmp_path / "icl.json")},
        env={"EXAMPLE_ICL_ROOT": str(tmp_path / "icl")},
        images={"robocasa365": "fixture-image"},
    )

    class Bootstrap:
        @staticmethod
        def prepare_benchmarks(benchmarks, work_dir):
            events.append("prepare")
            return prepared

        @staticmethod
        @contextmanager
        def benchmark_services(setup, **kwargs):
            assert setup is prepared
            events.append("start")
            try:
                yield {"robocasa365": "http://localhost:18000"}
            finally:
                events.append("stop")

    def evaluate(parser, args):
        assert args.endpoints == {"robocasa365": "http://localhost:18000"}
        assert args.presets == prepared.presets
        assert args.prepared_env == prepared.env
        events.append("evaluate")
        return 1

    monkeypatch.setattr(runner, "load_bootstrap", lambda: Bootstrap)
    monkeypatch.setattr(runner, "run_evaluation", evaluate)
    assert (
        runner.main(
            [
                "--base-url",
                "http://localhost:9/v1",
                "--model-name",
                "vision",
                "--api-key",
                "EMPTY",
                "--benchmark",
                "robocasa365",
                "--work-dir",
                str(tmp_path),
            ]
        )
        == 1
    )
    assert events == ["prepare", "start", "evaluate", "stop"]


def test_download_only_rejects_existing_external_services(runner):
    with pytest.raises(SystemExit) as exc:
        runner.parse_args(
            [
                "--prepare-only",
                "--benchmark",
                "libero",
                "--benchmark-url",
                "libero=http://localhost:9",
            ]
        )
    assert exc.value.code == 2


def test_benchmark_is_required_even_for_preparation(runner):
    with pytest.raises(SystemExit) as exc:
        runner.parse_args(["--prepare-only"])
    assert exc.value.code == 2


def test_automatic_catalog_options_use_environment(runner, monkeypatch):
    monkeypatch.setenv("ROBORSI_LIBERO_SUITE", "libero_90")
    monkeypatch.setenv("ROBORSI_ROBOCASA_CELLS", "seed1")
    _, args = runner.parse_args(
        ["--prepare-only", "--benchmark", "libero", "--benchmark", "robocasa365"]
    )
    assert args.libero_suite == "libero_90"
    assert args.robocasa_cells == "seed1"


@pytest.mark.parametrize(
    "name,benchmark,value",
    [
        ("ROBORSI_LIBERO_SUITE", "libero", "not-a-suite"),
        ("ROBORSI_ROBOCASA_CELLS", "robocasa365", "invalid"),
    ],
)
def test_invalid_catalog_environment_is_rejected_before_downloading(
    runner, monkeypatch, name, benchmark, value
):
    monkeypatch.setenv(name, value)
    with pytest.raises(SystemExit) as exc:
        runner.parse_args(["--prepare-only", "--benchmark", benchmark])
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "option,value", [("--libero-suite", "libero_90"), ("--robocasa-cells", "seed1")]
)
def test_catalog_options_are_not_part_of_cli(runner, option, value):
    with pytest.raises(SystemExit) as exc:
        runner.parse_args(["--prepare-only", "--benchmark", "libero", option, value])
    assert exc.value.code == 2


def test_distinct_model_yaml_profiles_use_their_own_connection_and_key(
    runner, registry, tmp_path, monkeypatch, capsys
):
    import yaml

    from scripts.evaluation import configuration

    root = tmp_path / "config"
    (root / "benchmarks").mkdir(parents=True)
    (root / "models").mkdir()
    names = ("libero", "libero_pro")
    for index, name in enumerate(names):
        original = configuration.load_benchmark(name)
        document = yaml.safe_load(original.path.read_text())
        document["agent"] = str(original.agent)
        document["model"] = f"../models/{name}.yaml"
        if name == "libero_pro":
            document["environment"]["icl"]["asset_lock"] = str(
                original.environment["icl"]["asset_lock"]
            )
        (root / f"benchmarks/{name}.yaml").write_text(yaml.safe_dump(document))
        (root / f"models/{name}.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "provider": "openai",
                    "base_url": f"http://model-{index}.example/v1",
                    "model_name": f"model-{index}",
                    "base_url_env": f"PROFILE_URL_{index}",
                    "model_name_env": f"PROFILE_NAME_{index}",
                    "api_key_env": f"PROFILE_KEY_{index}",
                    "openai_format": "chat",
                }
            )
        )
        monkeypatch.setenv(f"PROFILE_KEY_{index}", f"private-key-{index}")
        monkeypatch.delenv(f"PROFILE_URL_{index}", raising=False)
        monkeypatch.delenv(f"PROFILE_NAME_{index}", raising=False)
    monkeypatch.setattr(configuration, "CONFIG_ROOT", root)
    calls = []

    def execute(command, **kwargs):
        name = command[command.index("--benchmark") + 1]
        index = names.index(name)
        assert f'runtime.llm.base_url="http://model-{index}.example/v1"' in command
        assert f'runtime.llm.model="model-{index}"' in command
        assert kwargs["env"]["OPENAI_API_KEY"] == f"private-key-{index}"
        assert f"private-key-{1 - index}" not in kwargs["env"].values()
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(evaluation_report(registry, name, "success")),
            "private-key-0 private-key-1",
        )

    monkeypatch.setattr(runner.subprocess, "run", execute)
    assert (
        runner.main(
            [
                "--registry-root",
                str(registry),
                "--benchmark",
                "libero",
                "--benchmark",
                "libero_pro",
            ]
        )
        == 0
    )
    assert len(calls) == 4
    output = capsys.readouterr()
    assert "private-key" not in output.out + output.err
