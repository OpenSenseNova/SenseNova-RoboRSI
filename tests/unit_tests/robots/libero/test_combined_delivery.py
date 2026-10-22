# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");


"""Tests for the three-LIBERO combined delivery and sealed Agent inputs."""

import json
import sys
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.libero_three_combined import run_combined_subset
from benchmarks.libero_three_combined.prepare_agent_assets import (
    prepare as prepare_agent_assets,
)
from benchmarks.libero_three_combined.run_combined import _CASES as _COMBINED_CASES
from benchmarks.libero_three_combined.run_combined_subset import (
    _completed_attempt,
    _conditions,
    _json_sha256,
    _validated_result,
    _write_completion,
)
from benchmarks.libero_three_combined.run_combined_subset import (
    _summary as _combined_summary,
)
from benchmarks.libero_three_combined.run_combined_subset import (
    _tasks as _combined_tasks,
)
from roborsi.agents.config import AgentConfig
from roborsi.agents.package import export_agent
from roborsi.evaluation.messages import EvaluationResult, Provenance, Task
from roborsi.evaluation.store import ResultStore


@pytest.mark.parametrize(
    ("preset", "benchmark", "case_count"),
    [
        ("long_plus_plus", "libero_long_plus_plus", 12),
        ("ultra_long", "libero_ultra_long", 9),
        ("pro", "libero_pro", 80),
    ],
)
def test_combined_agent_bundles_load_their_exported_inputs(
    preset, benchmark, case_count, monkeypatch, tmp_path: Path
) -> None:
    repository = Path(__file__).resolve().parents[4]
    path = repository / "configs/agents/libero" / f"{preset}.yaml"
    original = AgentConfig.from_file(path)
    name = f"delivery_test_{benchmark}"
    project = export_agent(path, tmp_path / "export", name=name)
    monkeypatch.syspath_prepend(str(project / "src"))

    installed = AgentConfig.load(f"bundle:{name}")
    installed.verify()
    assert installed.fingerprint == original.fingerprint
    assert installed.source_root.is_relative_to(project)
    assert all(path.is_relative_to(project) for path in installed.input_files)
    assert (installed.source_root / "robots/libero/three_benchmark_agent.py").is_file()
    context_map = installed.resources["context_map"]
    assert context_map.is_relative_to(project)
    packages = json.loads(context_map.read_text(encoding="utf-8"))["packages"]
    assert len(packages) == case_count
    assert tuple(sorted(packages)) == _combined_tasks(repository, benchmark)


def test_combined_subset_dispatches_the_benchmark_owned_runner(
    monkeypatch, tmp_path: Path
) -> None:
    repository = Path(__file__).resolve().parents[4]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_combined_subset.py",
            "--benchmark",
            "libero_long_plus_plus",
            "--task",
            "t00-i01",
            "--run-id",
            "dispatch-test",
            "--results-root",
            str(tmp_path / "results"),
            "--api-key-file",
            str(tmp_path / "key"),
            "--base-url",
            "https://model.example/v1",
            "--max-concurrent",
            "1",
        ],
    )
    monkeypatch.setattr(run_combined_subset, "_conditions", lambda *_args, **_kw: {})
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(run_combined_subset.subprocess, "run", run)

    assert run_combined_subset.main() == 1
    assert len(calls) == 1
    command, options = calls[0]
    runner = Path(command[2])
    assert runner == repository / "benchmarks/libero_three_combined/run_combined.py"
    assert runner.is_file()
    assert options["cwd"] == repository
    assert command[command.index("--task") + 1] == "t00-i01"
    summary = json.loads((tmp_path / "results/summary.json").read_text())
    assert summary["failed"] == 1
    assert summary["completed"] == 0


def test_combined_runner_uses_the_installed_agent_bundles() -> None:
    assert _COMBINED_CASES == {
        "libero_long_plus_plus": (
            "bundle:libero_long_plus_plus_v13",
            "t00-i01",
        ),
        "libero_ultra_long": ("bundle:libero_ultra_long_v13", "t00-i01"),
        "libero_pro": ("bundle:libero_pro", "libero_spatial_task-t00-i03"),
    }


def test_combined_subset_uses_all_locked_context_map_tasks() -> None:
    repository = Path(__file__).resolve().parents[4]

    assert len(_combined_tasks(repository, "libero_long_plus_plus")) == 12
    assert len(_combined_tasks(repository, "libero_ultra_long")) == 9
    assert len(_combined_tasks(repository, "libero_pro")) == 80


def test_combined_summary_keeps_invalid_attempts_out_of_completed_count() -> None:
    summary = _combined_summary(
        "libero_pro",
        "combined",
        ("success", "task-failure", "agent-error"),
        {
            "success": {
                "returncode": 0,
                "status": "success",
                "success": True,
                "score": 1.0,
            },
            "task-failure": {
                "returncode": 0,
                "status": "task_failure",
                "success": False,
                "score": 0.5,
            },
            "agent-error": {
                "returncode": 1,
                "status": "agent_error",
                "success": False,
            },
        },
    )

    assert summary["planned"] == 3
    assert summary["completed"] == 2
    assert summary["failed"] == 1
    assert summary["successes"] == 1
    assert summary["task_failures"] == 1
    assert summary["mean_score"] == pytest.approx(0.75)


@pytest.mark.parametrize(
    "invalid_record",
    [
        {"returncode": 1, "status": "success", "success": True, "score": 1.0},
        {"returncode": 1, "status": "task_failure", "success": False, "score": 0.9},
        {"returncode": 0, "status": "agent_error", "success": True, "score": 1.0},
    ],
)
def test_combined_summary_excludes_invalid_attempt_scores(invalid_record) -> None:
    summary = _combined_summary(
        "libero_pro",
        "combined",
        ("success", "task-failure", "invalid"),
        {
            "success": {
                "returncode": 0,
                "status": "success",
                "success": True,
                "score": 1.0,
            },
            "task-failure": {
                "returncode": 0,
                "status": "task_failure",
                "success": False,
                "score": 0.25,
            },
            "invalid": invalid_record,
        },
    )

    assert summary["planned"] == 3
    assert summary["completed"] == 2
    assert summary["failed"] == 1
    assert summary["successes"] == 1
    assert summary["task_failures"] == 1
    assert summary["success_rate"] == pytest.approx(1 / 3)
    assert summary["mean_score"] == pytest.approx(0.625)


def test_combined_image_uses_only_roborsi_runtime_names() -> None:
    repository = Path(__file__).resolve().parents[4]
    dockerfile = (
        repository / "benchmarks/libero_three_combined/Dockerfile"
    ).read_text()

    assert "COPY roborsi/ /opt/roborsi/roborsi/" in dockerfile
    assert "ROBORSI_CONTEXT_TOKENIZER_PATH" in dockerfile
    assert "roborsi.evaluation.cli" not in dockerfile
    assert "/opt/heart" not in dockerfile


def test_combined_subset_documentation_uses_python_entrypoint() -> None:
    repository = Path(__file__).resolve().parents[4]

    for relative in (
        "benchmarks/libero_three_combined/README.md",
        "benchmarks/libero_three_combined/README.zh-CN.md",
    ):
        document = (repository / relative).read_text(encoding="utf-8")
        assert (
            "--entrypoint /opt/roborsi/benchmarks/libero_three_combined" not in document
        )
        assert "--entrypoint python" in document
        assert (
            "-B /opt/roborsi/benchmarks/libero_three_combined/run_combined_subset.py"
            in document
        )


def test_combined_conditions_bind_runtime_without_api_secret(
    monkeypatch, tmp_path: Path
) -> None:
    repository = Path(__file__).resolve().parents[4]
    manifest = tmp_path / "asset.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "assets": {"icl": "snapshot"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ROBORSI_IMAGE_REVISION", "a" * 40)
    monkeypatch.setenv("ROBORSI_AGENT_IMAGE", "sha256:" + "b" * 64)
    monkeypatch.setenv("ROBORSI_BENCHMARK_IMAGE", "image@sha256:" + "c" * 64)
    monkeypatch.setenv("ROBORSI_AGENT_ASSET_MANIFEST", str(manifest))
    monkeypatch.setattr(
        "benchmarks.libero_three_combined.run_combined_subset.AgentConfig.load",
        lambda _reference: SimpleNamespace(fingerprint="d" * 64),
    )

    first = _conditions(
        repository,
        "libero_pro",
        base_url="https://model.example/v1/",
        timeout_s=120.0,
        case_timeout_s=3600.0,
    )
    second = _conditions(
        repository,
        "libero_pro",
        base_url="https://other.example/v1",
        timeout_s=120.0,
        case_timeout_s=3600.0,
    )

    assert first["source_revision"] == "a" * 40
    assert first["agent_config_sha256"] == "d" * 64
    assert "model.example" not in json.dumps(first)
    assert _json_sha256(first) != _json_sha256(second)


def test_combined_cache_requires_completion_and_verified_native_evidence(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "cases" / "task-a" / "attempt-001"
    store = ResultStore(attempt / "evaluation")
    task = Task(
        benchmark="libero_pro",
        task_id="task-a",
        seed=0,
        instruction="test",
        max_steps=10,
    )
    provenance = Provenance(agent_version="agent-v1")
    case = store.claim("run", "case", task, provenance)
    native = case / "native" / "score.json"
    native.parent.mkdir()
    native.write_text('{"score": 1.0}', encoding="utf-8")
    result = EvaluationResult(
        task=task,
        episode_id="episode",
        status="success",
        success=True,
        score=1.0,
        native_score_source="native/score.json",
        native_score_sha256=sha256(native.read_bytes()).hexdigest(),
        provenance=provenance,
    )
    store.publish("run", "case", result)
    condition = "e" * 64

    assert (
        _completed_attempt(
            attempt.parent,
            benchmark="libero_pro",
            task_id="task-a",
            condition_sha256=condition,
        )
        is None
    )
    record = _validated_result(
        attempt, benchmark="libero_pro", task_id="task-a", seed=0
    )
    _write_completion(
        attempt,
        benchmark="libero_pro",
        task_id="task-a",
        condition_sha256=condition,
        returncode=0,
        record=record,
    )

    completed = _completed_attempt(
        attempt.parent,
        benchmark="libero_pro",
        task_id="task-a",
        condition_sha256=condition,
    )
    assert completed is not None
    assert completed[1]["native_score_sha256"] == result.native_score_sha256
    assert (
        _completed_attempt(
            attempt.parent,
            benchmark="libero_pro",
            task_id="task-a",
            condition_sha256="f" * 64,
        )
        is None
    )

    completion = attempt / "attempt-completion.json"
    completion_document = json.loads(completion.read_text(encoding="utf-8"))
    completion_document["returncode"] = 1
    completion.write_text(json.dumps(completion_document), encoding="utf-8")
    assert (
        _completed_attempt(
            attempt.parent,
            benchmark="libero_pro",
            task_id="task-a",
            condition_sha256=condition,
        )
        is None
    )
    completion_document["returncode"] = 0
    completion.write_text(json.dumps(completion_document), encoding="utf-8")
    native.write_text('{"score": 0.0}', encoding="utf-8")
    assert (
        _completed_attempt(
            attempt.parent,
            benchmark="libero_pro",
            task_id="task-a",
            condition_sha256=condition,
        )
        is None
    )


def test_agent_asset_lock_records_content_snapshots() -> None:
    repository = Path(__file__).resolve().parents[4]
    lock = json.loads(
        (repository / "benchmarks/libero_three_combined/assets.lock.json").read_text()
    )

    for name in (
        "libero_long_plus_plus_icl",
        "libero_ultra_long_icl",
        "libero_pro_icl",
    ):
        assert len(lock["assets"][name]["snapshot_sha256"]) == 64


def test_agent_asset_staging_requires_all_three_roots(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected ICL roots"):
        prepare_agent_assets(
            {},
            tmp_path / "tokenizer.json",
            tmp_path / "output",
            repository=tmp_path,
        )
