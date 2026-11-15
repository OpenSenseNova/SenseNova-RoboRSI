# Copyright 2026 The RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Claimed cases keep immutable scores and verifiable native evidence."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path

import pytest

from roborsi.evaluation.messages import (
    AgentRuntimeRecord,
    BenchmarkRuntimeInfo,
    BenchmarkRuntimeRecord,
    EvaluationResult,
    Provenance,
    RuntimeConditions,
    RuntimeVerification,
    Task,
)
from roborsi.evaluation.store import ResultStore, TraceState


def _task(seed: int = 7) -> Task:
    return Task(
        benchmark="demo", task_id="place", seed=seed, instruction="Place", max_steps=2
    )


def _provenance(agent_version: str = "agent-v1") -> Provenance:
    return Provenance(
        benchmark_version="bench-v1",
        benchmark_commit="a" * 40,
        docker_image_digest="sha256:" + "b" * 64,
        assets_version="assets-v1",
        adapter_version="adapter-v1",
        agent_version=agent_version,
    )


def _runtime(provenance: Provenance, *, benchmark: bool = True) -> RuntimeVerification:
    agent = AgentRuntimeRecord(
        agent_version=provenance.agent_version,
        environment_version="c" * 64,
        python={"version": "test-python"},
        installed_sha256="d" * 64,
        verified_after_close=True,
    )
    native = (
        BenchmarkRuntimeRecord(
            info=BenchmarkRuntimeInfo(
                benchmark="demo",
                benchmark_version=provenance.benchmark_version,
                benchmark_commit=provenance.benchmark_commit,
                adapter_version=provenance.adapter_version,
                evaluation={"score_metric": "native_score"},
                platform={"machine": "test-cpu"},
            ),
            docker_image_digest=provenance.docker_image_digest,
            assets_version=provenance.assets_version,
            assets=(),
            configuration={"ReadonlyRootfs": True},
            verified_after_close=True,
        )
        if benchmark
        else None
    )
    return RuntimeVerification(
        conditions=RuntimeConditions(
            timeout_s=30.0,
            case_timeout_s=600.0,
            evaluation={"score_metric": "native_score"},
        ),
        agent=agent,
        benchmark=native,
        observed=RuntimeVerification.observed_versions(agent, native),
    )


def _result(
    task: Task,
    provenance: Provenance,
    *,
    score: float = 1.0,
    success: bool = True,
    native_source: str | None = None,
    native_sha256: str | None = None,
    trace_manifest: str | None = None,
    runtime: RuntimeVerification | None = None,
) -> EvaluationResult:
    return EvaluationResult(
        task=task,
        episode_id="episode-1",
        status="success" if success else "task_failure",
        success=success,
        score=score,
        native_score_source=native_source,
        native_score_sha256=native_sha256,
        trace_manifest=trace_manifest,
        provenance=provenance,
        runtime=runtime,
    )


def test_store_preserves_native_result_and_reports_partial_trace(
    tmp_path: Path,
) -> None:
    store = ResultStore(tmp_path)
    task, provenance = _task(), _provenance()
    directory = store.claim("run1", "case1", task, provenance)
    native = directory / "native" / "result.json"
    native.parent.mkdir()
    native.write_text('{"native": true}', encoding="utf-8")
    trace = directory / "trace" / "manifest.json"
    trace.parent.mkdir()
    trace.write_text('{"status": "incomplete", "complete": false}', encoding="utf-8")
    result = _result(
        task,
        provenance,
        native_source="native/result.json",
        native_sha256=sha256(native.read_bytes()).hexdigest(),
        trace_manifest="trace/manifest.json",
    )

    assert store.publish("run1", "case1", result) == TraceState("incomplete", False)
    assert store.load("run1", "case1") == result
    assert store.trace_state("run1", "case1") == TraceState("incomplete", False)
    assert store.list_cases("run1") == {"case1": "published"}
    assert json.loads((directory / "result.json").read_text()) == result.as_dict()
    assert json.loads(native.read_text()) == {"native": True}

    with pytest.raises(FileExistsError):
        store.publish("run1", "case1", result)
    native.write_text('{"native": false}', encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        store.load("run1", "case1")


def test_store_rejects_wrong_inputs_and_bad_artifact_paths(tmp_path: Path) -> None:
    store = ResultStore(tmp_path)
    task, provenance = _task(), _provenance()
    directory = store.claim("run1", "case1", task, provenance)
    assert store.list_cases("run1") == {"case1": "claimed"}
    with pytest.raises(ValueError, match="claimed task"):
        store.publish("run1", "case1", _result(_task(8), provenance))
    with pytest.raises(ValueError, match="claimed provenance"):
        store.publish("run1", "case1", _result(task, _provenance("agent-v2")))
    with pytest.raises(ValueError, match="under native"):
        store.publish(
            "run1",
            "case1",
            _result(
                task,
                provenance,
                native_source="result.json",
                native_sha256="a" * 64,
            ),
        )
    (directory / "native").mkdir()
    (directory / "native" / "escape").symlink_to(directory.parent / "secret")
    (directory.parent / "secret").write_text("secret", encoding="utf-8")
    with pytest.raises(ValueError, match="outside case"):
        store.publish(
            "run1",
            "case1",
            _result(
                task,
                provenance,
                native_source="native/escape",
                native_sha256=sha256(b"secret").hexdigest(),
            ),
        )
    with pytest.raises(ValueError, match="run_id"):
        store.claim("../escape", "case2", task, provenance)


def test_store_comparison_requires_identical_verified_benchmark_inputs(
    tmp_path: Path,
) -> None:
    store = ResultStore(tmp_path)
    task = _task()
    old, new = _provenance("1" * 64), _provenance("2" * 64)
    store.claim("baseline", "case", task, old)
    store.claim("candidate", "case", task, new)
    store.publish(
        "baseline",
        "case",
        _result(task, old, score=0.25, success=False, runtime=_runtime(old)),
    )
    store.publish(
        "candidate", "case", _result(task, new, score=1.0, runtime=_runtime(new))
    )

    comparison = store.compare("baseline", "case", "candidate", "case")
    assert comparison.comparable
    assert comparison.score_delta == 0.75
    assert comparison.success_changed is True

    changed = new.model_copy(update={"adapter_version": "adapter-v2"})
    store.claim("changed", "case", task, changed)
    store.publish("changed", "case", _result(task, changed, runtime=_runtime(changed)))
    assert store.compare("baseline", "case", "changed", "case").reason == (
        "adapter_version differs"
    )
    unknown = Provenance(agent_version="3" * 64)
    store.claim("unknown", "case", task, unknown)
    store.publish(
        "unknown",
        "case",
        _result(task, unknown, runtime=_runtime(unknown, benchmark=False)),
    )
    assert store.compare("baseline", "case", "unknown", "case").reason == (
        "benchmark_version is unknown"
    )


def test_concurrent_claims_reserve_a_case_once(tmp_path: Path) -> None:
    store = ResultStore(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(store.claim, "run", "case", _task(), _provenance())
            for _ in range(2)
        ]
    assert sum(f.exception() is None for f in futures) == 1
    assert sum(isinstance(f.exception(), FileExistsError) for f in futures) == 1
    assert store.list_cases("run") == {"case": "claimed"}


def test_preclaimed_case_must_match_pinned_inputs(tmp_path: Path) -> None:
    store = ResultStore(tmp_path)
    task, provenance = _task(), _provenance()
    directory = store.claim("run", "case", task, provenance)
    assert store.verify_claim("run", "case", task, provenance) == directory
    with pytest.raises(ValueError, match="claimed case inputs"):
        store.verify_claim("run", "case", _task(8), provenance)
    store.publish("run", "case", _result(task, provenance))
    with pytest.raises(FileExistsError, match="already published"):
        store.verify_claim("run", "case", task, provenance)


@pytest.mark.parametrize("digest", [None, "sha256:" + "e" * 64])
def test_deployment_evidence_is_visible_and_user_controls_requirements(digest):
    from roborsi.evaluation.store import ComparisonPolicy, compare_results

    old = _provenance("1" * 64)
    new = old.model_copy(update={"docker_image_digest": digest})
    first = _result(_task(), old, score=0.25, runtime=_runtime(old))
    document = _runtime(new).as_dict()
    document["benchmark"]["configuration"] = {"transport": "sco_http"}
    document["benchmark"]["info"]["platform"] = {"machine": "L40S"}
    second = _result(_task(), new, runtime=RuntimeVerification.from_dict(document))
    default = compare_results(first, second)
    assert default.comparable and default.score_delta == 0.75
    evidence = {c.name: c for c in default.checks}
    assert evidence["docker_image_digest"].status == (
        "unknown" if digest is None else "differs"
    )
    assert not evidence["docker_image_digest"].required
    assert evidence["benchmark_configuration"].status == "differs"
    assert evidence["host_platform"].status == "differs"
    for option, reason in (
        ("require_image_digest", "docker_image_digest"),
        ("compare_benchmark_configuration", "container configuration"),
        ("compare_host_platform", "Benchmark host platform"),
    ):
        policy = ComparisonPolicy.from_dict({option: True})
        comparison = compare_results(first, second, policy=policy)
        assert not comparison.comparable and reason in comparison.reason
        assert comparison.policy == policy
        assert comparison.score_delta is None


@pytest.mark.parametrize("change", ["seed", "steps", "scoring", "commit", "assets"])
def test_default_policy_still_rejects_changed_evaluation_contract(change):
    from roborsi.evaluation.store import compare_results

    provenance = _provenance("1" * 64)
    first = _result(_task(), provenance, runtime=_runtime(provenance))
    task = _task()
    runtime = _runtime(provenance)
    if change in {"seed", "steps"}:
        task = task.model_copy(update={"seed" if change == "seed" else "max_steps": 99})
    elif change == "scoring":
        document = runtime.as_dict()
        document["conditions"]["evaluation"] = {"score_metric": "other"}
        document["benchmark"]["info"]["evaluation"] = {"score_metric": "other"}
        runtime = RuntimeVerification.from_dict(document)
    else:
        provenance = provenance.model_copy(
            update={
                "benchmark_commit" if change == "commit" else "assets_version": "c" * 40
            }
        )
        runtime = _runtime(provenance)
    second = _result(task, provenance, runtime=runtime)
    assert not compare_results(first, second).comparable


def test_comparison_policy_rejects_unknown_or_ambiguous_options():
    from roborsi.evaluation.store import ComparisonPolicy

    for value in ({"ignore_versions": True}, {"require_image_digest": "false"}):
        with pytest.raises(ValueError):
            ComparisonPolicy.from_dict(value)


def test_compare_cli_reports_policy_and_validates_configuration(tmp_path, capsys):
    from roborsi.evaluation.cli import main

    store = ResultStore(tmp_path / "results")
    provenance = _provenance("1" * 64).model_copy(update={"docker_image_digest": None})
    for run in ("before", "after"):
        store.claim(run, "case", _task(), provenance)
        store.publish(
            run, "case", _result(_task(), provenance, runtime=_runtime(provenance))
        )
    args = [
        "compare",
        "--results-root",
        str(store.root),
        "--left-run",
        "before",
        "--left-case",
        "case",
        "--right-run",
        "after",
        "--right-case",
        "case",
    ]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["comparable"]
    policy = tmp_path / "policy.json"
    policy.write_text('{"require_image_digest": true}')
    assert main([*args, "--comparison-policy", str(policy)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert not report["comparable"] and report["policy"]["require_image_digest"]
    policy.write_text('{"require_image_digest": "false"}')
    with pytest.raises(SystemExit) as exc:
        main([*args, "--comparison-policy", str(policy)])
    assert exc.value.code == 2
