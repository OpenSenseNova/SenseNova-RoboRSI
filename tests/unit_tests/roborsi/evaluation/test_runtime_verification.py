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

"""Observed versions and execution settings control whether scores compare."""

from __future__ import annotations

import json

import pytest

from roborsi.evaluation.messages import RuntimeVerification
from roborsi.evaluation.services import AdapterClient
from roborsi.evaluation.store import ComparisonPolicy, ResultStore
from tests.unit_tests.roborsi.evaluation.test_services import _service
from tests.unit_tests.roborsi.evaluation.test_store import (
    _provenance,
    _result,
    _runtime,
    _task,
)


def test_runtime_info_is_available_before_and_after_episode_close():
    with _service("adapter", factory="make_versioned_demo_adapter") as rpc:
        client = AdapterClient(rpc)
        before = client.runtime_info(timeout_s=3.0)
        assert before.benchmark_commit == "a" * 40
        rpc.call("close_episode", timeout_s=3.0)
        assert client.runtime_info(timeout_s=3.0) == before
    with _service("adapter") as rpc:
        assert AdapterClient(rpc).runtime_info(timeout_s=3.0) is None


def test_observed_versions_cannot_claim_an_unchecked_environment():
    document = _runtime(_provenance("1" * 64)).as_dict()
    document["agent"]["verified_after_close"] = False
    with pytest.raises(ValueError, match="observed versions differ"):
        RuntimeVerification.from_dict(document)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ("legacy", "runtime versions were not verified"),
        ("agent", "Agent source and environment were not verified"),
        ("timeout", "Runner settings or scoring rules differ"),
        ("assets", "asset contents differ"),
        ("container", "container configuration differs"),
        ("platform", "Benchmark host platform differs"),
    ],
)
def test_comparison_rejects_missing_checks_or_changed_conditions(
    tmp_path, change, reason
):
    store = ResultStore(tmp_path)
    task = _task()
    provenance = _provenance("1" * 64)
    original = _runtime(provenance)
    document = original.as_dict()
    if change == "agent":
        document["agent"] = None
        document["observed"]["agent_version"] = None
    elif change == "timeout":
        document["conditions"]["case_timeout_s"] = 300.0
    elif change == "assets":
        document["benchmark"]["assets"] = [{"name": "models", "sha256": "e" * 64}]
    elif change == "container":
        document["benchmark"]["configuration"]["NanoCpus"] = 2_000_000_000
    elif change == "platform":
        document["benchmark"]["info"]["platform"]["machine"] = "other-cpu"
    changed = None if change == "legacy" else RuntimeVerification.from_dict(document)
    for run, runtime in (("baseline", original), ("candidate", changed)):
        store.claim(run, "case", task, provenance)
        store.publish(run, "case", _result(task, provenance, runtime=runtime))
    if change == "legacy":
        path = tmp_path / "candidate" / "case" / "result.json"
        value = json.loads(path.read_text())
        del value["runtime"]
        path.write_text(json.dumps(value))
        assert store.load("candidate", "case").runtime is None
    comparison = store.compare(
        "baseline",
        "case",
        "candidate",
        "case",
        policy=ComparisonPolicy.strict_runtime(),
    )
    assert not comparison.comparable and comparison.reason == reason


def test_a_scored_result_rejects_a_version_mismatch():
    requested = _provenance("1" * 64)
    other = _provenance("2" * 64)
    with pytest.raises(ValueError, match="runtime agent_version differs"):
        _result(_task(), requested, runtime=_runtime(other))


def test_runner_records_version_mismatch_without_publishing_a_native_score(tmp_path):
    from contextlib import ExitStack

    from roborsi.evaluation.runner import CaseRunner
    from roborsi.evaluation.services import AgentClient

    store = ResultStore(tmp_path)
    requested, actual = _provenance("1" * 64), _provenance("2" * 64)
    with ExitStack() as stack:
        agent = AgentClient(stack.enter_context(_service("agent")))
        adapter = AdapterClient(stack.enter_context(_service("adapter")))
        result = CaseRunner(store).run(
            "mismatch",
            "case",
            "episode",
            _task(),
            requested,
            agent,
            adapter,
            timeout_s=3.0,
            runtime_verification=lambda: _runtime(actual),
        )
    assert result.status == "infrastructure_error" and result.score is None
    assert result.error.details["stage"] == "runtime.verify"
    assert result.runtime.observed.agent_version == actual.agent_version
    assert store.load("mismatch", "case") == result
    assert store.trace_state("mismatch", "case").status == "infrastructure_error"
