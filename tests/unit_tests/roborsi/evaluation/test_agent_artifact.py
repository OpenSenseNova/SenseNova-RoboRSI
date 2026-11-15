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

"""Agent versions remain executable after the editable source changes."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from roborsi.benchmarks.registry import BenchmarkRegistry
from roborsi.evaluation.agent_artifact import AgentArtifactStore
from roborsi.evaluation.messages import Observation, Task
from roborsi.evaluation.orchestration import CaseOrchestrator
from roborsi.evaluation.services import AdapterClient
from roborsi.evaluation.store import ResultStore
from tests.unit_tests.roborsi.benchmarks.test_registry_contracts import _manifest
from tests.unit_tests.roborsi.evaluation.test_services import _service

_FACTORY = "versioned_agent:make_agent"
_COMPONENTS = {
    "prompt": ["prompt.txt"],
    "skill": ["skills/task.md"],
    "tool": ["versioned_agent.py"],
    "control": ["control.json"],
}


def _source(root: Path) -> Path:
    root.mkdir()
    (root / "skills").mkdir()
    (root / "skills" / "task.md").write_text("Handle the task.")
    (root / "control.json").write_text('{"mode":"safe"}')
    (root / "prompt.txt").write_text("version one")
    (root / "versioned_agent.py").write_text(
        """import os
from pathlib import Path
from roborsi.evaluation.messages import Action

class VersionedAgent:
    def reset(self, task):
        self.task = task

    def act(self, observation):
        prompt = (Path(os.environ['ROBORSI_AGENT_ARTIFACT_ROOT']) / 'prompt.txt').read_text()
        return Action(benchmark=self.task.benchmark,
                      episode_id=observation.episode_id,
                      step_index=observation.step_index,
                      data={'prompt': prompt})

    def close(self):
        pass

def make_agent(case_dir):
    return VersionedAgent()
"""
    )
    return root


def _action(artifact, case_dir: Path) -> str:
    case_dir.mkdir()
    task = Task(
        benchmark="demo", task_id="place", seed=0, instruction="Place", max_steps=1
    )
    with artifact.launcher(startup_timeout_s=10).start(case_dir) as process:
        process.client.reset(task, "case-1", timeout_s=3)
        action = process.client.act(
            Observation(benchmark="demo", episode_id="case-1", step_index=0),
            timeout_s=3,
        )
        process.client.close()
    return action.data["prompt"]


def test_parent_and_candidate_launch_independently_from_pinned_sources(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "working-agent")
    store = AgentArtifactStore(tmp_path / "artifacts")
    baseline = store.create(source, _FACTORY, components=_COMPONENTS)
    (source / "prompt.txt").write_text("version two")
    candidate = store.create(
        source, _FACTORY, components=_COMPONENTS, parent_version=baseline.version_id
    )

    assert candidate.version_id != baseline.version_id
    assert candidate.parent_version == baseline.version_id
    assert candidate.changes == ({"path": "prompt.txt", "status": "modified"},)
    assert set(candidate.document["components"]) == set(_COMPONENTS)
    assert _action(store.load(baseline.version_id), tmp_path / "baseline-case") == (
        "version one"
    )
    assert _action(store.load(candidate.version_id), tmp_path / "candidate-case") == (
        "version two"
    )
    tool = source / "versioned_agent.py"
    tool.chmod(0o755)
    mode_candidate = store.create(
        source, _FACTORY, components=_COMPONENTS, parent_version=candidate.version_id
    )
    assert mode_candidate.version_id != candidate.version_id
    assert mode_candidate.changes == (
        {"path": "versioned_agent.py", "status": "modified"},
    )


def test_artifact_rejects_unknown_or_changed_source_before_launch(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "working-agent")
    store = AgentArtifactStore(tmp_path / "artifacts")
    artifact = store.create(source, _FACTORY, components=_COMPONENTS)
    launcher = artifact.launcher()
    with pytest.raises(ValueError, match="differs from pinned artifact"):
        launcher.verify_version("unrelated-version")
    manifest_root = tmp_path / "registry"
    _manifest(manifest_root)
    result_store = ResultStore(tmp_path / "results")

    class NoAdapterStart:
        def start_adapter(self, *args, **kwargs):
            raise AssertionError("mismatched Agent version must fail before startup")

    orchestrator = CaseOrchestrator(
        BenchmarkRegistry([manifest_root]), result_store, launcher, NoAdapterStart()
    )
    with pytest.raises(ValueError, match="differs from pinned artifact"):
        orchestrator.run(
            "run1", "case1", "episode1", "demo", "place_block", 0, "wrong-version"
        )
    assert result_store.list_cases("run1") == {}
    with pytest.raises(ValueError, match="component path"):
        store.create(source, _FACTORY, components={"prompt": ["../outside"]})

    prompt = artifact.source_dir / "prompt.txt"
    prompt.chmod(0o600)
    prompt.write_text("tampered")
    with pytest.raises(ValueError, match="changed"):
        launcher.start(tmp_path / "unused-case")


def test_artifact_metadata_is_content_addressed(tmp_path: Path) -> None:
    source = _source(tmp_path / "working-agent")
    store = AgentArtifactStore(tmp_path / "artifacts")
    artifact = store.create(source, _FACTORY, components=_COMPONENTS)
    metadata = store.root / "versions" / f"{artifact.version_id}.json"
    document = json.loads(metadata.read_text())
    document["factory"] = "other:make_agent"
    metadata.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="metadata changed"):
        store.load(artifact.version_id)


def test_two_artifact_versions_produce_separate_runner_results(tmp_path: Path) -> None:
    source = _source(tmp_path / "working-agent")
    artifacts = AgentArtifactStore(tmp_path / "artifacts")
    baseline = artifacts.create(source, _FACTORY, components=_COMPONENTS)
    (source / "prompt.txt").write_text("version two")
    candidate = artifacts.create(
        source, _FACTORY, components=_COMPONENTS, parent_version=baseline.version_id
    )
    manifest_root = tmp_path / "registry"
    _manifest(manifest_root)
    registry = BenchmarkRegistry([manifest_root])
    store = ResultStore(tmp_path / "results")

    class PromptAdapterExecutor:
        @contextmanager
        def start_adapter(self, lock, case_dir, *, deadline=None):
            assert lock.benchmark == "demo" and case_dir.is_dir()
            with _service("adapter", factory="make_prompt_score_adapter") as rpc:
                yield SimpleNamespace(client=AdapterClient(rpc))

    results = []
    for run_id, artifact in (("baseline", baseline), ("candidate", candidate)):
        orchestrator = CaseOrchestrator(
            registry, store, artifact.launcher(), PromptAdapterExecutor()
        )
        results.append(
            orchestrator.run(
                run_id,
                "case1",
                "episode1",
                "demo",
                "place_block",
                0,
                artifact.version_id,
                timeout_s=3.0,
                case_timeout_s=20.0,
            )
        )
    assert [result.score for result in results] == [0.0, 1.0]
    assert [result.provenance.agent_version for result in results] == [
        baseline.version_id,
        candidate.version_id,
    ]
    assert store.load("baseline", "case1") == results[0]
    assert store.load("candidate", "case1") == results[1]
