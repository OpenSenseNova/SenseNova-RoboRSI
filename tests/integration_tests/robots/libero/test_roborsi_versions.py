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

"""Check real container versions, comparison conditions, and rejected metadata."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from roborsi.benchmarks.registry import BenchmarkRegistry
from roborsi.evaluation.agent_artifact import AgentArtifactStore
from roborsi.evaluation.agent_process import PythonAgentLauncher
from roborsi.evaluation.docker_runtime import DockerExecutor
from roborsi.evaluation.orchestration import CaseOrchestrator
from roborsi.evaluation.store import ResultStore
from roborsi.utils.config import get_project_env, require_project_env

pytestmark = [
    pytest.mark.skipif(
        get_project_env("ROBORSI_RUN_LIBERO_BENCHMARK") != "1"
        or not get_project_env("ROBORSI_AGENT_WHEELS"),
        reason="requires the current LIBERO image and complete Agent wheels",
    ),
    pytest.mark.timeout(900),
]


def test_observed_versions_comparison_and_rejected_metadata(tmp_path, monkeypatch):
    original = BenchmarkRegistry([require_project_env("ROBORSI_LIBERO_REGISTRY")]).lock(
        "libero"
    )
    root = tmp_path / "registry" / "libero"
    root.mkdir(parents=True)
    manifest = original.as_dict()
    for asset in manifest["assets"]:
        shutil.copytree(
            original.manifest_path.parent / asset["path"], root / asset["path"]
        )
    manifest["tasks"] = [{**manifest["tasks"][0], "max_steps": 3}]
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    registry = BenchmarkRegistry([root.parent])
    task = registry.lock("libero").list_tasks()[0]
    source = tmp_path / "source"
    source.mkdir()
    repo = Path(__file__).resolve().parents[4]
    code = (repo / "examples/roborsi/hold_agent.py").read_text()
    agent_file = source / "hold_agent.py"
    agent_file.write_text(code)
    artifacts = AgentArtifactStore(tmp_path / "agents")
    environment = artifacts.environments.create(
        require_project_env("ROBORSI_AGENT_WHEELS")
    )
    baseline = artifacts.create(
        source,
        "hold_agent:make_agent",
        components={"control": ["hold_agent.py"]},
        environment_version=environment.version_id,
    )
    agent_file.write_text(code.replace("-1.0", "1.0"))
    candidate = artifacts.create(
        source,
        "hold_agent:make_agent",
        components={"control": ["hold_agent.py"]},
        parent_version=baseline.version_id,
    )
    store = ResultStore(tmp_path / "results")
    processes, containers, launched_names = [], [], []
    call_command = DockerExecutor._call_command

    def track_command(self, args, timeout_s, deadline):
        if args[1] == "run":
            launched_names.append(args[args.index("--name") + 1])
        return call_command(self, args, timeout_s, deadline)

    monkeypatch.setattr(DockerExecutor, "_call_command", track_command)
    start_agent, start_adapter = PythonAgentLauncher.start, DockerExecutor.start_adapter

    def track_agent(self, *args, **kwargs):
        process = start_agent(self, *args, **kwargs)
        processes.append(process)
        return process

    def track_adapter(self, *args, **kwargs):
        container = start_adapter(self, *args, **kwargs)
        containers.append(container)
        return container

    monkeypatch.setattr(PythonAgentLauncher, "start", track_agent)
    monkeypatch.setattr(DockerExecutor, "start_adapter", track_adapter)

    def run(name, artifact=baseline, *, timeout_s=120.0, launcher=None):
        return CaseOrchestrator(
            registry, store, launcher or artifact.launcher(), DockerExecutor()
        ).run(
            name,
            "task",
            f"episode-{name}",
            "libero",
            task.task_id,
            task.seed,
            artifact.version_id,
            timeout_s=timeout_s,
            case_timeout_s=300.0,
        )

    for name, artifact, timeout in (
        ("baseline", baseline, 120.0),
        ("candidate", candidate, 120.0),
        ("settings", baseline, 121.0),
    ):
        result = run(name, artifact, timeout_s=timeout)
        assert result.status in {"success", "task_failure"}, result.as_dict()
        assert result.runtime.observed == result.provenance
        assert result.runtime.agent.verified_after_close
        assert result.runtime.benchmark.verified_after_close
        assert result.runtime.benchmark.configuration["ReadonlyRootfs"]
        assert store.load(name, "task") == result
    comparison = store.compare("baseline", "task", "candidate", "task")
    assert comparison.comparable and comparison.score_delta == 0.0
    assert (
        store.compare("baseline", "task", "settings", "task").reason
        == "Runner settings or scoring rules differ"
    )

    unverified = run(
        "unverified",
        launcher=PythonAgentLauncher("examples.roborsi.hold_agent:make_agent"),
    )
    assert unverified.status in {"success", "task_failure"}, unverified.as_dict()
    assert unverified.runtime.observed.agent_version is None
    assert (
        store.compare("baseline", "task", "unverified", "task").reason
        == "Agent source and environment were not verified"
    )

    for field, value in (
        ("benchmark_commit", "f" * 40),
        ("adapter_version", "wrong-adapter"),
        (
            "evaluation",
            {**manifest["evaluation"], "aggregation": "case_mean"},
        ),
    ):
        manifest_path.write_text(json.dumps({**manifest, field: value}))
        result = run(f"wrong_{field}")
        assert result.status == "infrastructure_error" and result.score is None, (
            result.as_dict()
        )
        assert result.error.details["stage"] == "adapter.startup"
        assert f"runtime {field} differs" in result.error.message
        assert store.load(f"wrong_{field}", "task") == result
    manifest_path.write_text(json.dumps(manifest))
    assert len(processes) == len(containers) == 4
    assert len(launched_names) == 7
    assert all(p.daemon.poll() is not None for p in processes)
    for name in launched_names:
        assert not subprocess.check_output(
            ["docker", "ps", "-aq", "--filter", f"name=^{name}$"],
            text=True,
            timeout=15,
        ).strip()
