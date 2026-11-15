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

"""Rebuild saved Agent dependencies offline and use them with real LIBERO."""

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
        reason="requires real LIBERO and a complete Agent wheel directory",
    ),
    pytest.mark.timeout(900),
]


def test_rebuilt_versions_and_changed_dependencies(tmp_path: Path, monkeypatch):
    processes, containers = [], []
    start_agent = PythonAgentLauncher.start
    start_adapter = DockerExecutor.start_adapter

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
    original = BenchmarkRegistry(
        [Path(require_project_env("ROBORSI_LIBERO_REGISTRY"))]
    ).lock("libero")
    manifest = original.as_dict()
    root = tmp_path / "registry" / "libero"
    root.mkdir(parents=True)
    for asset in manifest["assets"]:
        shutil.copytree(
            original.manifest_path.parent / asset["path"], root / asset["path"]
        )
    # Three policy steps exercise environment rebuilding and cleanup. Full-length
    # scoring parity has its own native-reference integration test.
    manifest["tasks"] = [{**manifest["tasks"][0], "max_steps": 3}]
    (root / "manifest.json").write_text(json.dumps(manifest))
    registry = BenchmarkRegistry([root.parent])
    task = registry.lock("libero").list_tasks()[0]
    source = tmp_path / "working-agent"
    source.mkdir()
    repo = Path(__file__).resolve().parents[4]
    code = (repo / "examples" / "roborsi" / "hold_agent.py").read_text()
    code = code.replace(
        "import os\n",
        "import os\nassert 'AGENT_UNRELATED_TOKEN' not in os.environ\nassert os.environ['AGENT_ALLOWED_TOKEN'] == 'test-only'\n",
    )
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
    assert candidate.version_id != baseline.version_id
    assert candidate.document["environment_version"] == environment.version_id
    monkeypatch.setenv("AGENT_UNRELATED_TOKEN", "never-pass-this")
    monkeypatch.setenv("AGENT_ALLOWED_TOKEN", "test-only")
    monkeypatch.setenv("PYTHONPATH", "/must-not-be-used")
    store = ResultStore(tmp_path / "results")
    snapshots = []
    for run_id, artifact, gripper in (
        ("baseline", baseline, -1.0),
        ("candidate", candidate, 1.0),
    ):
        result = CaseOrchestrator(
            registry,
            store,
            artifact.launcher(credential_env=["AGENT_ALLOWED_TOKEN"]),
            DockerExecutor(),
        ).run(
            run_id,
            "task",
            f"episode-{run_id}",
            "libero",
            task.task_id,
            task.seed,
            artifact.version_id,
            timeout_s=120,
            case_timeout_s=300,
        )
        assert result.status in {"success", "task_failure"}, result.as_dict()
        assert store.load(run_id, "task") == result
        directory = tmp_path / "results" / run_id / "task" / "agent"
        runtime = json.loads((directory / "runtime.json").read_text())
        assert runtime["verified_after_close"]
        assert runtime["environment_version"] == environment.version_id
        assert runtime["agent_version"] == artifact.version_id
        assert "test-only" not in json.dumps(runtime)
        probe = subprocess.check_output(
            [
                str(directory / "python" / "bin" / "python"),
                "-I",
                "-B",
                "-c",
                "import sys, roborsi, json, importlib.util; print(json.dumps([sys.prefix, sys.base_prefix, roborsi.__file__, importlib.util.find_spec('mujoco') is None]))",
            ],
            text=True,
        )
        prefix, base_prefix, harness, no_simulator = json.loads(probe)
        assert prefix != base_prefix and Path(prefix) == directory / "python"
        assert Path(harness).is_relative_to(directory / "python") and no_simulator
        snapshots.append(prefix)
        actions = [
            e["payload"]["message"]["payload"]
            for e in store.trace_events(run_id, "task")
            if e["type"] == "roborsi_message"
            and e["payload"]["message"]["kind"] == "action"
        ]
        assert actions and all(a["data"]["vector"][-1] == gripper for a in actions)
    assert snapshots[0] != snapshots[1]

    # The Agent can still write under its OS user. Detect a dependency edit and
    # refuse its score; filesystem access controls are a separate requirement.
    mutation = code.replace(
        "import os\n",
        "import os\nimport jsonschema\nfrom pathlib import Path\nPath(jsonschema.__file__).write_text(Path(jsonschema.__file__).read_text() + '\\n# modified during evaluation\\n')\n",
    )
    agent_file.write_text(mutation)
    changed = artifacts.create(
        source,
        "hold_agent:make_agent",
        components={"control": ["hold_agent.py"]},
        parent_version=baseline.version_id,
    )
    result = CaseOrchestrator(
        registry,
        store,
        changed.launcher(credential_env=["AGENT_ALLOWED_TOKEN"]),
        DockerExecutor(),
    ).run(
        "modified",
        "task",
        "episode-modified",
        "libero",
        task.task_id,
        task.seed,
        changed.version_id,
        timeout_s=120,
        case_timeout_s=300,
    )
    assert result.status == "infrastructure_error" and result.score is None
    assert result.error.details["stage"] == "service.cleanup"
    assert "environment changed" in str(result.error.details)
    assert store.load("modified", "task") == result
    runtime = json.loads(
        (
            tmp_path / "results" / "modified" / "task" / "agent" / "runtime.json"
        ).read_text()
    )
    assert not runtime["verified_after_close"]
    assert len(processes) == len(containers) == 3
    assert all(process.daemon.poll() is not None for process in processes)
    for container in containers:
        assert not subprocess.check_output(
            ["docker", "ps", "-aq", "--filter", f"id={container.container_id}"],
            text=True,
            timeout=15,
        ).strip()
