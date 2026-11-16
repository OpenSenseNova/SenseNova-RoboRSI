# Copyright 2026 The SenseNova-RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI selects presets for host processes and installed container factories."""

import pytest

from roborsi.agents.config import AgentConfig
from roborsi.evaluation import cli
from roborsi.evaluation.agent_process import PythonAgentLauncher
from roborsi.evaluation.docker_agent import DockerAgentLauncher
from roborsi.evaluation.store import ResultStore
from tests.unit_tests.roborsi.agents.test_config import source
from tests.unit_tests.roborsi.evaluation.test_batch import (
    _PublishedCases,
    _registry,
)

IMAGE = "registry.example/agent@sha256:" + "a" * 64


@pytest.mark.parametrize("container", [False, True])
def test_cli_selects_config_and_records_derived_version(
    tmp_path, monkeypatch, container
):
    preset = source(tmp_path / "source")
    _registry(tmp_path / "registry")
    selected = []

    def runner(registry, store, launcher, executor):
        selected.append(launcher)
        return _PublishedCases(registry, store)

    monkeypatch.setattr(cli, "CaseOrchestrator", runner)
    assert (
        cli.main(
            [
                "eval",
                "--registry-root",
                str(tmp_path / "registry"),
                "--results-root",
                str(tmp_path / "results"),
                "--run-id",
                "configured",
                "--benchmark",
                "demo",
                "--agent-config",
                "bundle:inside_image" if container else str(preset),
                *(["--agent-image", IMAGE] if container else []),
            ]
        )
        == 0
    )
    launcher = selected[0]
    if container:
        # The Runner does not need the Agent wheel installed locally.
        assert isinstance(launcher, DockerAgentLauncher)
        assert launcher.factory == "roborsi_agent_bundle_inside_image:make_agent"
    else:
        assert isinstance(launcher, PythonAgentLauncher)
        assert launcher.agent_version == AgentConfig.from_file(preset).fingerprint
    store = ResultStore(tmp_path / "results")
    for case in store.list_cases("configured"):
        assert (
            store.load("configured", case).provenance.agent_version
            == launcher.agent_version
        )


@pytest.mark.parametrize("container", [False, True])
def test_invalid_preset_fails_before_benchmark_discovery(
    tmp_path, monkeypatch, capsys, container
):
    preset = tmp_path / "bad.yaml"
    preset.write_text("schema_version: 1\nunknown: true\n")

    def discover(*args, **kwargs):
        pytest.fail("invalid Agent config must fail before discovery")

    monkeypatch.setattr(cli, "BenchmarkRegistry", discover)
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "eval",
                "--registry-root",
                str(tmp_path / "registry"),
                "--results-root",
                str(tmp_path / "results"),
                "--run-id",
                "invalid",
                "--benchmark",
                "demo",
                "--agent-config",
                str(preset),
                *(["--agent-image", IMAGE] if container else []),
            ]
        )
    assert exc.value.code == 2
    assert (
        "bundle:lowercase_name" if container else "known fields"
    ) in capsys.readouterr().err
