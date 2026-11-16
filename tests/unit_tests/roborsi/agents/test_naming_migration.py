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

"""RoboRSI presets do not fall back to removed HEART packages or factories."""

import json
import sys

import pytest

from roborsi.agents.config import AgentConfig
from roborsi.agents.package import export_agent
from roborsi.evaluation.docker_agent import DockerAgentLauncher
from tests.unit_tests.roborsi.agents.test_config import source


def test_legacy_bundle_factory_is_rejected():
    factory = "heart_agent_bundle_example:make_agent"
    with pytest.raises(ValueError, match="factory"):
        DockerAgentLauncher(
            "example@sha256:" + "a" * 64,
            factory,
            agent_config="bundle:example",
            container_user="1000:1000",
        )


def test_bundle_loader_does_not_fall_back_to_a_legacy_package(tmp_path, monkeypatch):
    preset = source(tmp_path / "source")
    project = export_agent(preset, tmp_path / "export", name="legacy_naming")
    new = project / "src/roborsi_agent_bundle_legacy_naming"
    old = project / "src/heart_agent_bundle_legacy_naming"
    new.rename(old)
    manifest = old / "manifest.json"
    data = json.loads(manifest.read_text())
    data["factory"] = "heart_agent_bundle_legacy_naming:make_agent"
    manifest.write_text(json.dumps(data))
    before = manifest.read_bytes()
    monkeypatch.syspath_prepend(str(project / "src"))
    try:
        with pytest.raises(ModuleNotFoundError) as error:
            AgentConfig.load("bundle:legacy_naming")
        assert error.value.name == "roborsi_agent_bundle_legacy_naming"
        assert manifest.read_bytes() == before
    finally:
        sys.modules.pop("heart_agent_bundle_legacy_naming", None)


def test_legacy_builtin_factory_is_not_a_builtin_export(tmp_path):
    preset = tmp_path / "preset.json"
    preset.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "factory": "heart_runtime.agents.config:make_embodied_agent",
            }
        )
    )
    original = preset.read_bytes()
    with pytest.raises(ValueError, match="factory.*code"):
        export_agent(
            preset, tmp_path.parent / (tmp_path.name + "-export"), name="legacy_builtin"
        )
    assert preset.read_bytes() == original


def test_canonical_builtin_factory_rejects_custom_parameters(tmp_path):
    preset = tmp_path / "preset.json"
    preset.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "factory": "roborsi.agents.config:make_embodied_agent",
                "parameters": {"unexpected": True},
            }
        )
    )
    with pytest.raises(ValueError, match="parameters require a custom"):
        AgentConfig.from_file(preset)
