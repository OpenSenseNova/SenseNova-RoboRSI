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

"""Inheritance and overrides survive relocation and real service boundaries."""

import json

import pytest

from roborsi.agents.config import AgentConfig
from roborsi.agents.package import export_agent
from roborsi.evaluation.agent_artifact import AgentArtifactStore
from roborsi.evaluation.agent_process import PythonAgentLauncher
from tests.unit_tests.roborsi.agents.test_config import action, source


def inherited(root):
    base = source(root)
    config = json.loads(base.read_text())
    config["source_root"] = "."
    config["parameters"]["nested"] = {"keep": 1, "change": 2}
    config["agent"] = {"max_turns": 5}
    base.write_text(json.dumps(config))
    (root / "variants").mkdir()
    leaf = root / "variants/selected.yaml"
    leaf.write_text(
        "schema_version: 1\nextends: ../preset.json\nparameters:\n  nested: {change: 3}\n"
    )
    return base, leaf


def test_layers_paths_cli_precedence_and_input_identity(tmp_path, monkeypatch):
    base, leaf = inherited(tmp_path / "src")
    monkeypatch.chdir(tmp_path)
    config = AgentConfig.from_file(
        leaf, overrides=["parameters.distance=0.75", "agent.max_turns=9"]
    )
    assert config.source_root == base.parent
    assert config.parameters["nested"] == {"keep": 1, "change": 3}
    assert config.resources["message"] == base.parent / "message.txt"
    assert config.document["agent"]["max_turns"] == 9
    assert base in config.input_files and leaf in config.input_files
    launcher = PythonAgentLauncher(
        agent_config=leaf,
        agent_overrides=["parameters.distance=0.75", "/agent/max_turns=9"],
    )
    assert launcher.agent_version == config.fingerprint
    assert action(launcher, tmp_path / "case")["distance"] == 0.75
    saved = json.loads((tmp_path / "case/agent-config.json").read_text())
    assert saved["configuration"]["parameters"]["distance"] == 0.75
    assert saved["overrides"] == list(config.overrides)
    assert config.fingerprint != AgentConfig.from_file(leaf).fingerprint
    base.write_text(base.read_text() + "\n")
    with pytest.raises(ValueError, match="inputs changed"):
        launcher.start(tmp_path)


def test_export_keeps_parents_and_baked_overrides(tmp_path):
    base, leaf = inherited(tmp_path / "src")
    project = export_agent(
        leaf,
        tmp_path / "export",
        name="inherited",
        overrides=["parameters.distance=0.8"],
    )
    manifest = json.loads((project / "source.json").read_text())
    assert "preset.json" in manifest["files"]
    saved = AgentArtifactStore(tmp_path / "artifacts").create(
        project / "src",
        manifest["factory"],
        components={"control": ["roborsi_agent_bundle_inherited/manifest.json"]},
    )
    base.unlink()
    leaf.unlink()
    assert (
        action(saved.launcher(startup_timeout_s=20), tmp_path / "case")["distance"]
        == 0.8
    )


def test_multiple_parents_merge_maps_and_replace_lists(tmp_path):
    (tmp_path / "a.yaml").write_text(
        "schema_version: 1\nagent: {max_turns: 8, max_tokens: 50}\nruntime: {tool_names: [first, second]}\n"
    )
    (tmp_path / "b.yaml").write_text(
        "schema_version: 1\nagent: {max_turns: 3}\nruntime: {tool_names: [third]}\n"
    )
    leaf = tmp_path / "c.yaml"
    leaf.write_text(
        "schema_version: 1\nextends: [a.yaml, b.yaml]\nagent: {max_tokens: 25}\n"
    )
    config = AgentConfig.from_file(leaf, overrides=['runtime.tool_names.0="final"'])
    assert config.document["agent"] == {"max_turns": 3, "max_tokens": 25}
    assert config.runtime().tool_names == ("final",)


@pytest.mark.parametrize(
    "override",
    [
        "unknown=1",
        "agent.max_turns=true",
        "agent.max_turns=0",
        "runtime.unknown=1",
        'runtime.llm.api_key="secret"',
        "agent.max_tokens=NaN",
        "agent.max_turns=unquoted",
        "parameters.distance.x=1",
        "parameters=[]",
    ],
)
def test_invalid_overrides_fail_before_loading_factory(tmp_path, override):
    preset = source(tmp_path / "src")
    with pytest.raises(ValueError):
        AgentConfig.from_file(preset, overrides=[override])


def test_pointer_keys_and_typed_replacements(tmp_path):
    preset = source(tmp_path / "src")
    config = AgentConfig.from_file(
        preset,
        overrides=[
            '/parameters/key.with.dot={"enabled":true}',
            "/parameters/a~1b=null",
        ],
    )
    assert config.parameters["key.with.dot"] == {"enabled": True}
    assert config.parameters["a/b"] is None


def test_cycle_and_missing_parent_are_rejected(tmp_path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("schema_version: 1\nextends: second.yaml\n")
    second.write_text("schema_version: 1\nextends: first.yaml\n")
    with pytest.raises(ValueError, match="cycle"):
        AgentConfig.from_file(first)
    second.unlink()
    with pytest.raises(FileNotFoundError):
        AgentConfig.from_file(first)
