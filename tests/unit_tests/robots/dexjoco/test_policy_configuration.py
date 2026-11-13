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

"""Prompt/skill configuration is model input, not only provenance metadata."""

import shutil
from pathlib import Path

import pytest
import yaml

from roborsi.agents.config import AgentConfig
from robots.dexjoco.agents.baseline.factory import make_agent

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def config_tree(tmp_path, monkeypatch):
    root = tmp_path / "source"
    shutil.copytree(ROOT / "robots/dexjoco", root / "robots/dexjoco")
    monkeypatch.setenv("ROBORSI_DEXJOCO_AGENT_ASSETS", str(tmp_path / "assets"))
    return root / "robots/dexjoco/agents/baseline/agent.yaml"


def test_default_model_inputs_match_historical_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBORSI_DEXJOCO_AGENT_ASSETS", str(tmp_path / "assets"))
    suffix = (
        "\nThe pinned public robot model is at input/robot_reference/README.md, with simulator MJCF and "
        "Allegro hand URDF. The step helper also accepts a bare numeric action array emitted by the frozen ICL package; "
        "it wraps that array without changing any action row.\n"
    )
    for index, preset in enumerate(
        ("robots/dexjoco/agent.yaml", "configs/agents/dexjoco/compat.yaml")
    ):
        case = tmp_path / str(index)
        case.mkdir()
        config = AgentConfig.from_file(ROOT / preset)
        agent = config.create_agent(case)
        for phase, original in (
            ("initial", "dexjoco_policy_icl.md"),
            ("continuation", "dexjoco_policy_continuation.md"),
        ):
            assert (
                agent.prompts[phase].encode()
                == (agent.ref.ROOT / "prompts" / original).read_bytes()
                + suffix.encode()
            )
        assert config.document["agent"]["planner"] == "codex_cli"
        agent.close()


def test_inline_prompt_skills_and_phase_resources(config_tree, tmp_path):
    data = yaml.safe_load(config_tree.read_text())
    data.pop("prompt_file")
    data["agent"]["system_prompt"] = "INLINE_POLICY"
    data["skills"] = ["first.md", "second.md"]
    data["resources"]["continuation_prompt"] = "continue.md"
    for name, content in (
        ("first.md", "FIRST_SKILL"),
        ("second.md", "SECOND_SKILL"),
        ("continue.md", "NEW_CONTINUATION\n"),
    ):
        (config_tree.parent / name).write_text(content)
    config_tree.write_text(yaml.safe_dump(data))
    config = AgentConfig.from_file(config_tree)
    agent = make_agent(tmp_path / "case", config=config)
    assert agent.prompts["continuation"].startswith("NEW_CONTINUATION\nINLINE_POLICY")
    for prompt in agent.prompts.values():
        assert (
            prompt.index("INLINE_POLICY")
            < prompt.index("FIRST_SKILL")
            < prompt.index("SECOND_SKILL")
        )
    agent.close()
    (config_tree.parent / "first.md").write_text("CHANGED")
    assert AgentConfig.from_file(config_tree).fingerprint != config.fingerprint
    with pytest.raises(ValueError, match="inputs changed"):
        make_agent(tmp_path / "other", config=config)


@pytest.mark.parametrize(
    "change",
    [
        {"agent": {"working_dir": "workspace"}},
        {"parameters": {"ignored": True}},
        {"agent": {"model": "other-model"}},
        {"agent": {"codex_cli": {"inherit_env": True}}},
        {"agent": {"codex_cli": {"config_overrides": {"web_search": "live"}}}},
    ],
)
def test_unsupported_options_fail_instead_of_being_ignored(
    config_tree, tmp_path, change
):
    preset = config_tree.parent / "variant.yaml"
    preset.write_text(
        yaml.safe_dump({"schema_version": 1, "extends": "./agent.yaml", **change})
    )
    with pytest.raises(ValueError):
        config = AgentConfig.from_file(preset)
        make_agent(tmp_path / "case", config=config)
