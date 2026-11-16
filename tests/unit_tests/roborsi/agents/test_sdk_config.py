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

"""Backend selection survives preset composition without API-only defaults."""

import json

import pytest

from roborsi.agents.config import AgentConfig


@pytest.mark.parametrize("planner", ["api", "codex", "claude_code"])
def test_backend_preset_reconstructs_agent_and_records_selection(tmp_path, planner):
    preset = tmp_path / "agent.json"
    preset.write_text(json.dumps({"schema_version": 1, "agent": {"planner": planner}}))
    config = AgentConfig.from_file(preset, overrides=['agent.model="selected-model"'])
    case = tmp_path / "case"
    case.mkdir()
    agent = config.create_agent(case)
    assert agent.planner == planner
    assert agent.model == "selected-model"
    assert (agent.runtime is not None) == (planner == "api")
    assert agent.max_tokens == (8192 if planner == "api" else None)
    saved = json.loads((case / "agent-config.json").read_text())
    assert saved["configuration"]["agent"] == {
        "planner": planner,
        "model": "selected-model",
    }
    assert config.fingerprint != AgentConfig.from_file(preset).fingerprint


@pytest.mark.parametrize("planner", ["codex", "claude_code"])
@pytest.mark.parametrize(
    "settings",
    [
        {"runtime": {"context": {"strategy": "recent_turns", "keep_turns": 3}}},
        {"runtime": {"llm": {"provider": "openai", "model": "model"}}},
        {"runtime": {"trace": {"mode": "full"}}},
        {"agent": {"max_tokens": 32}},
    ],
)
def test_sdk_preset_rejects_unsupported_settings_before_launch(
    tmp_path, planner, settings
):
    preset = tmp_path / "agent.json"
    settings = {**settings, "agent": {"planner": planner, **settings.get("agent", {})}}
    preset.write_text(json.dumps({"schema_version": 1, **settings}))
    with pytest.raises(ValueError, match="api planner"):
        AgentConfig.from_file(preset)


@pytest.mark.parametrize(
    "agent", [{"planner": "native"}, {"planner": []}, {"model": " "}]
)
def test_invalid_backend_settings_rejected_at_config_boundary(tmp_path, agent):
    preset = tmp_path / "agent.json"
    preset.write_text(json.dumps({"schema_version": 1, "agent": agent}))
    with pytest.raises(ValueError, match="planner|model"):
        AgentConfig.from_file(preset)


def test_api_preset_rejects_two_model_sources(tmp_path):
    preset = tmp_path / "agent.json"
    preset.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "agent": {"model": "openai:one"},
                "runtime": {"llm": {"provider": "openai", "model": "two"}},
            }
        )
    )
    with pytest.raises(ValueError, match="runtime.llm.*agent.model"):
        AgentConfig.from_file(preset)


@pytest.mark.parametrize("planner", ["api", "codex", "claude_code"])
@pytest.mark.parametrize("extension", ["context_engine_factory", "response_processor"])
def test_context_extensions_are_native_only(tmp_path, planner, extension):
    preset = tmp_path / "agent.json"
    preset.write_text(json.dumps({"schema_version": 1, "agent": {"planner": planner}}))
    config = AgentConfig.from_file(preset)

    def callback(*args):
        return object()

    if planner == "api":
        agent = config.embodied_agent(tmp_path, **{extension: callback})
        assert getattr(agent.runtime, extension) is callback
        return
    with pytest.raises(ValueError, match="api planner"):
        config.embodied_agent(tmp_path, **{extension: callback})
