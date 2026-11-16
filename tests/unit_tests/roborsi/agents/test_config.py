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

"""Preset resources and settings survive process and installation boundaries."""

import json
from pathlib import Path

import pytest

from roborsi.agents.config import AgentConfig
from roborsi.evaluation.agent_process import PythonAgentLauncher
from roborsi.evaluation.messages import Observation, Task


def source(root: Path) -> Path:
    root.mkdir()
    (root / "policy.py").write_text("""from roborsi.evaluation.messages import Action

def make_agent(case_dir, *, config):
    class Policy:
        def reset(self, task):
            self.task = task
        def act(self, observation):
            return Action(benchmark=self.task.benchmark,
                          episode_id=observation.episode_id,
                          step_index=observation.step_index,
                          data={"distance": config.parameters["distance"],
                                "message": config.resources["message"].read_text()})
        def close(self):
            pass
    return Policy()
""")
    (root / "message.txt").write_text("portable resource")
    path = root / "preset.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "factory": "policy:make_agent",
                "code": ["policy.py"],
                "parameters": {"distance": 0.25},
                "resources": {"message": "message.txt"},
            }
        )
    )
    return path


def action(launcher, directory):
    directory.mkdir()
    task = Task(
        benchmark="demo", task_id="place", seed=0, instruction="Place", max_steps=1
    )
    with launcher.start(directory) as process:
        process.client.reset(task, "case", timeout_s=5)
        result = process.client.act(
            Observation(benchmark="demo", episode_id="case", step_index=0), timeout_s=5
        )
        process.client.close()
    return result.data


def test_preset_reaches_child_factory_and_saves_actual_inputs(tmp_path, monkeypatch):
    preset = source(tmp_path / "source")
    monkeypatch.chdir(tmp_path)
    launcher = PythonAgentLauncher(agent_config=preset, startup_timeout_s=20)
    assert action(launcher, tmp_path / "case") == {
        "distance": 0.25,
        "message": "portable resource",
    }
    saved = json.loads((tmp_path / "case/agent-config.json").read_text())
    assert saved["config_sha256"] == launcher.agent_version
    assert saved["configuration"]["parameters"]["distance"] == 0.25
    assert set(saved["files"]) == {"preset.json", "policy.py", "message.txt"}
    assert str(tmp_path) not in json.dumps(saved)


@pytest.mark.parametrize("changed", ["message.txt", "policy.py", "preset.json"])
def test_changed_inputs_rejected_before_launch(tmp_path, changed):
    preset = source(tmp_path / "source")
    launcher = PythonAgentLauncher(agent_config=preset)
    with pytest.raises(ValueError, match="version differs"):
        launcher.verify_version("arbitrary-label")
    path = preset.parent / changed
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="inputs changed"):
        launcher.start(tmp_path)
    assert not (tmp_path / "agent-service.log").exists()


@pytest.mark.parametrize(
    "change",
    [
        {"unknown": 1},
        {"agent": {"max_turns": True}},
        {"agent": {"action_timeout_s": 0}},
        {
            "runtime": {
                "llm": {"provider": "openai", "model": "fake", "api_key": "secret"}
            }
        },
        {"runtime": {"unknown": True}},
        {"skills": ["missing.md"]},
        {"resources": {"bad": "../outside.txt"}},
        {"source_root": "missing"},
        {"dependencies": ["--index-url https://example.org"]},
    ],
)
def test_invalid_config_fails_without_importing_custom_code(tmp_path, change):
    preset = source(tmp_path / "source")
    (preset.parent / "policy.py").write_text(
        "raise AssertionError('must not import')\n"
    )
    (tmp_path / "outside.txt").write_text("outside")
    data = json.loads(preset.read_text())
    data.update(change)
    preset.write_text(json.dumps(data))
    with pytest.raises((ValueError, FileNotFoundError)):
        AgentConfig.from_file(preset)


def test_symlink_resource_rejected(tmp_path):
    preset = source(tmp_path / "source")
    (preset.parent / "message.txt").unlink()
    (tmp_path / "private.txt").write_text("not part of the bundle")
    (preset.parent / "message.txt").symlink_to(tmp_path / "private.txt")
    with pytest.raises(ValueError, match="symlink"):
        AgentConfig.from_file(preset)


def test_launcher_rejects_symlink_preset(tmp_path):
    preset = source(tmp_path / "source")
    link = tmp_path / "selected.yaml"
    link.symlink_to(preset)
    with pytest.raises(ValueError, match="symlink"):
        PythonAgentLauncher(agent_config=link)


def test_changed_inputs_rejected_after_service_cleanup(tmp_path):
    preset = source(tmp_path / "source")
    launcher = PythonAgentLauncher(agent_config=preset, startup_timeout_s=20)
    case = tmp_path / "case"
    case.mkdir()
    with pytest.raises(ValueError, match="inputs changed"):
        with launcher.start(case) as process:
            (preset.parent / "message.txt").write_text("changed during episode")
    assert process.daemon.poll() is not None
    process.close()


def test_configured_embodied_agent_uses_model_prompt_skills_and_budget(
    tmp_path, monkeypatch
):
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    from roborsi.llm import LLMConfig

    (tmp_path / "rules.md").write_text("Configured robot rules.")
    (tmp_path / "skill.md").write_text("Configured grasp recipe.")
    preset = tmp_path / "agent.json"
    preset.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "prompt_file": "rules.md",
                "skills": ["skill.md"],
                "agent": {
                    "max_turns": 3,
                    "max_tokens": 123,
                    "planner_timeout_s": 30,
                    "action_timeout_s": 10,
                },
                "runtime": {
                    "llm": {"provider": "openai", "model": "selected-model"},
                    "trace": {"mode": "full"},
                },
            }
        )
    )
    selected = []

    def model(messages, info):
        assert "Configured robot rules." in info.instructions
        assert "Configured grasp recipe." in str(messages) + info.instructions
        return ModelResponse(parts=[ToolCallPart("roborsi_action", {"vector": [0.5]})])

    def build(config):
        selected.append(config.model)
        return FunctionModel(model)

    monkeypatch.setattr(LLMConfig, "build_model", build)
    media = tmp_path / "media"
    media.mkdir()
    monkeypatch.setenv("ROBORSI_MEDIA_DIR", str(media))
    case = tmp_path / "case"
    case.mkdir()
    config = AgentConfig.from_file(preset)
    agent = config.create_agent(case)
    assert agent.max_tokens == 123 and agent.max_turns == 3
    assert agent.planner_timeout_s == 30 and agent.action_timeout_s == 10
    agent.reset(
        Task(
            benchmark="demo",
            task_id="task",
            seed=0,
            instruction="Move",
            max_steps=1,
            capabilities={
                "action_schema": {
                    "type": "object",
                    "properties": {
                        "vector": {"type": "array", "items": {"type": "number"}}
                    },
                    "required": ["vector"],
                }
            },
        )
    )
    try:
        result = agent.act(
            Observation(benchmark="demo", episode_id="case", step_index=0)
        )
        assert result.data == {"vector": [0.5]}
    finally:
        agent.close()
    assert selected == ["selected-model"]
    assert (case / "agent-config.json").is_file()
