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

"""SDK Agents cross real HTTP/process boundaries through eval."""

import base64
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from roborsi.evaluation import cli
from roborsi.evaluation.store import ResultStore
from tests.unit_tests.roborsi.evaluation.test_services import _service


def _source(root, planner):
    root.mkdir()
    (root / "scripted_sdk.py").write_bytes(
        Path(__file__).with_name("scripted_sdk.py").read_bytes()
    )
    (root / "prompt.txt").write_text("version one")
    (root / "skill.md").write_text(
        "Inspect each image before submitting roborsi_action."
    )
    (root / "sdk_agent.py").write_text("""from pathlib import Path
from roborsi.agents.config import AgentConfig
from scripted_sdk import install

def make_agent(case_dir, *, config=None):
    config = config or AgentConfig.from_file(Path(__file__).with_name("agent.json"))
    prompt = (config.path.parent / "prompt.txt").read_text()
    install(config.document["agent"]["planner"], case_dir, prompt)
    return config.embodied_agent(case_dir)
""")
    (root / "agent.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "factory": "sdk_agent:make_agent",
                "code": ["sdk_agent.py", "scripted_sdk.py"],
                "prompt_file": "prompt.txt",
                "skills": ["skill.md"],
                "agent": {
                    "planner": planner,
                    "model": "scripted",
                    "max_turns": 4,
                    "planner_timeout_s": 20,
                    "action_timeout_s": 15,
                },
            }
        )
    )
    return root


def _assert_case(root, run, score, planner):
    store = ResultStore(root)
    cases = store.list_cases(run)
    assert len(cases) == 1
    case = next(iter(cases))
    result = store.load(run, case)
    assert result.error is None, result.error
    assert result.score == score
    assert result.status == ("success" if score else "task_failure")
    audit_path = next((root / run).rglob("sdk-audit.json"))
    audit = json.loads(audit_path.read_text())
    assert audit["closed"]
    assert len(audit["inputs"]) == 1

    def images(value):
        if isinstance(value, str) and value.startswith("data:image/"):
            yield base64.b64decode(value.split(",", 1)[1])
        elif isinstance(value, dict):
            if value.get("type") == "base64" and value.get("media_type") == "image/png":
                yield base64.b64decode(value["data"])
            else:
                for child in value.values():
                    yield from images(child)
        elif isinstance(value, list):
            for child in value:
                yield from images(child)

    initial_images = list(images(audit["inputs"]))
    assert len(initial_images) == 1
    assert Image.open(io.BytesIO(initial_images[0])).getpixel((0, 0)) == (0, 0, 0)
    assert len(audit["feedback"]) == 2
    for index, blocks in enumerate(audit["feedback"], 1):
        camera = next(block for block in blocks if block["type"] == "image")
        assert camera["mimeType"] == "image/png"
        pixels = Image.open(io.BytesIO(base64.b64decode(camera["data"])))
        assert pixels.getpixel((0, 0)) == (index, 0, 0)
        state = json.loads(
            next(block["text"] for block in blocks if block["type"] == "text")
        )
        assert state["step_index"] == index
        assert bool(state.get("_finish")) == (index == 2)
    suffix = "codex" if planner == "codex" else "claude"
    assert (audit_path.parent / f"{suffix}_embodied_agent.txt.stream.jsonl").is_file()
    assert result.trace_manifest is not None


@pytest.mark.parametrize("planner", ["codex", "claude_code"])
def test_eval_cli_runs_sdk_preset_with_media_and_terminal_feedback(tmp_path, planner):
    source = _source(tmp_path / "source", planner)
    with _service("benchmark", factory="make_sdk_benchmark") as service:
        assert (
            cli.main(
                [
                    "eval",
                    "--benchmark-url",
                    f"demo={service.endpoint}",
                    "--results-root",
                    str(tmp_path / "results"),
                    "--run-id",
                    "eval",
                    "--benchmark",
                    "demo",
                    "--agent-config",
                    str(source / "agent.json"),
                    "--timeout-s",
                    "30",
                    "--case-timeout-s",
                    "60",
                ]
            )
            == 0
        )
        _assert_case(tmp_path / "results", "eval", 0, planner)
        assert service.call("healthz", timeout_s=3)["busy"] is False
