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

"""Navigation entries retain shared policy resources across packaging boundaries."""

import shutil
import subprocess
import sys
import sysconfig
import venv
from pathlib import Path
from types import SimpleNamespace

import pytest

from roborsi.agents.config import AgentConfig
from roborsi.agents.package import export_agent
from robots.vln_shared import cli, exploration
from robots.vln_shared.storage import freeze_snapshot, verify_snapshot

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def framework_wheel(tmp_path_factory):
    directory = tmp_path_factory.mktemp("navigation-framework-wheel")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(directory),
            str(ROOT),
        ],
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return next(directory.glob("*.whl"))


@pytest.mark.parametrize(
    "entry",
    [
        "robots/vlnce/agent.yaml",
        "robots/vlnce/agents/baseline/agent.yaml",
        "robots/lhpr_vln/agent.yaml",
        "robots/lhpr_vln/agents/baseline/agent.yaml",
        "agents/vln_shared/agent.yaml",
    ],
)
def test_entries_share_policy_and_export_inherited_resources(tmp_path, entry):
    config = AgentConfig.from_file(ROOT / entry)
    baseline = AgentConfig.from_file(cli.agent_preset())
    assert config.factory == baseline.factory
    assert config.document["agent"] == baseline.document["agent"]
    assert config.resources == baseline.resources
    project = export_agent(config.path, tmp_path / "export", name="navigation")
    payload = project / "src/roborsi_agent_bundle_navigation/payload"
    relocated = AgentConfig.from_file(payload / entry)
    assert relocated.fingerprint == config.fingerprint
    assert all(path.is_relative_to(payload) for path in relocated.input_files)
    assert relocated.resources["exploration_prompt"].read_bytes() == (
        baseline.resources["exploration_prompt"].read_bytes()
    )
    assert not (payload / "robots/vln_shared/native_worker.py").exists()


@pytest.mark.parametrize("dataset", ["r2r", "rxr", "lhpr_vln"])
def test_campaign_composes_benchmark_entry_before_copying(
    tmp_path, monkeypatch, dataset
):
    selected = cli.agent_preset(dataset)
    benchmark = "lhpr_vln" if dataset == "lhpr_vln" else "vlnce"
    assert selected == ROOT / "robots" / benchmark / "agent.yaml"
    # Relocate just the declared entry tree, then change its leaf configuration.
    template = AgentConfig.from_file(selected)
    checkout = tmp_path / "checkout"
    for source in template.input_files:
        target = checkout / source.relative_to(template.source_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    leaf = checkout / selected.relative_to(ROOT)
    leaf.write_text(leaf.read_text() + "agent:\n  model: test-navigation-model\n")
    monkeypatch.setattr(cli, "agent_preset", lambda dataset=None: leaf)
    preset = cli.make_preset(tmp_path / dataset, {}, dataset=dataset)
    shutil.rmtree(checkout)
    config = AgentConfig.from_file(preset)
    assert config.document["agent"]["model"] == "test-navigation-model"
    assert config.resources["exploration_prompt"].is_file()
    assert all(path.is_relative_to(preset.parent) for path in config.input_files)
    config.verify()


def test_exploration_uses_declared_prompt_from_generated_preset(tmp_path, monkeypatch):
    preset = cli.make_preset(tmp_path / "explorer", {}, dataset="r2r")
    config = AgentConfig.from_file(preset)
    prompt = "Explore using this packaged prompt."
    config.resources["exploration_prompt"].write_text(prompt)
    captured = []

    class TrainingStub:
        ready = True
        resets = 1

        def __init__(self, training, directory, workspace):
            self.workspace = workspace

        def specs(self):
            return []

        def close(self):
            captured.append("tools_closed")

    class AgentStub:
        def run(self, task, *, system_prompt, output_dir):
            assert system_prompt == prompt
            (output_dir / "workspace/EXPLORATION.md").write_text("Training memory.")
            return SimpleNamespace(error=None, stats={})

        def close(self):
            captured.append("agent_closed")

    monkeypatch.setattr(exploration, "TrainingTools", TrainingStub)
    monkeypatch.setattr(AgentConfig, "embodied_agent", lambda *a, **k: AgentStub())
    exploration.explore({}, preset, tmp_path / "run", tmp_path / "snapshot")
    assert verify_snapshot(tmp_path / "snapshot")
    assert captured == ["agent_closed", "tools_closed"]


@pytest.mark.timeout(120)
@pytest.mark.parametrize("entry", ["vlnce", "lhpr_vln", "generated"])
def test_wheel_installs_and_creates_agent_outside_checkout(
    tmp_path, entry, framework_wheel
):
    def run(*command):
        result = subprocess.run(
            command, cwd=tmp_path, capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0, result.stdout + result.stderr

    if entry == "generated":
        workspace = tmp_path / "training"
        workspace.mkdir()
        (workspace / "EXPLORATION.md").write_text("Frozen training memory.")
        (workspace / "navigation.py").write_text("# Frozen training helper.\n")
        freeze_snapshot(workspace, tmp_path / "snapshot")
        preset = cli.make_preset(
            tmp_path / "r2r", {}, tmp_path / "snapshot", dataset="r2r"
        )
    else:
        preset = ROOT / "robots" / entry / "agent.yaml"
    project = export_agent(preset, tmp_path / "project", name="navigation_installed")
    run(sys.executable, "-m", "build", "--wheel", "--no-isolation", str(project))
    wheel = next((project / "dist").glob("*.whl"))
    installed = tmp_path / "installed"
    environment = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True).create(environment)
    site = next((environment / "lib").glob("python*/site-packages"))
    (site / "test_dependencies.pth").write_text(sysconfig.get_path("purelib") + "\n")
    python = str(environment / "bin/python")
    run(
        python,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-deps",
        "--no-compile",
        "--target",
        str(installed),
        str(wheel),
        str(framework_wheel),
    )
    shutil.rmtree(project)
    if entry == "generated":
        shutil.rmtree(preset.parent)
        shutil.rmtree(workspace)
        shutil.rmtree(tmp_path / "snapshot")
    # Reuse third-party test dependencies; no model or simulator is started.
    # The factory and every Agent resource must come from the installed wheel.
    run(
        python,
        "-I",
        "-c",
        """import importlib, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import roborsi
assert Path(roborsi.__file__).is_relative_to(Path(sys.argv[1]))
from roborsi.agents.config import AgentConfig
config = AgentConfig.load('bundle:navigation_installed')
config.verify()
assert all(p.is_relative_to(Path(sys.argv[1])) for p in config.input_files)
case = Path('case')
case.mkdir()
agent = config.create_agent(case)
try:
    assert agent.planner == 'codex_cli'
    factory = importlib.import_module(config.factory.split(':')[0])
    assert Path(factory.__file__).is_relative_to(config.source_root)
    assert (case / 'workspace/.codex/config.toml').read_bytes() == config.resources['tool_policy'].read_bytes()
    assert config.resources['exploration_prompt'].is_file()
    if 'exploration' in config.resources:
        assert (case / 'workspace/EXPLORATION.md').read_text() == 'Frozen training memory.'
finally:
    agent.close()
""",
        str(installed),
    )
