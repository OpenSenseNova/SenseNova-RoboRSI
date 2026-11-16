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

"""Agent delivery contains the code and resources actually used after install."""

import json
import os
import subprocess
import sys
import sysconfig
import venv
import zipfile
from pathlib import Path

import pytest

from roborsi.agents.config import AgentConfig
from roborsi.agents.package import export_agent
from roborsi.evaluation.agent_artifact import AgentArtifactStore
from tests.unit_tests.roborsi.agents.test_config import action, source


def test_bundle_exports_only_declared_inputs_and_runs_from_saved_artifact(tmp_path):
    preset = source(tmp_path / "source")
    (preset.parent / "private.env").write_text("UNRELATED_SECRET=do-not-copy")
    project = export_agent(preset, tmp_path / "export", name="fixture")
    manifest = json.loads((project / "source.json").read_text())
    payload = project / "src/roborsi_agent_bundle_fixture/payload"
    assert not (payload / "private.env").exists()
    assert (
        AgentConfig.from_file(payload / "preset.json").fingerprint
        == AgentConfig.from_file(preset).fingerprint
    )
    saved = AgentArtifactStore(tmp_path / "store").create(
        project / "src",
        manifest["factory"],
        components={
            "control": ["roborsi_agent_bundle_fixture/manifest.json"],
            "tool": ["roborsi_agent_bundle_fixture/payload/policy.py"],
        },
    )
    (preset.parent / "message.txt").write_text("changed working copy")
    assert (
        action(saved.launcher(startup_timeout_s=20), tmp_path / "case")["message"]
        == "portable resource"
    )


def test_export_refuses_nonrelocatable_paths_and_preserves_existing_destination(
    tmp_path,
):
    preset = source(tmp_path / "source")
    data = json.loads(preset.read_text())
    data["resources"]["message"] = str(preset.parent / "message.txt")
    preset.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        export_agent(preset, tmp_path / "export", name="bad")
    assert not (tmp_path / "export").exists()
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep").write_text("keep")
    with pytest.raises(FileExistsError):
        export_agent(preset, existing, name="bad")
    assert (existing / "keep").read_text() == "keep"


def test_export_preserves_executable_resources(tmp_path):
    preset = source(tmp_path / "source")
    tool = preset.parent / "tool.sh"
    tool.write_text("#!/bin/sh\nprintf 'portable tool'\n")
    tool.chmod(0o755)
    data = json.loads(preset.read_text())
    data["resources"]["tool"] = "tool.sh"
    preset.write_text(json.dumps(data))

    project = export_agent(preset, tmp_path / "export", name="executable")
    exported = project / "src/roborsi_agent_bundle_executable/payload/tool.sh"
    assert (
        subprocess.check_output([str(exported)], text=True, cwd=tmp_path)
        == "portable tool"
    )


def _run(command, *, cwd, env=None):
    result = subprocess.run(
        command, cwd=cwd, env=env, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.timeout(180)
def test_wheels_install_and_execute_outside_source_checkout(tmp_path):
    root = Path(__file__).resolve().parents[4]
    preset = source(tmp_path / "source")
    # Match the repository's nested preset layout, including runtime skill data.
    inputs = preset.parent
    (inputs / "skills/move").mkdir(parents=True)
    (inputs / "skills/move/SKILL.md").write_text(
        "---\nname: move\ndescription: Move safely.\n---\nRead the state first.\n"
    )
    (inputs / ".skills/inspect/resources").mkdir(parents=True)
    (inputs / ".skills/inspect/SKILL.md").write_text(
        "---\nname: inspect\ndescription: Inspect the scene.\n---\nRead the hint.\n"
    )
    (inputs / ".skills/inspect/resources/.hint[1].txt").write_text("hidden hint")
    (inputs / ".prompt.md").write_text("Use the installed resources.")
    (inputs / ".private.env").write_text("UNRELATED_SECRET=do-not-copy")
    (inputs / "tool.sh").write_text("#!/bin/sh\nprintf 'installed tool'\n")
    (inputs / "tool.sh").chmod(0o755)
    data = json.loads(preset.read_text())
    data.update(
        source_root="../../..",
        code=["../../../policy.py"],
        resources={"message": "../../../message.txt"},
        runtime={"skill_paths": ["../../../skills/move"]},
    )
    preset.unlink()
    preset = inputs / "configs/agents/fixture/default.json"
    preset.parent.mkdir(parents=True)
    # The parent lives at a different depth, so its resource paths must be rebased.
    data.update(
        source_root=".",
        code=["policy.py"],
        prompt_file=".prompt.md",
        resources={"message": "message.txt", "tool": "tool.sh"},
        runtime={"skill_paths": ["skills/move", ".skills/inspect"]},
    )
    (inputs / "base.json").write_text(json.dumps(data))
    preset.write_text(
        json.dumps({"schema_version": 1, "extends": "../../../base.json"})
    )
    project = export_agent(
        preset,
        tmp_path / "export",
        name="installed",
        overrides=["parameters.distance=0.75"],
    )
    wheels = tmp_path / "wheels"
    for directory in (root, project):
        _run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(wheels),
                str(directory),
            ],
            cwd=tmp_path,
        )
    agent_wheel = next(wheels.glob("roborsi_agent_bundle_installed-*.whl"))
    with zipfile.ZipFile(agent_wheel) as archive:
        assert "roborsi_agent_bundle_installed/payload/base.json" in archive.namelist()
        assert (
            "roborsi_agent_bundle_installed/payload/message.txt" in archive.namelist()
        )
        assert (
            "roborsi_agent_bundle_installed/payload/configs/agents/fixture/default.json"
            in archive.namelist()
        )
        assert (
            "roborsi_agent_bundle_installed/payload/skills/move/SKILL.md"
            in archive.namelist()
        )
        assert "roborsi_agent_bundle_installed/payload/policy.py" in archive.namelist()
        assert "roborsi_agent_bundle_installed/payload/.prompt.md" in archive.namelist()
        assert (
            "roborsi_agent_bundle_installed/payload/.skills/inspect/SKILL.md"
            in archive.namelist()
        )
        assert (
            "roborsi_agent_bundle_installed/payload/.skills/inspect/resources/.hint[1].txt"
            in archive.namelist()
        )
        assert (
            "roborsi_agent_bundle_installed/payload/.private.env"
            not in archive.namelist()
        )
        assert (
            archive.getinfo(
                "roborsi_agent_bundle_installed/payload/tool.sh"
            ).external_attr
            >> 16
        ) & 0o111
    environment = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / "bin/python"
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            *map(str, wheels.glob("*.whl")),
        ],
        cwd=tmp_path,
    )
    # Reuse test dependencies, but import RoboRSI itself from the installed wheel.
    site = next((environment / "lib").glob("python*/site-packages"))
    (site / "test_dependencies.pth").write_text(sysconfig.get_path("purelib") + "\n")
    # Removing the sources makes an accidental checkout-relative resource fail.
    import shutil

    shutil.rmtree(inputs)
    shutil.rmtree(project)
    script = """from pathlib import Path
import json, subprocess, sys
import roborsi
from roborsi.agents.config import AgentConfig
from roborsi.evaluation.agent_process import PythonAgentLauncher
from roborsi.evaluation.messages import Task, Observation
assert Path(roborsi.__file__).is_relative_to(sys.prefix)
config = AgentConfig.load('bundle:installed')
assert (config.path.parent / config.document['prompt_file']).read_text() == 'Use the installed resources.'
assert (config.runtime().skill_paths[1] / 'resources/.hint[1].txt').read_text() == 'hidden hint'
assert subprocess.check_output([str(config.resources['tool'])], text=True) == 'installed tool'
case = Path('installed-case'); case.mkdir()
launcher = PythonAgentLauncher(agent_config='bundle:installed', agent_overrides=['parameters.distance=0.9'], isolated=True, startup_timeout_s=20)
with launcher.start(case) as process:
    process.client.reset(Task(benchmark='demo', task_id='t', seed=0, instruction='Move', max_steps=1), 'e', timeout_s=5)
    result = process.client.act(Observation(benchmark='demo', episode_id='e', step_index=0), timeout_s=5)
    assert result.data == {'distance': 0.9, 'message': 'portable resource'}
    process.client.close()
assert json.loads((case / 'agent-config.json').read_text())['config_sha256'] == launcher.agent_version
"""
    clean = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME"}
    }
    _run([str(python), "-I", "-B", "-c", script], cwd=tmp_path, env=clean)
