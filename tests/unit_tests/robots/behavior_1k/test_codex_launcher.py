# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Exercise the delivered launcher without contacting a model provider."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from roborsi.agents.config import AgentConfig
from roborsi.planner.codex_cli import preflight_codex_cli
from roborsi.planner.codex_cli_config import CodexCliConfig
from robots.behavior_1k.agents.baseline.factory import prepare_workspace

ROOT = Path(__file__).resolve().parents[4]
pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="The delivered Codex CLI launcher requires POSIX exec"
)


@pytest.fixture
def cli(tmp_path, monkeypatch):
    executable = tmp_path / "codex"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "if sys.argv[1:] == ['--version']:\n"
        "    print('codex-cli 0.162.0')\n"
        "else:\n"
        "    print(json.dumps({'args':sys.argv[1:], 'stdin':sys.stdin.read()}))\n"
        "    sys.exit(23)\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-fixture")
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("BEHAVIOR_CODEX_EXECUTABLE", str(executable))
    return executable


@pytest.mark.parametrize(
    "global_args", [[], ["--search"], ["--ask-for-approval", "never"]]
)
def test_delivered_launcher_forwards_streams_status_and_scoped_approvals(
    tmp_path, cli, global_args
):
    config = AgentConfig.load(ROOT / "robots/behavior_1k/agent.yaml")
    prepare_workspace(config, tmp_path)
    arguments = [*global_args, "exec", "--json", "--sandbox", "workspace-write", "-"]
    result = subprocess.run(
        ["./codex_mcp.py", *arguments],
        cwd=tmp_path,
        input="model input",
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 23
    assert result.stderr == ""
    assert json.loads(result.stdout) == {
        "args": [
            *global_args,
            "exec",
            "-c",
            'mcp_servers.roborsi.tools.icl_demonstration.approval_mode="approve"',
            "-c",
            'mcp_servers.roborsi.tools.roborsi_action.approval_mode="approve"',
            "--json",
            "--sandbox",
            "workspace-write",
            "-",
        ],
        "stdin": "model input",
    }


@pytest.mark.parametrize("preset", ["agent.yaml", "agents/baseline/icl.yaml"])
def test_default_preflight_resolves_packaged_launcher(preset, cli):
    config = AgentConfig.load(ROOT / "robots/behavior_1k" / preset)
    options = config.document["agent"]
    cli_config = CodexCliConfig.from_dict(options["codex_cli"])
    workspace = (config.path.parent / options["working_dir"]).resolve()
    identity = preflight_codex_cli(
        cli_config, repo_root=workspace, env=cli_config.subprocess_env()
    )
    assert identity["version"] == "codex-cli 0.162.0"


def test_launcher_rejects_recursive_executable(tmp_path, monkeypatch):
    config = AgentConfig.load(ROOT / "robots/behavior_1k/agent.yaml")
    prepare_workspace(config, tmp_path)
    launcher = tmp_path / "codex_mcp.py"
    monkeypatch.setenv("BEHAVIOR_CODEX_EXECUTABLE", str(launcher))
    result = subprocess.run(
        [str(launcher), "--version"], capture_output=True, text=True, check=False
    )
    assert result.returncode != 0
    assert "underlying CLI" in result.stderr


def test_workspace_rejects_launcher_in_learned_payload(tmp_path):
    config = AgentConfig.load(ROOT / "robots/behavior_1k/agent.yaml")
    (tmp_path / "codex_mcp.py").symlink_to("missing")
    with pytest.raises(ValueError, match="reserved workspace path"):
        prepare_workspace(config, tmp_path)
