# Copyright 2026 The RoboRSI Contributors.
# Licensed under the Apache License, Version 2.0.

"""Release boundaries do not require Isaac, XPolicyLab or a model service."""

import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.package import prepare_context
from roborsi.benchmarks.runtime_source import benchmark_source_files

ROOT = Path(__file__).resolve().parents[4]


def test_native_bundle_is_registered_without_agent_payload():
    files = benchmark_source_files("robodojo")
    assert "robots/robodojo/worker.py" in files
    assert "benchmarks/robodojo/prepare.py" in files
    assert not any("/agents/" in name for name in files)


def test_exported_benchmark_imports_without_agent_or_simulator(tmp_path):
    context = prepare_context("robodojo", tmp_path / "context", source_root=ROOT)
    runtime = context / "benchmark-runtime"
    code = """
import importlib.abc, importlib, sys
sys.path.insert(0, sys.argv[1])
blocked = ('XPolicyLab', 'torch', 'isaaclab', 'isaacsim', 'openai', 'anthropic', 'mcp', 'pydantic_ai', 'roborsi.embodied_agent', 'roborsi.planner')
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == n or fullname.startswith(n + '.') for n in blocked):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Guard())
importlib.import_module('robots.robodojo.benchmark_service')
module = importlib.import_module('benchmarks.robodojo.prepare')
sys.argv = ['prepare', 'manifest', '--help']
module.main()
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(runtime)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--image" in result.stdout


def test_submission_resources_are_unchanged():
    from benchmarks.robodojo.provenance import verify_submission

    report = verify_submission(ROOT)
    assert report["protected_files"] > 100


def test_presets_load_and_package_all_submitted_resources():
    from roborsi.agents.config import AgentConfig

    config = AgentConfig.from_file(ROOT / "robots/robodojo/agent.yaml")
    assert config.factory == "robots.robodojo.agents.factory:make_agent"
    names = {p.relative_to(ROOT).as_posix() for p in config.input_files}
    assert "robots/robodojo/eef_types.py" in names
    assert "robots/robodojo/agents/submission/cci_model_template.json" in names
    assert "robots/robodojo/agents/submission/skillpack/frozen_common.json" in names
    assert "robots/robodojo/agents/submission/tools/runtime.json" in names
    assert (
        "robots/robodojo/agents/submission/vendor/robodojo_adaptive/source/adaptive_steps.py"
        in names
    )


def test_submission_model_uses_current_framework():
    pytest.importorskip("fcntl")
    module = importlib.import_module("robots.robodojo.agents.submission.model")
    assert module.Model.__module__ == "robots.robodojo.agents.submission.model"
    assert "XPolicyLab" not in sys.modules


def test_agent_export_is_relocatable_and_excludes_installed_runtimes(tmp_path):
    from roborsi.agents.config import AgentConfig
    from roborsi.agents.package import export_agent

    output = export_agent(
        ROOT / "robots/robodojo/agent.yaml", tmp_path / "agent", name="robodojo_check"
    )
    source = json.loads((output / "source.json").read_text())
    assert not any(
        name.startswith("robots/robodojo/agents/submission/runtime/")
        or name.startswith("robots/robodojo/agents/submission/outputs/")
        for name in source["files"]
    )
    package = next((output / "src").iterdir())
    moved = AgentConfig.from_file(package / "payload" / source["config"])
    assert moved.fingerprint == source["config_sha256"]
    from benchmarks.robodojo.install_agent import resolve_agent_source

    selected_root, policy = resolve_agent_source(moved.path)
    assert selected_root == package / "payload"
    assert policy == selected_root / "robots/robodojo/agents/submission"
    assert (policy / "install_simate.sh").is_file()


def test_optional_simate_source_matches_its_original_pin():
    from robots.robodojo.agents.submission.simate_assets import validate_source

    report = validate_source()
    assert report["files"] == 9


@pytest.mark.skipif(
    sys.platform != "linux", reason="submitted policy owns Linux processes"
)
def test_exported_agent_can_construct_its_policy_without_checkout_robot_modules(
    tmp_path,
):
    from roborsi.agents.package import export_agent

    output = export_agent(
        ROOT / "robots/robodojo/agent.yaml", tmp_path / "export", name="robodojo_import"
    )
    package = next((output / "src").iterdir())
    payload = package / "payload"
    framework = tmp_path / "framework"
    shutil.copytree(ROOT / "roborsi", framework / "roborsi")
    code = """
import os, sys
from pathlib import Path
sys.path[:0] = [sys.argv[1], sys.argv[2]]
os.environ.update(CODEX_API_KEY='fixture-not-a-key', ROBORSI_CODEX_MODEL='fixture-model',
                  ROBORSI_CODEX_BASE_URL='https://provider.example.invalid/v1')
from robots.robodojo.agents.submission import tool_runtime
tool_runtime.resolve_tool_python = lambda policy: Path(sys.executable)
from robots.robodojo.agents.submission.model import Model
model = Model({'bench_name':'RoboDojo', 'env_cfg_type':'arx_x5', 'action_type':'joint',
               'task_name':'build_tower', 'vla_backend':'disabled'})
try:
    from robots.robodojo import eef_types
    assert Path(eef_types.__file__).is_relative_to(sys.argv[1])
finally:
    model.shutdown()
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(payload), str(framework)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_catalog_rejects_duplicate_task_layouts(tmp_path):
    from roborsi.evaluation.messages import Task
    from robots.robodojo.benchmark_service import native_commit, read_catalog
    from robots.robodojo.roborsi_adapter import ACTION_SCHEMA

    task = Task(
        benchmark="robodojo",
        task_id="build_tower",
        seed=0,
        instruction="Build the tower",
        max_steps=100,
        native={
            "config": "arx_x5",
            "eval_seed": 0,
            "native_step_limit": 100,
            "layout_sha256": "a" * 64,
        },
        capabilities={
            "action_schema": ACTION_SCHEMA,
            "observation_channels": ["cam_head"],
        },
    )
    document = {
        "schema_version": 1,
        "benchmark_commit": native_commit(),
        "assets_sha256": "b" * 64,
        "tasks": [task.as_dict(), task.as_dict()],
    }
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="unique"):
        read_catalog(catalog)
