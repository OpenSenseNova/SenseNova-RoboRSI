# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from roborsi.agents.config import AgentConfig
from roborsi.evaluation.messages import Task
from robots.libero.three_benchmark_agent import (
    ThreeBenchmarkIclAgent,
    _boolean,
    _context_root,
    _mcp_server,
)


def test_optional_tool_flags_are_explicit(monkeypatch):
    monkeypatch.setenv("TEST_VLA_URL", "http://127.0.0.1:9001")
    server = _mcp_server(
        {"vla_url_env": "TEST_VLA_URL"}, "vla", _boolean({}, "enable_vla", True)
    )
    assert server is not None
    assert server.url == "http://127.0.0.1:9001"
    assert _mcp_server({}, "sam3", False) is None


@pytest.mark.parametrize(
    ("name", "context_environment", "tool_enabled"),
    [
        ("long_plus_plus.yaml", "ROBORSI_LIBERO_LONGPP_ICL_ROOT", False),
        ("ultra_long.yaml", "ROBORSI_LIBERO_ULTRA_ICL_ROOT", False),
        ("pro.yaml", "ROBORSI_LIBERO_PRO_ICL_ROOT", False),
    ],
)
def test_delivery_presets_compose_from_canonical_agent_directory(
    name, context_environment, tool_enabled
):
    repository = Path(__file__).resolve().parents[4]
    config = AgentConfig.from_file(repository / "configs/agents/libero" / name)

    assert config.factory.endswith(":make_three_benchmark_agent")
    assert config.parameters["context_root_env"] == context_environment
    assert config.parameters["enable_vla"] is tool_enabled
    assert config.parameters["enable_sam3"] is tool_enabled
    assert config.document["agent"]["reasoning_effort"] == "max"
    assert config.resources["context_map"].is_file()
    assert config.document["dependencies"] == [
        "roborsi[runtime]",
        "tokenizers==0.22.2",
    ]


def test_pro_reproduces_the_full_run_policy_limits() -> None:
    repository = Path(__file__).resolve().parents[4]
    config = AgentConfig.from_file(repository / "configs/agents/libero/pro.yaml")

    assert config.document["agent"]["max_turns"] == 100
    assert config.document["agent"]["planner_timeout_s"] == 5000
    assert config.document["runtime"]["context"]["strategy"] == "summary"
    assert config.document["runtime"]["llm"]["retry"]["max_retries"] == 0


def test_context_root_is_supplied_by_named_environment_variable(
    monkeypatch, tmp_path: Path
):
    packages = tmp_path / "packages"
    packages.mkdir()
    monkeypatch.setenv("TEST_ICL_ROOT", str(packages))
    assert _context_root({"context_root_env": "TEST_ICL_ROOT"}) == packages

    monkeypatch.delenv("TEST_ICL_ROOT")
    with pytest.raises(ValueError, match="TEST_ICL_ROOT"):
        _context_root({"context_root_env": "TEST_ICL_ROOT"})


def test_reset_records_exact_external_context_identity(tmp_path: Path):
    @dataclass
    class Planner:
        evaluation_skills: tuple = ()
        evaluation_skill_paths: tuple = ()
        evaluation_memory: tuple = ()
        evaluation_initial_context: tuple = ()
        task: Task | None = None

        def reset(self, task):
            self.task = task

    planner = Planner()
    config = SimpleNamespace(
        parameters={}, embodied_agent=lambda *_args, **_kwargs: planner
    )
    package = SimpleNamespace(
        package_id="sealed-task-00",
        manifest_sha256="a" * 64,
        memory=(),
        skills=(),
        skill_paths=(),
        initial_context=(),
    )
    agent = ThreeBenchmarkIclAgent(tmp_path, config, {"task-00": package}, ())
    task = Task(
        benchmark="libero_long_plus_plus",
        task_id="task-00",
        seed=0,
        instruction="do it",
        max_steps=1,
    )

    agent.reset(task)

    assert agent._agent.task == task
    assert json.loads((tmp_path / "context-package.json").read_text()) == {
        "schema_version": 1,
        "task_id": "task-00",
        "package_id": "sealed-task-00",
        "manifest_sha256": "a" * 64,
    }
