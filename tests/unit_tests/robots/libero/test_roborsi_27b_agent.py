# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""27B-only policy and unchanged default Agent routing."""

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from roborsi.agents.config import AgentConfig
from roborsi.evaluation.messages import Task
from robots.libero.roborsi_27b_agent import Roborsi27BIclAgent
from robots.libero.three_benchmark_agent import _boolean

ROOT = Path(__file__).resolve().parents[4]


def test_original_and_27b_configs_choose_separate_factories(monkeypatch):
    from robots.libero.vla_client import openpi_enabled
    from scripts.evaluation import vla

    monkeypatch.delenv("ROBORSI_VLA_MCP_URL", raising=False)
    original = AgentConfig.from_file(ROOT / "configs/agents/libero/pro_vla.yaml")
    custom = AgentConfig.from_file(
        ROOT / "configs/agents/libero/sensenova_roborsi_27b_vla_sam3.yaml"
    )
    assert original.factory.endswith(":make_three_benchmark_agent")
    assert custom.factory == "robots.libero.roborsi_27b_agent:make_agent"
    assert openpi_enabled(original) is True
    assert openpi_enabled(custom) is False
    assert vla.requests({"libero_pro": custom.path}) == {}


@pytest.mark.parametrize(
    "benchmark", ["libero_pro", "libero_long_plus_plus", "libero_ultra_long"]
)
def test_default_service_does_not_load_custom_tools(monkeypatch, tmp_path, benchmark):
    import builtins

    from robots.libero.three_benchmark_adapter import ThreeBenchmarkAdapter
    from robots.libero.three_benchmark_service import ThreeLiberoBenchmark

    monkeypatch.setenv("ROBORSI_LIBERO_BENCHMARK", benchmark)
    for key in (
        "ROBORSI_PRO_EVAL_CONFIG",
        "ROBORSI_IMAGE_OBSERVATION_TOOLS",
        "ROBORSI_LIBERO_PLUS9",
    ):
        monkeypatch.delenv(key, raising=False)
    # Unrelated leftover tool URLs must not activate the default service.
    monkeypatch.setenv("ROBORSI_LIBERO_TOOL_MODE", "vla_sam3")
    monkeypatch.setenv("ROBORSI_VLA_URL", "invalid")
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("robots.libero.image_", "robots.libero.pro_protocol")):
            pytest.fail(f"default service imported optional extension: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    service = ThreeLiberoBenchmark()
    adapter = service.create_adapter(tmp_path, None)
    assert type(adapter) is ThreeBenchmarkAdapter
    assert service.pro_config is None
    assert service.tool_urls == {}
    adapter.close()


@pytest.mark.parametrize("compat", [False, True])
@pytest.mark.parametrize("framework", [None, "no_vla_no_sam3", "vla_sam3"])
def test_reset_records_exact_external_context_identity(
    tmp_path: Path, framework, compat
):
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
        parameters={}
        if framework is None
        else {
            "tool_transport": "openpi_sam3",
            "enable_vla": framework == "vla_sam3",
            "enable_sam3": framework == "vla_sam3",
        },
        embodied_agent=lambda *_args, **_kwargs: planner,
    )
    package = SimpleNamespace(
        package_id="sealed-task-00",
        manifest_sha256="a" * 64,
        memory=(),
        skills=(),
        skill_paths=(),
        initial_context=(),
    )
    agent = Roborsi27BIclAgent(tmp_path, config, {"task-00": package}, ())
    task = Task(
        benchmark="libero_long_plus_plus",
        task_id="task-00",
        seed=0,
        instruction="do it",
        max_steps=1,
    )

    if framework:
        from robots.libero.image_tools import tool_action_schema

        task = task.model_copy(
            update={
                "capabilities": {
                    "action_schema": tool_action_schema(framework == "vla_sam3")
                }
            }
        )
    config.parameters["expose_action_parameter_types"] = compat
    if framework is None and compat:
        from robots.libero.image_tools import tool_action_schema

        task = task.model_copy(
            update={"capabilities": {"action_schema": tool_action_schema()}}
        )
    original = task.as_dict()
    agent.reset(task)

    assert task.as_dict() == original
    if compat:
        model_schema = agent._agent.task.capabilities["action_schema"]
        assert model_schema["properties"]["xyz"]["type"] == "array"
        assert model_schema["oneOf"] == task.capabilities["action_schema"]["oneOf"]
        assert "properties" not in task.capabilities["action_schema"]
    else:
        assert agent._agent.task is task
    assert json.loads((tmp_path / "context-package.json").read_text()) == {
        "schema_version": 1,
        "task_id": "task-00",
        "package_id": "sealed-task-00",
        "manifest_sha256": "a" * 64,
    }


@pytest.mark.parametrize("mode, enabled", [("no_tools", False), ("vla_sam3", True)])
def test_sensenova_parser_compatibility_is_explicit(mode, enabled):
    root = Path(__file__).resolve().parents[4] / "configs/agents/libero"
    config = AgentConfig.from_file(root / f"sensenova_roborsi_27b_{mode}.yaml")
    assert config.parameters["expose_action_parameter_types"] is True
    assert config.parameters["enable_vla"] is enabled
    assert config.parameters["enable_sam3"] is enabled
    for name in ("pro.yaml", "pro_no_vla_no_sam3.yaml", "pro_vla_sam3.yaml"):
        baseline = AgentConfig.from_file(root / name)
        assert not baseline.parameters.get("expose_action_parameter_types", False)


@pytest.mark.parametrize("value", ["true", 1, None])
def test_parser_compatibility_rejects_non_boolean(value):
    with pytest.raises(ValueError, match="expose_action_parameter_types"):
        _boolean(
            {"expose_action_parameter_types": value},
            "expose_action_parameter_types",
            False,
        )


@pytest.mark.parametrize("tools", [False, True])
def test_rotation_explanation_changes_only_private_descriptions(tools):
    from robots.libero.image_tools import action_schema
    from robots.libero.roborsi_27b_agent import _model_action_task

    task = Task(
        benchmark="libero_pro",
        task_id="task-00",
        seed=0,
        instruction="do it",
        max_steps=2000,
        capabilities={"action_schema": action_schema(tools)},
    )
    original = task.as_dict()
    assert _model_action_task(task, {}) is task
    baseline = _model_action_task(task, {"expose_action_parameter_types": True})
    changed = _model_action_task(
        task,
        {
            "expose_action_parameter_types": True,
            "explain_image_rotation_conventions": True,
        },
    )
    assert task.as_dict() == original

    def without_descriptions(value):
        if isinstance(value, dict):
            return {
                k: without_descriptions(v)
                for k, v in value.items()
                if k != "description"
            }
        if isinstance(value, list):
            return [without_descriptions(v) for v in value]
        return value

    assert without_descriptions(changed.as_dict()) == without_descriptions(
        baseline.as_dict()
    )
    annotated = [
        b
        for b in changed.capabilities["action_schema"]["oneOf"]
        if "world-X tilt" in b.get("description", "")
    ]
    assert len(annotated) == 3  # move_to, move_pose, joint rotation branch
    for b in annotated:
        assert "atan2(R[1,2], -R[2,2])" in b["description"]
        assert "positive yaw rotates from world +X toward +Y" in b["description"]


def test_rotation_explanation_is_27b_opt_in():
    root = ROOT / "configs/agents/libero"
    for name in (
        "sensenova_roborsi_27b_no_tools.yaml",
        "sensenova_roborsi_27b_vla_sam3.yaml",
    ):
        assert (
            AgentConfig.from_file(root / name).parameters[
                "explain_image_rotation_conventions"
            ]
            is True
        )
    for name in ("pro.yaml", "pro_no_vla_no_sam3.yaml", "pro_vla_sam3.yaml"):
        assert not AgentConfig.from_file(root / name).parameters.get(
            "explain_image_rotation_conventions", False
        )
