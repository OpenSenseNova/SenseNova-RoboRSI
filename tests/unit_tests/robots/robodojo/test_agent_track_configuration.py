# Copyright 2026 The RoboRSI Contributors.
# Licensed under the Apache License, Version 2.0.

"""The public Track preset must not inherit the legacy case deployment."""

from pathlib import Path

import pytest

from roborsi.agents.config import AgentConfig
from robots.robodojo.agents.factory import make_agent
from robots.robodojo.agents.submission.runtime_config import (
    resolve_eef_execution_backend,
)

ROOT = Path(__file__).resolve().parents[4]
TASKS = [
    "fasten_screws",
    "imitate_sorting_sequence",
    "make_kong",
    "pack_objects_into_box",
    "pour_liquid_into_cup",
    "press_by_number",
    "swap_T",
    "sweep_blocks",
]


def test_track_preset_has_no_inherited_case_routing(tmp_path, monkeypatch):
    config = AgentConfig.from_file(ROOT / "robots/robodojo/agent-cci.json")
    monkeypatch.setenv("ROBORSI_MEDIA_DIR", str(tmp_path / "media"))
    agent = make_agent(tmp_path / "case", config=config)
    assert agent.model is None
    assert agent.parameters["skill_library_mode"] == "general_only"
    assert agent.parameters["eef_execution_backend_by_task"] == {}
    assert agent.parameters["vla_backend"] == "simate_beta"
    assert agent.parameters["provider_kind"] == "cci"
    assert agent.parameters["sampling"] == {
        "temperature": 1.0,
        "top_k": 20,
        "top_p": 0.95,
    }
    assert {
        resolve_eef_execution_backend(agent.parameters, task) for task in TASKS
    } == {"bounded_curobo"}
    assert "public_icl_root" not in agent.parameters
    names = {p.name for p in config.input_files}
    assert "general_eval.yml" in names


@pytest.mark.parametrize("task", TASKS)
def test_track_materializes_only_common_skills(tmp_path, monkeypatch, task):
    from robots.robodojo.agents.submission.execution_profile import (
        prepare_skillpack_context,
    )

    # Even a stale corpus environment must not select demonstrations for Track.
    monkeypatch.setenv(
        "ROBORSI_FULL_ICL_BASELINE_ROOT", str(tmp_path / "missing-corpus")
    )
    policy = ROOT / "robots/robodojo/agents/submission"
    context = prepare_skillpack_context(
        policy,
        task,
        tmp_path,
        vla_backend="simate_beta",
        skill_library_mode="general_only",
    )
    skills = context.workspace / ".agents/skills"
    assert {p.name for p in skills.iterdir()} == {"robo-general", "manipulation"}
    assert not (context.workspace / "icl_reference").exists()
    assert not list(context.workspace.rglob("*.jpg"))
    assert not list(context.workspace.rglob("*.npz"))
    assert not any(path.is_symlink() for path in context.workspace.rglob("*"))
    # Model stages the workspace and the controller profile in separate owners.
    # Identical general files must pass validation across these two directories.
    from robots.robodojo.agents.submission.agent_track import load_profile

    controller = tmp_path / "controller-owner"
    controller.mkdir()
    load_profile(
        policy, task, controller, vla_backend="simate_beta"
    ).validate_staged_references(context.workspace)
    assert set(context.source_identity["mapping"]) == {"base_task"}
