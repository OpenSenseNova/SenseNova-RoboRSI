# Copyright 2026 The RoboRSI Contributors.
# Licensed under the Apache License, Version 2.0.

"""The public package contains runnable shared skills and current presets."""

import json
from pathlib import Path

import pytest

from roborsi.agents.config import AgentConfig
from robots.robodojo.agents.submission import agent_track, preflight
from robots.robodojo.agents.submission.execution_profile import (
    prepare_skillpack_context,
)

ROOT = Path(__file__).resolve().parents[4]
POLICY = ROOT / "robots/robodojo/agents/submission"


def test_default_entry_uses_the_published_track_configuration():
    default = AgentConfig.from_file(ROOT / "robots/robodojo/agent.yaml")
    track = AgentConfig.from_file(ROOT / "robots/robodojo/agents/agent-track.yaml")
    assert default.factory == track.factory
    assert default.parameters == track.parameters
    skill_files = [
        path.relative_to(POLICY / "skillpack").as_posix()
        for path in default.input_files
        if path.is_relative_to(POLICY / "skillpack")
    ]
    assert skill_files
    assert all(
        name.startswith(("common/", "robot_runtime/")) or "/" not in name
        for name in skill_files
    )
    assert "task_catalog.json" in skill_files


@pytest.mark.parametrize("task", ["cover_blocks", "imitate_sorting_sequence", "swap_T"])
def test_tasks_receive_identical_shared_skills_and_tools(tmp_path, task):
    context = prepare_skillpack_context(
        POLICY, task, tmp_path, vla_backend="simate_beta"
    )
    skills = context.workspace / ".agents/skills"
    assert {path.name for path in skills.iterdir()} == {"robo-general", "manipulation"}
    assert (skills / "robo-general/SKILL.md").read_text() == agent_track.GENERAL
    assert (skills / "manipulation/SKILL.md").read_text() == agent_track.MANIPULATION
    assert (
        tmp_path / "execution-profile/agent_core.md"
    ).read_text() == agent_track.CORE
    assert set(context.source_identity["mapping"]) == {"base_task"}
    assert not any(path.is_symlink() for path in context.workspace.rglob("*"))


def test_preflight_verifies_all_shared_resources_and_rejects_changes(tmp_path):
    import shutil

    assert preflight.load_submission(POLICY)["cli"]["version"] == "0.160.1"
    report = preflight.validate_skillpack(POLICY, "cover_blocks")
    catalog = json.loads((POLICY / "skillpack/task_catalog.json").read_text())
    assert report["task_count"] == len(catalog["tasks"])
    assert report["exposed_skills"] == ["robo-general", "manipulation"]
    shutil.copytree(POLICY / "skillpack", tmp_path / "skillpack")
    shutil.copy2(POLICY / "submission.json", tmp_path / "submission.json")
    (tmp_path / "skillpack/common/skills/manipulation/SKILL.md").write_text("changed")
    with pytest.raises(ValueError, match="resource differs"):
        preflight.validate_skillpack(tmp_path, "cover_blocks")
