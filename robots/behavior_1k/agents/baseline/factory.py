# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Materialize a sealed task-scoped ICL workspace for an EmbodiedAgent."""

from __future__ import annotations

import shutil
import stat
from dataclasses import dataclass, fields
from pathlib import Path

from roborsi.agents.config import AgentConfig
from roborsi.components.icl.package import ContextPackage
from roborsi.components.icl.workspace import materialize_package
from roborsi.embodied_agent import EmbodiedAgent
from roborsi.evaluation.messages import Task


@dataclass(kw_only=True, slots=True)
class BehaviorIclAgent(EmbodiedAgent):
    """Bind the sealed learning package to its evaluation task."""

    icl_task_name: str

    def reset(self, task: Task) -> None:
        """Reject mismatched learning before starting the framework lifecycle."""
        if (task.benchmark, task.task_id) != ("behavior_1k", self.icl_task_name):
            raise ValueError(
                f"ICL package is for behavior_1k/{self.icl_task_name}, "
                f"but evaluation requested {task.benchmark}/{task.task_id}"
            )
        EmbodiedAgent.reset(self, task)


def prepare_workspace(config: AgentConfig, workspace: Path) -> None:
    """Install the declared launcher and instructions in a fresh case workspace."""
    for source, destination in (
        ("README.md", "WORKSPACE.md"),
        ("codex_mcp.py", "codex_mcp.py"),
    ):
        target = workspace / destination
        if target.exists() or target.is_symlink():
            raise ValueError(f"ICL payload uses reserved workspace path: {destination}")
        shutil.copy2(config.resources["workspace"] / source, target)
        if source == "codex_mcp.py":
            target.chmod(
                target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
            )


def make_agent(case_dir: Path, *, config: AgentConfig) -> EmbodiedAgent:
    """Require learned ICL resources and use the framework's selected planner."""
    if config.parameters:
        raise ValueError("the official ICL factory has no custom parameters")
    if set(config.resources) != {"workspace", "icl"}:
        raise ValueError(
            "prepare a task ICL package and declare resources.icl before evaluation"
        )
    package = ContextPackage.load(config.resources["icl"])
    task_name = package.metadata.get("task_name")
    if (
        package.metadata.get("benchmark") != "behavior_1k"
        or package.metadata.get("action_profile") != "r1pro_eef21_v1"
        or not isinstance(task_name, str)
        or not task_name.strip()
        or package.metadata.get("stage") != "icl"
    ):
        raise ValueError(
            "ICL package does not identify a learned official BEHAVIOR task"
        )
    for name in ("TASK.md", "SOURCES.md"):
        if not (package.root / "payload" / name).read_text(encoding="utf-8").strip():
            raise ValueError(f"ICL package needs nonempty {name}")
    workspace = materialize_package(package, case_dir / "workspace")
    prepare_workspace(config, workspace.root)
    configured = config.embodied_agent(case_dir, working_dir=workspace.root)
    return BehaviorIclAgent(
        **{
            field.name: getattr(configured, field.name)
            for field in fields(EmbodiedAgent)
            if field.init
        },
        icl_task_name=task_name,
    )
