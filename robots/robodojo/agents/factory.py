# Copyright 2026 The RoboRSI Contributors.
# Licensed under the Apache License, Version 2.0.

"""Expose the submitted skillpack policy through the standard Agent factory."""

from pathlib import Path

from roborsi.evaluation.media import CaseMediaChannel
from roborsi.utils.config import require_project_env

from .bridge import SubmissionAgent

_ALLOWED = {
    "vla_backend",
    "checkpoint_path",
    "provider_kind",
    "sampling",
    "storage_mode",
    "eef_execution_backend",
    "eef_execution_backend_by_task",
    "reasoning_effort",
    "skill_library_mode",
    "official_description_bundle",
    "official_description_manifest_sha256",
    "episode_seconds",
    "agent_exit_policy",
}


def make_agent(case_dir: Path, *, config) -> SubmissionAgent:
    """Load deployment controls without importing a model or the simulator."""
    if set(config.parameters) - _ALLOWED:
        raise ValueError("Unknown submitted RoboDojo deployment parameter")
    return SubmissionAgent(
        case_dir,
        parameters=config.parameters,
        media=CaseMediaChannel(Path(require_project_env("ROBORSI_MEDIA_DIR"))),
    )
