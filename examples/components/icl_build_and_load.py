# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Assemble references using existing EmbodiedAgent evaluation arguments."""

from pathlib import Path

from roborsi.components.icl import ContextPackage, materialize_package
from roborsi.components.icl.adapters import context_from, make_icl_tools
from roborsi.embodied_agent import EmbodiedAgent
from roborsi.llm import LLMConfig


def make_agent(
    package: ContextPackage, case_dir: Path, llm: LLMConfig
) -> EmbodiedAgent:
    """Give one evaluation case its own mutable reference copy."""
    workspace = materialize_package(package, case_dir / "icl-workspace")
    context = context_from(workspace, planner="api")
    return EmbodiedAgent(
        mcp_servers=(),
        output_dir=case_dir,
        planner="api",
        llm=llm,
        local_tools=make_icl_tools(workspace),
        evaluation_memory=context.memory,
        evaluation_skills=context.skills,
        evaluation_skill_paths=context.skill_paths,
        evaluation_initial_context=context.initial_context,
    )
