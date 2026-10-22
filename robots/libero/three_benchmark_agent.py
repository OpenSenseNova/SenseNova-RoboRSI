# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Agent factory that binds sealed ICL packages and optional VLA/SAM3 MCP tools."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from roborsi.agents.config import AgentConfig
from roborsi.agents.context_package import ContextPackage, load_package_map
from roborsi.data_convert import TextDocument
from roborsi.embodied_agent import EmbodiedAgent, McpServer
from roborsi.evaluation.messages import Action, Observation, StepResult, Task
from roborsi.evaluation.result import write_json_atomic
from roborsi.utils.config import get_project_env
from robots.libero.vla_client import openpi_enabled


def _boolean(parameters: dict[str, Any], name: str, default: bool) -> bool:
    value = parameters.get(name, default)
    if type(value) is not bool:
        raise ValueError(f"parameters.{name} must be a boolean")
    return value


def _mcp_server(
    parameters: dict[str, Any], name: str, enabled: bool
) -> McpServer | None:
    if not enabled:
        return None
    field = f"{name}_url_env"
    env_name = parameters.get(field, f"ROBORSI_{name.upper()}_MCP_URL")
    if not isinstance(env_name, str) or not env_name.strip():
        raise ValueError(f"parameters.{field} must name an environment variable")
    url = get_project_env(env_name)
    if not url:
        raise ValueError(f"{name.upper()} is enabled but {env_name} is unset")
    return McpServer(name=name, url=url)


def _context_root(parameters: dict[str, Any]) -> Path:
    env_name = parameters.get("context_root_env", "ROBORSI_LIBERO_ICL_ROOT")
    if not isinstance(env_name, str) or not env_name.strip():
        raise ValueError(
            "parameters.context_root_env must name an environment variable"
        )
    value = get_project_env(env_name)
    if not value:
        raise ValueError(f"sealed ICL packages require {env_name}")
    root = Path(os.path.abspath(value))
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{env_name} must name a real package directory")
    return root


def _initial_context(package: ContextPackage) -> tuple[Any, ...]:
    parts = []
    for entry in package.initial_context:
        if entry.type == "text":
            parts.append(entry.path.read_text(encoding="utf-8"))
        else:
            from pydantic_ai import BinaryContent

            media_type = entry.media_type
            if media_type is None:
                media_type = {
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".png": "image/png",
                    ".webp": "image/webp",
                }.get(entry.path.suffix.lower())
            if media_type is None or not media_type.startswith("image/"):
                raise ValueError("image context requires an image media_type or suffix")
            parts.append(
                BinaryContent(data=entry.path.read_bytes(), media_type=media_type)
            )
    return tuple(parts)


class ThreeBenchmarkIclAgent:
    """Resolve one immutable context package when the RoboRSI task is bound."""

    def __init__(
        self,
        case_dir: Path,
        config: AgentConfig,
        packages: dict[str, ContextPackage],
        servers: tuple[McpServer, ...],
    ) -> None:
        self._case_dir = case_dir
        self._config = config
        self._packages = packages
        self._servers = servers
        self._agent: EmbodiedAgent | None = None

    def reset(self, task: Task) -> None:
        if self._agent is not None:
            raise ValueError("ICL Agent owns only one evaluation case")
        package = self._packages.get(task.task_id)
        if package is None:
            raise ValueError(f"no sealed ICL package is mapped for {task.task_id}")
        write_json_atomic(
            self._case_dir / "context-package.json",
            {
                "schema_version": 1,
                "task_id": task.task_id,
                "package_id": package.package_id,
                "manifest_sha256": package.manifest_sha256,
            },
        )
        memory = tuple(
            TextDocument(title, path.read_text(encoding="utf-8"), path.name)
            for title, path in package.memory
        )
        agent = self._config.embodied_agent(self._case_dir, mcp_servers=self._servers)
        if openpi_enabled(self._config):
            from robots.libero.vla_agent import vla_prompt

            agent = vla_prompt(agent)
        agent = replace(
            agent,
            evaluation_skills=package.skills,
            evaluation_skill_paths=package.skill_paths,
            evaluation_memory=memory,
            evaluation_initial_context=_initial_context(package),
        )
        agent.reset(task)
        self._agent = agent

    def act(self, observation: Observation) -> Action:
        if self._agent is None:
            raise ValueError("ICL Agent has not been reset")
        return self._agent.act(observation)

    def observe_end(self, step: StepResult) -> None:
        if self._agent is None:
            raise ValueError("ICL Agent has not been reset")
        self._agent.observe_end(step)

    def close(self) -> None:
        if self._agent is not None:
            self._agent.close()


def make_three_benchmark_agent(
    case_dir: Path, *, config: AgentConfig
) -> ThreeBenchmarkIclAgent:
    """Create a strict ICL Agent from config-declared package resources."""
    parameters = config.parameters
    allowed = {
        "context_root_env",
        "enable_vla",
        "enable_sam3",
        "vla_url_env",
        "vla_endpoint",
        "vla_endpoint_env",
        "vla_model",
        "sam3_url_env",
    }
    if set(parameters) - allowed:
        raise ValueError("unknown LIBERO three-benchmark Agent parameter")
    if set(config.resources) != {"context_map"}:
        raise ValueError("Agent resources need context_map")
    packages = load_package_map(
        config.resources["context_map"], _context_root(parameters)
    )
    servers = tuple(
        server
        for server in (
            _mcp_server(
                parameters,
                "vla",
                _boolean(parameters, "enable_vla", False)
                and not openpi_enabled(config),
            ),
            _mcp_server(parameters, "sam3", _boolean(parameters, "enable_sam3", False)),
        )
        if server is not None
    )
    agent = ThreeBenchmarkIclAgent(case_dir, config, packages, servers)
    if openpi_enabled(config):
        from robots.libero.vla_agent import LiberoVlaAgent

        return LiberoVlaAgent(agent, config, case_dir)
    return agent
