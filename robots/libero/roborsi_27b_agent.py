# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
"""Opt-in 27B Agent policy; the original LIBERO Agent remains unchanged."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, ValidationError

from roborsi.agents.config import AgentConfig
from roborsi.agents.context_package import load_package_map
from roborsi.data_convert import TextDocument
from roborsi.evaluation.messages import Task
from roborsi.evaluation.result import write_json_atomic
from roborsi.tools.toolkit import ToolResult
from robots.libero.three_benchmark_agent import (
    ThreeBenchmarkIclAgent,
    _boolean,
    _context_root,
    _initial_context,
)


def _validation_retries(parameters: dict[str, Any]) -> int:
    value = parameters.get("action_validation_max_retries", 0)
    if type(value) is not int or not 0 <= value <= 3:
        raise ValueError(
            "parameters.action_validation_max_retries must be an integer from 0 to 3"
        )
    return value


class _RecoveringEpisode:
    """Return invalid deferred calls to the model before the benchmark sees them."""

    def __init__(self, episode, connector, retries: int):
        self._episode = episode
        self._connector = connector
        self._retries = retries

    def __getattr__(self, name):
        return getattr(self._episode, name)

    def _record(self, event, attempt, call, **detail):
        connector = self._connector
        path = connector._output_dir / "action_recovery.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": 1,
            "event": event,
            "time": time.time(),
            "episode_id": connector._episode_id,
            "step_index": connector._last_step + 1,
            "attempt": attempt,
            "submitted": call.arguments,
            **detail,
        }
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def next_call(self, *, timeout=None):
        connector = self._connector
        schema = connector._task.capabilities["action_schema"]
        validator = Draft202012Validator(schema)
        deadline = None if timeout is None else time.monotonic() + timeout
        for attempt in range(self._retries + 1):
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            call = self._episode.next_call(timeout=remaining)
            if call is None or call.name != connector._action_tool_name:
                return (
                    call  # Original connector handles ended episodes and wrong tools.
                )
            try:
                validator.validate(call.arguments)
            except ValidationError as error:
                exhausted = attempt == self._retries
                self._record(
                    "exhausted" if exhausted else "rejected",
                    attempt + 1,
                    call,
                    error=error.message[:2000],
                    executed=False,
                    retries_remaining=self._retries - attempt,
                )
                if exhausted:
                    raise  # Preserve strict failure after the bounded correction budget.
                self._episode.complete(
                    call,
                    ToolResult(
                        call.name,
                        {
                            "status": "invalid_action",
                            "executed": False,
                            "physical_steps_used": 0,
                            "step_index": connector._last_step + 1,
                            "error": error.message[:2000],
                            "submitted": call.arguments,
                            "action_schema": schema,
                            "retries_remaining": self._retries - attempt,
                            "instruction": (
                                "The action was rejected before execution. The environment is unchanged. "
                                "Correct the arguments using the action schema and submit a new "
                                "roborsi_action for the same observation. Do not assume any motion occurred."
                            ),
                        },
                    ),
                )
            else:
                if attempt:
                    self._record("accepted", attempt + 1, call, executed=False)
                return call


class _RecoveringPlanner:
    """Wrap only this Agent instance; shared planner and connector code stay intact."""

    def __init__(self, planner, connector, retries):
        self._planner = planner
        self._connector = connector
        self._retries = retries

    def __getattr__(self, name):
        return getattr(self._planner, name)

    def start_episode(self, *args, **kwargs):
        return _RecoveringEpisode(
            self._planner.start_episode(*args, **kwargs), self._connector, self._retries
        )


def _enable_action_recovery(agent, retries: int) -> None:
    if retries:
        connector = agent._evaluation_connector
        if connector is None or connector._episode is not None:
            raise ValueError(
                "action recovery must be enabled immediately after Agent reset"
            )
        connector._planner = _RecoveringPlanner(connector._planner, connector, retries)


IMAGE_ROTATION_CONVENTIONS = (
    "Image-tool rotation conventions (all angles in radians): pitch is a custom "
    "world-X tilt, NOT the usual Euler pitch about Y. For the end-effector "
    "rotation matrix R in world coordinates, pitch = atan2(R[1,2], -R[2,2]). "
    "Pitch 0 means the gripper z-axis points down along world -Z; +pi/2 points "
    "along world +Y; -pi/2 points along world -Y. rotate_pitch drives the "
    "world-X rotation action[3]. Yaw = atan2(R[1,0], R[0,0]), the angle of "
    "the end-effector x-axis projected onto world XY; positive yaw rotates "
    "from world +X toward +Y about world +Z. rotate_wrist drives action[5]. "
    "move_pose target_pitch/target_yaw use these SAME definitions. For rotation "
    "primitives, target is an absolute angle and delta is relative to the "
    "current angle; provide exactly one. states[3:6] is an axis-angle rotation "
    "vector, NOT these pitch/yaw values or Euler angles. Do not read its "
    "individual components as pitch or yaw. Pitch/yaw control does not specify "
    "an arbitrary full 3D orientation."
)


def _model_action_task(task: Task, parameters: dict[str, Any]) -> Task:
    """Annotate only this Agent's private schema; never change valid actions."""
    types = _boolean(parameters, "expose_action_parameter_types", False)
    rotations = _boolean(parameters, "explain_image_rotation_conventions", False)
    if not (types or rotations):
        return task
    from robots.libero.image_tools import _expose_action_property_types

    capabilities = copy.deepcopy(task.capabilities)
    schema = capabilities["action_schema"]
    if types:
        _expose_action_property_types(schema)
    if rotations:
        for branch in schema["oneOf"]:
            primitive = branch.get("properties", {}).get("primitive", {})
            names = primitive.get("enum", [primitive.get("const")])
            if set(names) & {"move_to", "move_pose", "rotate_pitch", "rotate_wrist"}:
                branch["description"] = (
                    branch.get("description", "") + " " + IMAGE_ROTATION_CONVENTIONS
                )
    return task.model_copy(update={"capabilities": capabilities})


class Roborsi27BIclAgent(ThreeBenchmarkIclAgent):
    """Reuse lifecycle and ICL helpers, with the selected image-tool policy."""

    def reset(self, task: Task) -> None:
        if self._agent is not None:
            raise ValueError("ICL Agent owns only one evaluation case")
        from robots.libero.image_tools import PROFILE, enabled, tool_action_schema
        from robots.libero.image_tools import action_schema as image_schema

        image_tools = enabled(self._config.parameters)
        if image_tools and (
            task.benchmark != "libero_pro"
            or task.capabilities.get("image_observation_tools") != PROFILE
        ):
            raise ValueError("Agent requires the image observation tool service")
        expected_tools = all(
            _boolean(self._config.parameters, key, False)
            for key in ("enable_vla", "enable_sam3")
        )
        if self._config.parameters.get(
            "tool_transport"
        ) == "openpi_sam3" and task.capabilities.get("action_schema") != (
            image_schema(expected_tools)
            if image_tools
            else tool_action_schema(expected_tools)
        ):
            raise ValueError("Agent tool mode differs from the Benchmark service")
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
        if image_tools:
            from robots.libero.image_tools import observation_prompt

            agent = observation_prompt(agent)
        agent = replace(
            agent,
            evaluation_skills=package.skills,
            evaluation_skill_paths=package.skill_paths,
            evaluation_memory=memory,
            evaluation_initial_context=_initial_context(package),
        )
        agent.reset(_model_action_task(task, self._config.parameters))
        _enable_action_recovery(agent, _validation_retries(self._config.parameters))
        self._agent = agent


def make_agent(case_dir: Path, *, config: AgentConfig) -> Roborsi27BIclAgent:
    """Create a strict ICL Agent from config-declared package resources."""
    parameters = config.parameters
    allowed = {
        "tool_transport",
        "action_validation_max_retries",
        "expose_action_parameter_types",
        "explain_image_rotation_conventions",
        "enable_observation_tools",
        "context_root_env",
        "enable_vla",
        "enable_sam3",
        "vla_url_env",
        "vla_endpoint",
        "vla_endpoint_env",
        "vla_model",
        "sam3_url_env",
    }
    _validation_retries(parameters)
    _boolean(parameters, "expose_action_parameter_types", False)
    _boolean(parameters, "explain_image_rotation_conventions", False)
    _boolean(parameters, "enable_observation_tools", False)
    if parameters.get("tool_transport") != "openpi_sam3":
        raise ValueError("unsupported LIBERO tool transport")
    if parameters.get("tool_transport") == "openpi_sam3" and _boolean(
        parameters, "enable_vla", False
    ) != _boolean(parameters, "enable_sam3", False):
        raise ValueError("the HTTP RPC profile enables or disables VLA/SAM3 together")
    if set(parameters) - allowed:
        raise ValueError("unknown LIBERO three-benchmark Agent parameter")
    if set(config.resources) != {"context_map"}:
        raise ValueError("Agent resources need context_map")
    packages = load_package_map(
        config.resources["context_map"], _context_root(parameters)
    )
    return Roborsi27BIclAgent(case_dir, config, packages, ())
