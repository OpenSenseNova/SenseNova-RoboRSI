# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Translate a planner's optional VLA action into scored native LIBERO actions."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

from robots.libero.vla_client import (
    endpoint_url,
    openpi_enabled,
    parse_profile,
    predict,
)

ROOT = Path(__file__).resolve().parents[2]
VLA_SCHEMA = {
    "type": "object",
    "description": "Ask the configured pi05 LIBERO VLA to execute the next short skill from the current images and robot state. Then inspect native feedback and replan.",
    "properties": {
        "primitive": {"const": "vla"},
        "instruction": {"type": "string", "minLength": 1, "maxLength": 2048},
    },
    "required": ["primitive", "instruction"],
    "additionalProperties": False,
}
VLA_PROMPT = "\nVLA is enabled for this run. You may use roborsi_action with primitive='vla' and a short instruction for the next skill, or use native direct controls. Each VLA call executes a bounded action chunk, then returns native observation feedback. Earlier frozen-preset statements that VLA is disabled describe the original condition and do not apply to this configured run. The Benchmark alone determines stopping and success."


def vla_prompt(agent):
    return replace(agent, system_prompt=agent.system_prompt + VLA_PROMPT)


class LiberoVlaAgent:
    def __init__(self, inner, config, case_dir: Path):
        self.inner = inner
        self.case_dir = case_dir
        if "vla_model" not in config.parameters:
            raise ValueError(
                "VLA model is unbound; use evaluate_model.py --model-config"
            )
        self.profile = parse_profile(
            config.parameters["vla_model"], base_dir=case_dir, resolve=False
        )
        endpoint = (
            config.parameters.get("vla_endpoint")
            or os.environ.get(
                config.parameters.get("vla_endpoint_env", "ROBORSI_LIBERO_VLA_URL")
            )
            or self.profile.endpoint
        )
        if not endpoint:
            raise ValueError(
                "VLA service is unavailable; use evaluate_model.py automatic setup or set ROBORSI_LIBERO_VLA_URL"
            )
        self.endpoint = endpoint_url(endpoint)

    def reset(self, task):
        self.inner.reset(
            task.model_copy(
                update={
                    "capabilities": {
                        **task.capabilities,
                        "action_schema": {
                            "oneOf": [task.capabilities["action_schema"], VLA_SCHEMA]
                        },
                    }
                }
            )
        )

    def act(self, observation):
        action = self.inner.act(observation)
        if action.data.get("primitive") != "vla":
            return action
        instruction = action.data.get("instruction")
        if (
            set(action.data) != {"primitive", "instruction"}
            or not isinstance(instruction, str)
            or not 0 < len(instruction.strip()) <= 2048
        ):
            raise ValueError("VLA action requires a nonempty bounded instruction")
        actions = predict(
            self.endpoint, self.profile, observation, self.case_dir, instruction
        )
        native = (
            {"vector": actions[0]}
            if action.benchmark == "libero"
            else {"vectors": actions}
        )
        self.case_dir.mkdir(parents=True, exist_ok=True)
        with (self.case_dir / "vla.jsonl").open("a", encoding="utf-8") as evidence:
            evidence.write(
                json.dumps(
                    {
                        "step_index": observation.step_index,
                        "instruction": instruction,
                        "policy_config": self.profile.policy_config,
                        "openpi_revision": self.profile.openpi_revision,
                        "checkpoint": self.profile.checkpoint,
                        "action": native,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
        return action.model_copy(update={"data": native})

    def observe_end(self, step):
        self.inner.observe_end(step)

    def close(self):
        self.inner.close()


def make_agent(case_dir: Path, *, config):
    """Standard LIBERO keeps the default Agent unless VLA was explicitly enabled."""
    enabled = openpi_enabled(config)
    from robots.libero.three_benchmark_agent import _mcp_server

    server = _mcp_server(
        config.parameters,
        "vla",
        config.parameters.get("enable_vla", False) and not enabled,
    )
    agent = config.embodied_agent(case_dir, mcp_servers=(server,) if server else ())
    return LiberoVlaAgent(vla_prompt(agent), config, case_dir) if enabled else agent
