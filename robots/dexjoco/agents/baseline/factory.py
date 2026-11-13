# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build the historical file protocol on RoboRSI's common CLI backend."""

from __future__ import annotations

import os
from pathlib import Path

from roborsi.agents.config import AgentConfig
from roborsi.planner.codex_cli_config import CodexCliConfig
from robots.dexjoco.agents.baseline.policy import FrozenPolicyAgent
from robots.dexjoco.reference import load_reference


def make_agent(case_dir: Path, *, config: AgentConfig) -> FrozenPolicyAgent:
    """Consume declared prompts/skills; reject unsupported policy overrides.

    The shared policy prompt and ordered, full-text skills apply to both fresh
    CLI sessions. Phase resources supply the original initial/continuation
    interface context. No model-facing prompt is loaded from the archive.
    """
    config.verify()
    document = config.document
    settings = document.get("agent", {})
    required = {
        "planner": "codex_cli",
        "model": "gpt-6-astra",
        "reasoning_effort": "max",
        "max_turns": None,
        "max_tokens": None,
        "planner_timeout_s": 0,
        "action_timeout_s": None,
    }
    if (
        any(settings.get(key) != value for key, value in required.items())
        or set(settings) - (set(required) | {"codex_cli", "system_prompt"})
        or config.parameters
        or document.get("runtime")
    ):
        raise ValueError(
            "DexJoCo requires Astra/max, codex_cli, unlimited model time/turns "
            "and no runtime/working_dir/parameter overrides"
        )
    resources = config.resources
    if set(resources) != {
        "reference",
        "reference_license",
        "initial_prompt",
        "continuation_prompt",
    }:
        raise ValueError("Declare exactly the DexJoCo reference and phase resources")
    reference = load_reference(resources["reference"])
    cli = CodexCliConfig.from_dict(settings.get("codex_cli", {}))
    flags = [
        "--ignore-user-config",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--sandbox",
        "workspace-write",
    ]
    for feature in reference.isolation.DISABLED:
        flags.extend(("--disable", feature))
    if (
        cli.expected_version != "codex-cli 0.154.0"
        or cli.tools != "external"
        or cli.inherit_env
        or cli.env
        or cli.env_vars
        or cli.provider_capture
        or cli.cli_args != tuple(flags)
        or dict(cli.config_overrides)
        != {
            "approval_policy": "never",
            "web_search": "disabled",
            "sandbox_workspace_write.network_access": False,
            "allow_login_shell": False,
            "project_doc_max_bytes": 0,
        }
    ):
        raise ValueError("DexJoCo requires the pinned CLI isolation/tool settings")

    prompt_file = document.get("prompt_file")
    common = (
        (config.path.parent / prompt_file).read_text()
        if prompt_file is not None
        else settings.get("system_prompt", "")
    )
    if not common.strip():
        raise ValueError("Declare a nonempty prompt_file or agent.system_prompt")
    skill_texts = []
    for name in document.get("skills", []):
        content = (config.path.parent / name).read_text()
        if not content.strip():
            raise ValueError("DexJoCo skills must contain nonempty text")
        skill_texts.append(content)
    skills = "".join(
        f'\n\n<skill index="{index}">\n{content}\n</skill>\n'
        for index, content in enumerate(skill_texts, 1)
    )
    prompts = {}
    for phase in ("initial", "continuation"):
        context = resources[phase + "_prompt"].read_text()
        if not context.strip():
            raise ValueError("DexJoCo phase prompts must contain nonempty text")
        prompts[phase] = context + common + skills
    return FrozenPolicyAgent(
        case_dir,
        resources["reference"],
        Path(os.environ["ROBORSI_DEXJOCO_AGENT_ASSETS"]),
        prompts=prompts,
        cli_config=cli,
        config_fingerprint=config.fingerprint,
    )
