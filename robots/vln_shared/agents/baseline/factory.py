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

"""Shared navigation Agent factory with declared, relocatable training memory."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from roborsi.agents.config import AgentConfig
from roborsi.embodied_agent import EmbodiedAgent


def make_agent(case_dir: Path, *, config: AgentConfig) -> EmbodiedAgent:
    """Create a fresh CLI session and a private copy of the frozen training memory."""
    workspace = case_dir / "workspace"
    snapshot = config.resources.get("exploration")
    if snapshot is None:
        workspace.mkdir(exist_ok=False)
    else:
        manifest = json.loads((snapshot / "snapshot.json").read_text())
        source = snapshot / "workspace"
        actual = {}
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                raise ValueError("exploration resource contains a symlink")
            if path.is_file():
                actual[path.relative_to(source).as_posix()] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
        if actual != manifest["files"]:
            raise ValueError("exploration resource changed")
        shutil.copytree(source, workspace)
    policy = config.resources["tool_policy"]
    (workspace / ".codex").mkdir(exist_ok=True)
    shutil.copy2(policy, workspace / ".codex/config.toml")
    return config.embodied_agent(case_dir, working_dir=workspace)
