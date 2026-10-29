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

"""Train-only shared exploration through EmbodiedAgent and public Benchmark HTTP."""

from __future__ import annotations

import shutil
import uuid
from contextlib import ExitStack
from pathlib import Path, PurePosixPath

from roborsi.agents.config import AgentConfig
from roborsi.embodied_agent import LocalToolSpec
from roborsi.evaluation.benchmark_client import BenchmarkClient
from roborsi.evaluation.media import CaseMediaChannel
from roborsi.evaluation.messages import Action, Observation
from roborsi.tools.toolkit import ToolResult
from robots.vln_shared.benchmark import NavigationBenchmark, action_schema, serve
from robots.vln_shared.storage import freeze_snapshot, write_json


class TrainingTools:
    """Expose only training tasks; own a single currently active training scene."""

    def __init__(self, training: dict, directory: Path, workspace: Path) -> None:
        self.training, self.directory, self.workspace = training, directory, workspace
        self.episodes = {e["id"]: e for e in training["episodes"]}
        self.stack = ExitStack()
        self.client = None
        self.observation = None
        self.ready = False
        self.resets = 0
        self.attempts = 0

    def _feedback(self, name: str, observation: Observation) -> ToolResult:
        self.client.collect_observation(observation, self.channel, timeout_s=600)
        data = dict(observation.data)
        for key, ref in zip(
            ("_image_bytes", "_image_cam_bytes", "_image_nav_bytes"), observation.media
        ):
            data[key] = self.channel.read(
                ref, observation.episode_id, observation.step_index
            )
        return ToolResult(name, data)

    def browse(self, arguments: dict) -> ToolResult:
        """Page through public instructions without goal or test annotations."""
        offset = arguments["offset"]
        episodes = list(self.episodes.values())
        return ToolResult(
            "training_tasks",
            {
                "total": len(episodes),
                "tasks": [
                    {"id": e["id"], "instruction": e["instruction"]}
                    for e in episodes[offset : offset + 20]
                ],
            },
        )

    def reset(self, arguments: dict) -> ToolResult:
        """Reset an original training annotation in a fresh native process."""
        episode = self.episodes[arguments["id"]]
        self.close()
        self.stack = ExitStack()
        self.attempts += 1
        directory = self.directory / f"practice-{self.attempts:04d}"
        provider = NavigationBenchmark(self.training["spec"], [episode], 5000)
        endpoint = self.stack.enter_context(serve(provider, directory / "service"))
        self.client = BenchmarkClient(endpoint)
        self.stack.callback(self.client.close)
        self.channel = CaseMediaChannel(directory / "client/media")
        result = self.client.start(
            provider.tasks()[0], "practice-" + uuid.uuid4().hex, timeout_s=600
        )
        self.observation = result.observation
        feedback = self._feedback("reset_training", self.observation)
        self.resets += 1
        return feedback

    def act(self, arguments: dict) -> ToolResult:
        """Perform a training batch and expose native feedback after termination."""
        if self.observation is None or self.client is None:
            raise ValueError("reset a training task first")
        observation = self.observation
        result = self.client.step(
            Action(
                benchmark=observation.benchmark,
                episode_id=observation.episode_id,
                step_index=observation.step_index,
                data=arguments,
            ),
            timeout_s=600,
        )
        self.observation = result.observation
        reply = self._feedback("training_action", self.observation)
        if result.terminated or result.truncated:
            native = self.client.finish(timeout_s=600)
            self.client.collect_native(
                native, self.channel.media_dir.parent, timeout_s=600
            )
            reply.result["native_result"] = native.details
        return ToolResult("training_action", reply.result)

    def remember(self, arguments: dict) -> ToolResult:
        """Save transferable text inside the private exploration workspace."""
        path = PurePosixPath(arguments["path"])
        if (
            path.is_absolute()
            or ".." in path.parts
            or not path.parts
            or "\\" in str(path)
        ):
            raise ValueError("memory path must be relative")
        target = self.workspace / path
        if not target.resolve().is_relative_to(self.workspace.resolve()):
            raise ValueError("memory path escapes workspace")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(arguments["content"], encoding="utf-8")
        return ToolResult("save_memory", {"saved": str(path)})

    def complete(self, arguments: dict) -> ToolResult:
        """Require practice and a nonempty document before freezing exploration."""
        if (
            not self.resets
            or not (self.workspace / "EXPLORATION.md").read_text().strip()
        ):
            raise ValueError("practice and write EXPLORATION.md before ready")
        self.ready = True
        return ToolResult("ready", {"_finish": True, "status": "success"})

    def specs(self) -> list[LocalToolSpec]:
        """Declare model-visible tools and bounded argument schemas."""

        def obj(properties: dict) -> dict:
            return {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            }

        return [
            LocalToolSpec(
                "training_tasks",
                "List 20 training tasks at an offset.",
                obj({"offset": {"type": "integer", "minimum": 0}}),
                self.browse,
            ),
            LocalToolSpec(
                "reset_training",
                "Start or restart a training task.",
                obj({"id": {"type": "string"}}),
                self.reset,
            ),
            LocalToolSpec(
                "training_action",
                "Execute native training actions.",
                action_schema(self.training["spec"]["dataset"]),
                self.act,
            ),
            LocalToolSpec(
                "save_memory",
                "Save transferable navigation skills.",
                obj(
                    {
                        "path": {"type": "string"},
                        "content": {"type": "string", "maxLength": 1000000},
                    }
                ),
                self.remember,
            ),
            LocalToolSpec(
                "ready",
                "Finish exploration after writing EXPLORATION.md.",
                obj({}),
                self.complete,
            ),
        ]

    def close(self) -> None:
        """Close the current client before stopping its HTTP server."""
        self.stack.close()
        self.client = None
        self.observation = None


def explore(training: dict, preset: Path, directory: Path, snapshot: Path) -> None:
    """Run one exploration session and publish a frozen snapshot only on ready."""
    directory.mkdir(parents=True, exist_ok=False)
    workspace = directory / "workspace"
    workspace.mkdir()
    tools = TrainingTools(training, directory, workspace)
    config = AgentConfig.from_file(preset)
    (workspace / ".codex").mkdir()
    shutil.copy2(config.resources["tool_policy"], workspace / ".codex/config.toml")
    agent = config.embodied_agent(
        directory, working_dir=workspace, local_tools=tools.specs()
    )
    try:
        prompt = config.resources["exploration_prompt"].read_text(encoding="utf-8")
        result = agent.run(
            "Explore the training tasks, write EXPLORATION.md, then call ready.",
            system_prompt=prompt,
            output_dir=directory,
        )
        write_json(
            directory / "exploration-result.json",
            {
                "error": result.error,
                "ready": tools.ready,
                "stats": result.stats,
                "practice_resets": tools.resets,
            },
        )
        if result.error or not tools.ready:
            raise RuntimeError(
                "exploration did not complete ready; see exploration-result.json"
            )
    finally:
        agent.close()
        tools.close()
    freeze_snapshot(workspace, snapshot)
