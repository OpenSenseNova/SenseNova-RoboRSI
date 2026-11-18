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

"""Lazy native SIMPLE runtime; all simulator access belongs to its worker thread."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from roborsi.evaluation.messages import Task
from robots.simple.catalog import Catalog
from robots.simple.contracts import SIMPLE_REVISION, reset_instruction, validate_task
from robots.simple.runtime_setup import configure_runtime, register_compatibility_task


class ChunkClient:
    """Feed explicit rows into the original Psi0 agent's action conversion."""

    def __init__(self) -> None:
        self.rows = None

    def query_action(self, *args: Any, **kwargs: Any) -> tuple[Any, dict]:
        if self.rows is None:
            raise RuntimeError("native action converter requested an absent chunk")
        rows, self.rows = self.rows, None
        return rows, {}


class SimpleRuntime:
    """Retain original controller conversion, reset and task success checks."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.env = self.agent = None
        self.client = ChunkClient()
        self.task = self.observation = self.info = None

    def start(self, task: Task) -> dict:
        """Load a pinned native checkout and a checked private reset."""
        spec = validate_task(task)
        self.catalog = Catalog(
            Path(
                os.environ.get(
                    "ROBORSI_SIMPLE_CATALOG",
                    "/roborsi/assets/simple-catalog/catalog.json",
                )
            )
        )
        state = self.catalog.reset(task)
        root = Path(os.environ["ROBORSI_SIMPLE_ROOT"]).resolve()
        # The image checkout belongs to root; the native worker runs as the
        # caller. Trust only this explicitly configured checkout and put LFS
        # scratch data in the owned episode directory. Do not inherit arbitrary
        # Git configuration or credentials from the HTTP service environment.
        git = [
            "git",
            "-c",
            f"safe.directory={root}",
            "-c",
            f"lfs.storage={self.directory / 'native' / 'lfs'}",
            "-C",
            str(root),
        ]
        commit = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
        if commit != SIMPLE_REVISION:
            raise ValueError("SIMPLE checkout differs from the pinned revision")
        # A commit label alone does not identify modified evaluator/controller files.
        if subprocess.run(
            [*git, "diff", "--quiet", "HEAD", "--", "src/simple"],
            check=False,
        ).returncode:
            raise ValueError(
                "SIMPLE tracked source changes require a separately validated profile"
            )
        if os.environ.get("OMNI_KIT_ACCEPT_EULA") != "Y":
            raise ValueError("accept the NVIDIA EULA before starting Isaac Sim")
        resource_root = (
            root / "src/simple/robots/policy"
            if spec["controller"] == "amo"
            else root
            / "third_party/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1"
        )
        for resource in resource_root.rglob("*"):
            if resource.is_file() and resource.suffix.lower() in {
                ".pt",
                ".stl",
                ".onnx",
            }:
                with resource.open("rb") as stream:
                    if stream.read(80).startswith(
                        b"version https://git-lfs.github.com/spec/v1"
                    ):
                        raise ValueError(
                            f"materialize upstream Git LFS resource before simulation: {resource.name}"
                        )
        for path in [
            root / "src",
            root / "third_party/GR00T-WholeBodyControl",
            root / "third_party",
        ]:
            sys.path.insert(0, str(path))
        import numpy as np
        import simple
        import simple.utils

        if not Path(simple.__file__).resolve().is_relative_to(root):
            raise ValueError(
                "another SIMPLE installation shadows the requested checkout"
            )
        assets = os.environ.get("ROBORSI_SIMPLE_ASSETS")
        if assets:
            asset_path = Path(assets).resolve()
            if not asset_path.is_dir():
                raise ValueError("SIMPLE assets directory is missing")
            simple.utils.get_data_dir = lambda: asset_path
            from simple.core.simulator import Simulator

            # The native simulator has its own resolver, separate from utils.
            Simulator.get_data_dir = lambda _self: asset_path
        runtime_facts = configure_runtime(self.directory / "native", spec["controller"])
        import gymnasium as gym
        import simple.envs  # noqa: F401

        runtime_facts["compatibility_task"] = register_compatibility_task(
            task.native["alias"],
            Path(os.environ.get("ROBORSI_SIMPLE_COMPAT_ROOT", "/opt/simple-compat")),
        )

        self.task = task
        self.controller = spec["controller"]
        native = task.native
        kwargs: dict[str, Any] = {
            "headless": True,
            "sim_mode": "mujoco_isaac",
            "render_hz": 50,
        }
        config = None
        if self.controller == "amo":
            from simple.baselines.psi0 import Psi0Agent as Agent
            from simple.envs.wrappers.stand_stabilization import (
                StandStabilizationWrapper,
            )

            kwargs["success_criteria"] = 0.9
        else:
            from simple.baselines.psi0_decoupled_wbc import (
                Psi0DecoupledWbcAgent as Agent,
            )
            from simple.cli.eval_decoupled_wbc import _make_sonic_config

            config = _make_sonic_config()
            config["DOMAIN_ID"] = int(os.environ.get("ROBORSI_SIMPLE_DDS_DOMAIN", "21"))
            kwargs["sonic_config"] = config
        self.env = gym.make("simple/" + spec["env_id"], **kwargs).unwrapped
        if self.env.task.metadata["max_episode_steps"] != native["official_steps"]:
            raise ValueError("native SIMPLE task budget changed")
        if self.controller != "amo" and native["alias"] != "mobile_pick":
            self.env.task.success_criteria = 0.5
            self.env.task.metadata["success_criteria"] = 0.5
        self.agent = Agent(
            self.env.task.robot,
            "127.0.0.1",
            1,
            **({"sonic_config": config} if config else {}),
        )
        self.agent.client = self.client
        random.seed(task.seed)
        np.random.seed(task.seed)
        if self.controller == "amo":
            self.observation, self.info = StandStabilizationWrapper(self.env).reset(
                options={"state_dict": state}
            )
            self.agent.reset()
        else:
            self.observation, self.info = self.env.reset(options={"state_dict": state})
            self.agent.reset()
            self.agent._wbc_policy.lower_body_policy.use_policy_action = True
            for _ in range(300):
                if self.env.task.robot.stabilized:
                    break
                started = time.monotonic()
                action = self.agent.get_stabilize_action(self.observation)
                self.observation, *_, self.info = self.env.step(action)
                self.env.update_viewer()
                self.env.update_reward()
                time.sleep(max(0, 0.02 - (time.monotonic() - started)))
            if not self.env.task.robot.stabilized:
                raise RuntimeError("native WBC stabilization did not complete")
            import torch

            self.agent._wbc_policy.lower_body_policy.gait_indices = torch.zeros(
                (1), dtype=torch.float32
            )
        native_instruction = self.env.task.instruction
        if native_instruction != reset_instruction(state):
            raise ValueError(
                "native instruction differs from the released reset language"
            )
        (self.directory / "native/reset_verification.json").write_text(
            json.dumps(
                {
                    "reset_sha256": hashlib.sha256(
                        json.dumps(state, sort_keys=True).encode()
                    ).hexdigest(),
                    "simple_revision": commit,
                    "seed": task.seed,
                    "official_steps": native["official_steps"],
                    "execution_steps": task.max_steps,
                    "score_protocol": native["score_protocol"],
                    "official_score": False,
                    "agent_instruction": task.instruction,
                    "native_instruction": native_instruction,
                    "instruction_match": native_instruction == task.instruction,
                    "runtime": runtime_facts,
                },
                indent=2,
            )
            + "\n"
        )
        return self._public()

    def _public(self) -> dict:
        import numpy as np

        q = np.asarray(self.observation["joint_qpos"])
        if q.shape != (43,):
            raise ValueError("unexpected G1/Dex3 joint feedback ordering")
        indices = [
            *range(29, 32),
            *range(34, 36),
            *range(32, 34),
            *range(36, 43),
            *range(15, 29),
            13,
            14,
            12,
        ]
        state = q[indices].tolist()
        if self.controller == "amo":
            state[28:31] = self.agent._last_cmd_torso_rpyh[:3].tolist()
            height = self.agent._last_cmd_torso_rpyh[3]
        else:
            height = self.agent._last_base_height_cmd
        return {
            "state": [*state, float(height)],
            "rgb": self.observation["head_stereo_left"],
        }

    def submit(self, rows: list[list[float]]) -> None:
        """Provide one chunk to the original converter without opening a model service."""
        import numpy as np

        if self.client.rows is not None or len(self.agent._action_queue):
            raise RuntimeError("previous native action chunk was not drained")
        self.client.rows = np.asarray(rows, dtype=np.float32)

    def step(self) -> tuple[dict, bool, bool]:
        """Advance one original native action and retain privileged scoring locally."""
        action = self.agent.get_action(
            self.observation, info=self.info, instruction=self.task.instruction
        )
        self.observation, _, terminated, truncated, self.info = self.env.step(action)
        return self._public(), bool(self.env._success), bool(terminated or truncated)

    def close(self) -> None:
        """Close the owned simulator after the parent has saved result evidence."""
        if self.env is not None:
            env, self.env = self.env, None
            env.close()


def make_runtime(directory: Path) -> SimpleRuntime:
    """Construct a lazy native runtime; optional simulator imports happen at start."""
    return SimpleRuntime(directory)
