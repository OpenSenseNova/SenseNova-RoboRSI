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

"""Bind the pinned AgentCanvas LHPR runtime without importing its agent harness."""

from __future__ import annotations

import base64
import importlib
import io
import sys
from pathlib import Path
from types import ModuleType


class Episode:
    """Preserve native STOP, force-escape, annotated starts and fixed-object scoring."""

    actions = ("stop", "move_forward", "turn_left", "turn_right")

    def __init__(self, spec: dict) -> None:
        package = ModuleType("lhpr_native")
        package.__path__ = [spec["runtime_root"]]
        sys.modules["lhpr_native"] = package
        module = importlib.import_module("lhpr_native._runtime")
        protocol = module.Protocol(
            max_steps=spec["max_steps"],
            seed=spec["seed"],
            invalid_streak_seed=spec["seed"],
            fixed_object_success=True,
            region_segmentation_dir=spec["regions_root"],
            invalid_streak_strategy="force_escape",
        )
        extra = {}
        if spec.get("external_scenes"):
            extra = {
                "scenes_root": spec["scenes_root"],
                "scene_dataset": spec["scene_dataset"],
            }
        self.runtime = module.LHVLNRuntime(
            Path(spec["data_root"]),
            protocol,
            gpu_id=spec.get("gpu_device_id", -1),
            **extra,
        )
        try:
            self.runtime.load_split(spec["split"])
            metadata = self.runtime.metadata(spec["episode_index"])
            if str(metadata["episode_id"]) != str(spec["episode_id"]):
                raise ValueError("LHPR episode ID and original row index differ")
            self.runtime.reset(spec["episode_index"])
        except BaseException:
            self.runtime.close()
            raise
        self.provenance = {
            "episode_id": spec["episode_id"],
            "episode_index": spec["episode_index"],
            "effective_seed": spec["seed"] + spec["episode_index"],
            "split_sha256": metadata["split_sha256"],
            "protocol": metadata["protocol"],
        }

    def observe(self) -> dict:
        """Return only public progress, instruction and processed RGB views."""
        from PIL import Image

        images = {}
        for name, array in self.runtime.observe().items():
            if array is not None:
                output = io.BytesIO()
                Image.fromarray(array).convert("RGB").save(output, format="PNG")
                role = {"rgb": "front", "rgb_left": "left", "rgb_right": "right"}[name]
                images[role] = base64.b64encode(output.getvalue()).decode("ascii")
        sim = self.runtime.sim
        return {
            "instruction": sim.instruction,
            "step_index": sim.step,
            "terminated": bool(sim.done),
            "truncated": bool(sim.episode_over and not sim.done),
            "stage_index": sim.stage,
            "stage_count": sim.target_num,
            "actions": list(self.actions),
            "turn_degrees": 30,
            "forward_metres": 0.25,
            "images": images,
        }

    def step(self, action: str) -> dict:
        """Let native recovery count its own physical actions against the budget."""
        return self.runtime.step(self.actions.index(action))

    def evaluate(self) -> dict:
        """Never finalize an unfinished episode into an apparent scored failure."""
        sim = self.runtime.sim
        terminated, truncated = bool(sim.done), bool(sim.episode_over and not sim.done)
        if not (terminated or truncated):
            raise ValueError("unfinished LHPR episode cannot be scored")
        native = self.runtime.evaluate()
        aggregate = importlib.import_module("lhpr_native._metrics").aggregate_lhvln
        return {
            "success": bool(native["SR"]),
            "steps": sim.step,
            "terminated": terminated,
            "truncated": truncated,
            "metrics": aggregate([{"metrics": native}]),
            "official_results": native,
            "provenance": self.provenance,
        }

    def close(self) -> None:
        """Release the native simulator."""
        self.runtime.close()
