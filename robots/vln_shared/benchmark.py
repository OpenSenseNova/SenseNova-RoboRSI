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

"""Navigation Providers and Adapters using RoboRSI's public HTTP transport."""

from __future__ import annotations

import base64
import platform
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

from jsonschema import Draft202012Validator

from roborsi.benchmarks.registry import directory_sha256
from roborsi.benchmarks.runtime_source import benchmark_source_digest
from roborsi.evaluation.benchmark_http import BenchmarkApplication, BenchmarkHttpServer
from roborsi.evaluation.http_artifacts import ArtifactPublisher
from roborsi.evaluation.media import CaseMediaChannel
from roborsi.evaluation.messages import (
    Action,
    BenchmarkRuntimeInfo,
    NativeResult,
    Observation,
    StartResult,
    StepResult,
    Task,
)
from robots.vln_shared.native_process import NativeProcess
from robots.vln_shared.storage import digest, file_digest, read_json, write_json

EVALUATION = {
    "success_metric": "success",
    "score_metric": "success",
    "aggregation": "case_mean",
    "error_denominator": "valid",
}


def action_schema(dataset: str) -> dict:
    """Describe bounded action batches, including RxR camera tilt."""
    actions = ["stop", "move_forward", "turn_left", "turn_right"]
    if dataset == "rxr":
        actions += ["look_up", "look_down"]
    return {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "items": {"enum": actions},
                "minItems": 1,
                "maxItems": 32,
            }
        },
        "required": ["actions"],
        "additionalProperties": False,
    }


def make_task(episode: dict, spec: dict, budget: int) -> Task:
    """Expose public instructions and stable original indices, without goal labels."""
    benchmark = "lhpr_vln" if spec["dataset"] == "lhpr_vln" else "vlnce"
    return Task(
        benchmark=benchmark,
        task_id=episode["id"],
        seed=spec["seed"],
        instruction=episode["instruction"],
        max_steps=budget,
        native={
            "dataset": spec["dataset"],
            "episode_id": episode["episode_id"],
            "episode_index": episode["episode_index"],
            "split": spec["split"],
        },
        capabilities={
            "action_schema": action_schema(spec["dataset"]),
            "observation_channels": ["front", "left", "right"]
            if benchmark == "lhpr_vln"
            else ["front"],
        },
    )


class NavigationAdapter:
    """Map one RPC batch to native actions, stopping at boundaries and budget."""

    def __init__(
        self,
        directory: Path,
        tasks: list[Task],
        spec: dict,
        process_factory=NativeProcess,
        *,
        on_progress: Callable[[int], None] | None = None,
    ) -> None:
        self.directory, self.tasks, self.spec = directory, tasks, spec
        self.channel = CaseMediaChannel(directory / "media")
        self.process_factory = process_factory
        self.on_progress = on_progress
        self.worker = None
        self.ended = False
        self.step_index = 0

    def start(self, task: Task, episode_id: str) -> StartResult:
        """Start a new native process using the annotation's original index and seed."""
        if task not in self.tasks or self.worker is not None:
            raise ValueError("task is not in catalog or adapter already started")
        self.task, self.episode_id = task, episode_id
        self.worker = self.process_factory(
            self.spec["python"], self.directory, timeout=self.spec.get("timeout_s", 600)
        )
        try:
            raw = self.worker.call(
                "start",
                spec={
                    **self.spec,
                    **task.native,
                    "max_steps": task.max_steps,
                    "seed": task.seed,
                },
            )
            self.raw = raw
            return StartResult(
                task=task,
                observation=self._observation(raw),
                capabilities=task.capabilities,
            )
        except BaseException:
            self.close()
            raise

    def _observation(self, raw: dict) -> Observation:
        images = raw.get("images", {})
        media = tuple(
            self.channel.write(
                self.episode_id,
                self.step_index,
                base64.b64decode(value, validate=True),
                "image/png",
            )
            for value in images.values()
        )
        data = {k: v for k, v in raw.items() if k not in {"images", "step_index"}}
        data.update({"physical_steps": raw["step_index"], "view_order": list(images)})
        if self.on_progress is not None:
            self.on_progress(raw["step_index"])
        return Observation(
            benchmark=self.task.benchmark,
            episode_id=self.episode_id,
            step_index=self.step_index,
            data=data,
            media=media,
        )

    def step(self, action: Action) -> StepResult:
        """Validate the complete batch before executing; never cross an LHPR subtask."""
        if self.ended or (action.benchmark, action.episode_id, action.step_index) != (
            self.task.benchmark,
            self.episode_id,
            self.step_index,
        ):
            raise ValueError("action identity, order or terminal state mismatch")
        Draft202012Validator(self.task.capabilities["action_schema"]).validate(
            action.data
        )
        executed = []
        for name in action.data["actions"]:
            raw = self.worker.call("step", action=name)
            if (
                raw["step_index"] <= self.raw["step_index"]
                or raw["step_index"] > self.task.max_steps
            ):
                raise ValueError(
                    "native action accounting violated the physical budget"
                )
            self.raw = raw
            executed.append(
                {
                    "action": name,
                    "physical_steps": raw["step_index"],
                    "feedback": raw.get("feedback", {}),
                }
            )
            self.ended = raw["terminated"] or raw["truncated"]
            if self.ended or raw.get("feedback", {}).get("subtask_boundary"):
                break
        self.step_index += 1
        return StepResult(
            observation=self._observation({**self.raw, "executed_batch": executed}),
            terminated=self.raw["terminated"],
            truncated=self.raw["truncated"],
            termination_reason="native_terminal" if self.ended else None,
            native={"executed": executed, "physical_steps": self.raw["step_index"]},
        )

    def finish(self) -> NativeResult:
        """Persist the whole original score record before releasing the simulator."""
        if not self.ended:
            raise ValueError("native scoring requires a terminal episode")
        result = self.worker.call("finish")
        if not (result.get("terminated") or result.get("truncated")):
            raise ValueError("missing native terminal evidence")
        path = self.directory / "native/result.json"
        write_json(path, result)
        return NativeResult(
            task=self.task,
            episode_id=self.episode_id,
            success=result["success"],
            score=float(result["success"]),
            details=result,
            source="native/result.json",
            source_sha256=file_digest(path),
        )

    def close(self) -> None:
        """Release an owned worker, including partially started episodes."""
        if self.worker is not None:
            self.worker.close()


class NavigationBenchmark:
    """Serve a pinned catalog; native inputs stay private to the environment."""

    def __init__(
        self,
        spec: dict,
        episodes: list[dict],
        budget: int,
        *,
        asset_root: Path | None = None,
        on_progress: Callable[[int], None] | None = None,
    ) -> None:
        self.spec = spec
        self.asset_root = asset_root
        self.on_progress = on_progress
        self.catalog = [make_task(e, spec, budget) for e in episodes]
        if not self.catalog or len({t.task_id for t in self.catalog}) != len(
            self.catalog
        ):
            raise ValueError("catalog requires unique episode IDs")
        self.name = self.catalog[0].benchmark

    def tasks(self) -> list[Task]:
        """Return stable public task descriptions."""
        return self.catalog[:]

    def runtime_info(self) -> BenchmarkRuntimeInfo:
        """Verify pinned native source and annotation files before and after evaluation."""
        for name, expected in self.spec["inputs"].items():
            if file_digest(Path(name)) != expected:
                raise ValueError(f"native input changed: {name}")
        return BenchmarkRuntimeInfo(
            benchmark=self.name,
            benchmark_version=self.spec["native_commit"][:12],
            benchmark_commit=self.spec["native_commit"],
            adapter_version=benchmark_source_digest(
                Path(__file__).resolve().parents[2], self.name
            ),
            evaluation=EVALUATION,
            platform={
                "system": platform.system(),
                "native_python": self.spec["python"],
            },
        )

    def assets(self) -> list[dict]:
        """Report identities checked by the runtime."""
        if self.asset_root is not None:
            return [
                {
                    "name": self.name,
                    "kind": "directory",
                    "sha256": directory_sha256(self.asset_root),
                }
            ]
        return [
            {
                "name": self.name + "-inputs",
                "kind": "directory",
                "sha256": digest(self.spec["inputs"]),
            }
        ]

    def create_adapter(
        self, directory: Path, artifacts: ArtifactPublisher
    ) -> NavigationAdapter:
        """Let the common application publish media and native score artifacts."""
        return NavigationAdapter(
            directory, self.tasks(), self.spec, on_progress=self.on_progress
        )


def portable_provider(catalog: Path, budget: int) -> NavigationBenchmark:
    """Resolve a sealed catalog's relative paths under its mounted asset directory."""
    section = read_json(catalog)
    root = catalog.absolute().parent
    spec = dict(section["spec"])

    def resolve(name: str) -> str:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("portable catalog inputs must be asset-relative")
        candidate = root / path
        if any(p.is_symlink() for p in (candidate, *candidate.parents)):
            raise ValueError("portable catalog must not contain symlink components")
        return str(candidate)

    for key in (
        "repo_root",
        "runtime_root",
        "data_root",
        "scenes_root",
        "scene_dataset",
        "regions_root",
    ):
        if key in spec:
            spec[key] = resolve(spec[key])
    spec["inputs"] = {resolve(k): v for k, v in spec["inputs"].items()}
    return NavigationBenchmark(spec, section["episodes"], budget, asset_root=root)


@contextmanager
def serve(provider: NavigationBenchmark, directory: Path) -> Iterator[str]:
    """Own one loopback service, its server thread, and all native episodes."""
    application = BenchmarkApplication(provider, directory)
    server = BenchmarkHttpServer(("127.0.0.1", 0), application)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        application.close()
        thread.join(timeout=5)
