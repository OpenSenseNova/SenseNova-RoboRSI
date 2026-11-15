# Copyright 2026 The RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LIBERO discovery and private Adapter instances for the Benchmark HTTP API."""

from __future__ import annotations

import json
from pathlib import Path

from roborsi.evaluation.http_artifacts import ArtifactPublisher
from roborsi.evaluation.messages import BenchmarkRuntimeInfo, Task
from roborsi.utils.config import get_project_env
from robots.libero.roborsi_adapter import (
    ACTION_SCHEMA,
    LiberoRoboRSIAdapter,
    runtime_info,
)


class LiberoBenchmark:
    """Keep simulator files inside the service and expose tasks over HTTP."""

    name = "libero"

    def runtime_info(self) -> BenchmarkRuntimeInfo | None:
        return runtime_info()

    def tasks(self) -> list[Task]:
        from libero.libero import benchmark

        suite_name = get_project_env("ROBORSI_LIBERO_SUITE", "libero_spatial")
        suite = benchmark.get_benchmark(suite_name)()
        seeds = json.loads(get_project_env("ROBORSI_LIBERO_SEEDS", "[0]"))
        steps = int(get_project_env("ROBORSI_LIBERO_MAX_STEPS", "600"))
        return [
            Task(
                benchmark="libero",
                task_id=str(index),
                seed=seed,
                instruction=suite.get_task(index).language,
                max_steps=steps,
                native={"suite": suite_name},
                capabilities={
                    "action_schema": ACTION_SCHEMA,
                    "observation_channels": ["main_images", "wrist_images", "states"],
                },
            )
            for index in range(suite.n_tasks)
            for seed in seeds
        ]

    def assets(self) -> list[dict]:
        from roborsi.benchmarks.registry import directory_sha256

        root = Path("/roborsi/assets/libero")
        return [
            {"name": "libero", "kind": "directory", "sha256": directory_sha256(root)}
        ]

    def create_adapter(
        self, directory: Path, artifacts: ArtifactPublisher
    ) -> LiberoRoboRSIAdapter:
        return LiberoRoboRSIAdapter(directory / "media", directory / "native")


def make_libero_benchmark() -> LiberoBenchmark:
    """Create the provider used by the image's HTTP service entry point."""
    return LiberoBenchmark()
