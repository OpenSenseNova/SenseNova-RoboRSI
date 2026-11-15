# Copyright 2026 The RoboRSI Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Discover the historical 55-case cohort through RoboRSI's shared HTTP API."""

from __future__ import annotations

import hashlib
import os
import platform
from pathlib import Path

from roborsi.evaluation.messages import BenchmarkRuntimeInfo, Task
from robots.dexjoco.contracts import (
    CAPABILITIES,
    COMMIT,
    EVALUATION,
    LIMITS,
    PROTOCOL,
    SEEDS,
)
from robots.dexjoco.native import task_config, verify_upstream


def runtime_info() -> BenchmarkRuntimeInfo:
    """Identify the adapter files independently of deployment claims."""
    digest = hashlib.sha256()
    for name in (
        "contracts.py",
        "native.py",
        "horizon.py",
        "roborsi_adapter.py",
        "benchmark_service.py",
    ):
        digest.update(
            name.encode() + b"\0" + Path(__file__).with_name(name).read_bytes()
        )
    return BenchmarkRuntimeInfo(
        benchmark="dexjoco",
        benchmark_version=PROTOCOL,
        benchmark_commit=COMMIT,
        adapter_version=f"dexjoco-adapter-sha256:{digest.hexdigest()}",
        evaluation=dict(EVALUATION),
        platform={
            "system": platform.system(),
            "machine": platform.machine(),
            "rendering": "osmesa",
        },
    )


def make_task(root: Path, name: str, seed: int, multiplier: int = 10) -> Task:
    """Keep task IDs, seeds, instructions and physical step budgets native."""
    if multiplier not in (1, 10) or seed not in SEEDS:
        raise ValueError("Historical cohort requires seeds 100000..100004 and 1x/10x")
    cfg = task_config(root, name)
    return Task(
        benchmark="dexjoco",
        task_id=name,
        seed=seed,
        instruction=cfg["prompt"],
        max_steps=LIMITS[name] * multiplier,
        native={"regime": "rand_full", "protocol": PROTOCOL, "multiplier": multiplier},
        capabilities=CAPABILITIES,
    )


class DexJocoBenchmark:
    """Provide fresh process-owned simulators and case-scoped artifacts."""

    name = "dexjoco"
    supports_episode_stop = True

    def __init__(self, upstream: Path, multiplier: int = 10) -> None:
        self.upstream = upstream.resolve()
        verify_upstream(self.upstream)
        if multiplier not in (1, 10):
            raise ValueError("Supported horizons are 1 and 10")
        self.multiplier = multiplier

    def runtime_info(self) -> BenchmarkRuntimeInfo:
        """Return source identity, native scoring rules and rendering platform."""
        return runtime_info()

    def tasks(self) -> list[Task]:
        """Return eleven tasks with five prespecified seeds each."""
        return [
            make_task(self.upstream, name, seed, self.multiplier)
            for name in LIMITS
            for seed in SEEDS
        ]

    def assets(self) -> list[dict]:
        """Simulator assets are contained in its verified pinned Git checkout."""
        verify_upstream(self.upstream)
        return []

    def create_adapter(self, directory: Path, artifacts):
        """Use the framework's existing artifact and HTTP ownership."""
        from robots.dexjoco.roborsi_adapter import DexJocoAdapter

        return DexJocoAdapter(
            directory, self.upstream, multiplier=self.multiplier, artifacts=artifacts
        )


def make_dexjoco_benchmark() -> DexJocoBenchmark:
    """Factory for roborsi.evaluation.service_host --kind benchmark."""
    return DexJocoBenchmark(
        Path(os.environ["ROBORSI_DEXJOCO_UPSTREAM"]),
        int(os.environ.get("ROBORSI_DEXJOCO_MULTIPLIER", "10")),
    )
