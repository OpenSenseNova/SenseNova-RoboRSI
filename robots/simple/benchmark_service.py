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

"""Publish only a pinned private catalog and the selected SIMPLE runtime."""

from __future__ import annotations

import hashlib
import os
import platform
from pathlib import Path

from roborsi.benchmarks.runtime_source import benchmark_source_digest
from roborsi.evaluation.http_artifacts import ArtifactPublisher
from roborsi.evaluation.messages import BenchmarkRuntimeInfo, Task
from robots.simple.catalog import Catalog
from robots.simple.contracts import EVALUATION_RULES, SIMPLE_REVISION
from robots.simple.roborsi_adapter import SimpleAdapter


class SimpleBenchmark:
    """Serve immutable task identities with a fresh native process per case."""

    name = "simple"

    def __init__(
        self, catalog: Path, *, factory: str = "robots.simple.native:make_runtime"
    ) -> None:
        self.catalog = Catalog(catalog)
        self.factory = factory

    def tasks(self) -> list[Task]:
        return self.catalog.tasks()

    def runtime_info(self) -> BenchmarkRuntimeInfo:
        root = Path(__file__).resolve().parents[2]
        # Registration is owned by the framework maintainer; do not bypass it.
        source_digest = benchmark_source_digest(root, "simple")
        return BenchmarkRuntimeInfo(
            benchmark="simple",
            benchmark_version="SIMPLE-" + SIMPLE_REVISION[:12],
            benchmark_commit=SIMPLE_REVISION,
            adapter_version="roborsi-source-sha256:" + source_digest,
            evaluation={
                key: value
                for key, value in EVALUATION_RULES.items()
                if key != "score_protocol"
            },
            platform={
                "system": platform.system(),
                "kernel": platform.release(),
                "native_verification": "pending; see benchmarks/simple/VALIDATION.md",
            },
        )

    def assets(self) -> list[dict]:
        return [
            {
                "name": "simple-catalog",
                "kind": "file",
                "sha256": hashlib.sha256(self.catalog.path.read_bytes()).hexdigest(),
            }
        ]

    def create_adapter(
        self, directory: Path, artifacts: ArtifactPublisher
    ) -> SimpleAdapter:
        return SimpleAdapter(
            directory, artifacts, factory=self.factory, catalog=self.catalog
        )


def make_simple_benchmark() -> SimpleBenchmark:
    """Load the explicitly selected private catalog; never synthesize test cases."""
    return SimpleBenchmark(
        Path(
            os.environ.get(
                "ROBORSI_SIMPLE_CATALOG",
                "/roborsi/assets/simple-catalog/catalog.json",
            )
        )
    )
