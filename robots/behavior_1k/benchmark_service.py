# Copyright 2026 The RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""BEHAVIOR Provider for the common Benchmark HTTP service."""

from __future__ import annotations

import os
import platform
from pathlib import Path

from roborsi.benchmarks.registry import directory_sha256
from roborsi.benchmarks.runtime_source import benchmark_source_digest
from roborsi.evaluation.http_artifacts import ArtifactPublisher
from roborsi.evaluation.messages import BenchmarkRuntimeInfo, Task
from roborsi.utils.config import get_repo_root
from robots.behavior_1k.catalog import EVALUATION_RULES, NATIVE_COMMIT, read_catalog
from robots.behavior_1k.provenance import verify_native_source
from robots.behavior_1k.roborsi_adapter import BehaviorRoboRSIAdapter


class BehaviorBenchmark:
    """Discover a native-generated catalog; isolate each Isaac episode."""

    name = "behavior_1k"
    supports_episode_stop = True

    def __init__(self) -> None:
        self.catalog = Path(
            os.environ.get(
                "ROBORSI_BEHAVIOR_CATALOG", "/roborsi/assets/behavior_1k-catalog"
            )
        )
        self.data = Path(
            os.environ.get("ROBORSI_BEHAVIOR_ASSETS", "/roborsi/assets/behavior_1k")
        )

    def tasks(self) -> list[Task]:
        return read_catalog(self.catalog / "catalog.json")

    def assets(self) -> list[dict]:
        return [
            {"name": name, "kind": "directory", "sha256": directory_sha256(path)}
            for name, path in (
                ("behavior_1k", self.data),
                ("behavior_1k-catalog", self.catalog),
            )
        ]

    def runtime_info(self) -> BenchmarkRuntimeInfo:
        verify_native_source(
            Path(os.environ.get("ROBORSI_BEHAVIOR_NATIVE", "/workspace/BEHAVIOR-1K"))
        )
        # Registration belongs to the framework owner; do not bypass its inventory.
        return BenchmarkRuntimeInfo(
            benchmark=self.name,
            benchmark_version="3.9.2",
            benchmark_commit=NATIVE_COMMIT,
            adapter_version=benchmark_source_digest(get_repo_root(), self.name),
            evaluation=EVALUATION_RULES,
            platform={
                "python": platform.python_version(),
                "system": platform.platform(),
            },
        )

    def create_adapter(
        self, directory: Path, artifacts: ArtifactPublisher
    ) -> BehaviorRoboRSIAdapter:
        return BehaviorRoboRSIAdapter(directory)


def make_behavior_benchmark() -> BehaviorBenchmark:
    return BehaviorBenchmark()
