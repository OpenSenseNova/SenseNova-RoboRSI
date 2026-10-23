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

"""Explicit server source boundary shared by packaging and Adapter versions."""

from __future__ import annotations

import hashlib
from importlib.metadata import distributions
from pathlib import Path

_COMMON = (
    "roborsi/__init__.py",
    "roborsi/benchmarks/__init__.py",
    "roborsi/benchmarks/registry.py",
    "roborsi/benchmarks/runtime_source.py",
    "roborsi/evaluation/__init__.py",
    "roborsi/evaluation/adapter_service.py",
    "roborsi/evaluation/benchmark_http.py",
    "roborsi/evaluation/http_artifacts.py",
    "roborsi/evaluation/media.py",
    "roborsi/evaluation/messages.py",
    "roborsi/evaluation/protocol.py",
    "roborsi/evaluation/result.py",
    "roborsi/evaluation/service_host.py",
    "roborsi/evaluation/service_protocol.py",
    "roborsi/schemas/__init__.py",
    "roborsi/schemas/roborsi_benchmark_manifest.v1.json",
    "roborsi/schemas/roborsi_interaction_envelope.v1.json",
    "roborsi/utils/__init__.py",
    "roborsi/utils/config.py",
    "roborsi/utils/daemon.py",
    "roborsi/utils/logging.py",
    "roborsi/utils/rwlock.py",
    "roborsi/utils/serialization.py",
    "roborsi/utils/templates.py",
    "roborsi/utils/rpc/__init__.py",
    "roborsi/utils/rpc/client_utils.py",
    "roborsi/utils/rpc/http_abort.py",
    "roborsi/utils/rpc/http_rpc.py",
    "roborsi/utils/rpc/main_thread_serve.py",
    "roborsi/utils/rpc/rpc_client.py",
    "roborsi/utils/rpc/rpc_facade.py",
    "roborsi/utils/rpc/socket_rpc.py",
    "robots/__init__.py",
)
_ENV_FACADE = (
    "roborsi/robots/__init__.py",
    "roborsi/robots/components/__init__.py",
    "roborsi/robots/components/env_facade_base.py",
)
_BENCHMARK = {
    "robocasa365": (
        *_ENV_FACADE,
        "roborsi/dashboard/__init__.py",
        "roborsi/dashboard/events.py",
        "roborsi/robots/runtime.py",
        "roborsi/robots/components/env_client_base.py",
        "roborsi/robots/components/vla_client_base.py",
        "roborsi/robots/components/vla_facade_base.py",
        "roborsi/session/__init__.py",
        "roborsi/session/base.py",
        "roborsi/tools/__init__.py",
        "roborsi/tools/common.py",
        "roborsi/tools/toolkit.py",
        "roborsi/utils/egl.py",
        "robots/robocasa/__init__.py",
        "robots/robocasa/benchmark_service.py",
        "robots/robocasa/roborsi_adapter.py",
        "robots/robocasa/runtime.py",
        "robots/robocasa/toolkit.py",
        "robots/robocasa/tools.py",
        "robots/robocasa/primitives.py",
        "robots/robocasa/rldx_skill.py",
        "robots/robocasa/env_client.py",
        "robots/robocasa/env_server.py",
        "robots/robocasa/vla_client.py",
        "robots/robocasa/vla_server.py",
        "robots/robocasa/eval/__init__.py",
        "robots/robocasa/eval/target50.json",
        "benchmarks/robocasa365/manifest.py",
    ),
    "behavior_1k": (
        "robots/behavior_1k/benchmark_service.py",
        "robots/behavior_1k/catalog.py",
        "robots/behavior_1k/contracts.py",
        "robots/behavior_1k/episode.py",
        "robots/behavior_1k/native.py",
        "robots/behavior_1k/official.py",
        "robots/behavior_1k/provenance.py",
        "robots/behavior_1k/roborsi_adapter.py",
        "robots/behavior_1k/worker.py",
        "benchmarks/behavior_1k/__init__.py",
        "benchmarks/behavior_1k/prepare.py",
        "benchmarks/behavior_1k/official_tasks.json",
    ),
    "bench2dex": (
        "LICENSE",
        "robots/bench2dex/benchmark_service.py",
        "robots/bench2dex/roborsi_adapter.py",
        "robots/bench2dex/native.py",
        "robots/bench2dex/native_entry.py",
        "robots/bench2dex/contracts.py",
        "robots/bench2dex/protocol.json",
        "robots/bench2dex/tasks.json",
        "robots/bench2dex/reference/__init__.py",
        "robots/bench2dex/reference/core.py",
        "robots/bench2dex/reference/robot.py",
        "robots/bench2dex/reference/calibration.py",
        "robots/bench2dex/reference/kinematics_tool.py",
        "robots/bench2dex/reference/trajectory.py",
        "robots/bench2dex/reference/rendering.py",
        "robots/bench2dex/reference/transport.py",
        "robots/bench2dex/reference/throughput.py",
        "robots/bench2dex/reference/simulator.py",
        "robots/bench2dex/reference/SOURCES.json",
        "robots/bench2dex/reference/LICENSE_NOTICE.md",
    ),
    "dexjoco": (
        "robots/dexjoco/contracts.py",
        "robots/dexjoco/horizon.py",
        "robots/dexjoco/native.py",
        "robots/dexjoco/roborsi_adapter.py",
        "robots/dexjoco/benchmark_service.py",
        "benchmarks/dexjoco/manifest.py",
        "benchmarks/dexjoco/LICENSE.reference",
    ),
    "libero": (
        *_ENV_FACADE,
        "roborsi/utils/egl.py",
        "robots/libero/__init__.py",
        "robots/libero/benchmark_service.py",
        "robots/libero/roborsi_adapter.py",
        "robots/libero/env_server.py",
        "benchmarks/libero/manifest.py",
        "benchmarks/libero/download_assets.py",
    ),
    "libero_pro": (
        *_ENV_FACADE,
        "roborsi/utils/egl.py",
        "robots/libero/__init__.py",
        "robots/libero/benchmark_catalog.py",
        "robots/libero/three_benchmark_adapter.py",
        "robots/libero/three_benchmark_env.py",
        "robots/libero/three_benchmark_service.py",
        "robots/libero/three_benchmark_serve.py",
        "robots/libero/env_server.py",
        "robots/libero/lilo_task.py",
        "robots/libero/lilo_assistance.py",
        "benchmarks/libero_pro/manifest.py",
    ),
    "libero_lilo": (
        "robots/libero/__init__.py",
        "robots/libero/benchmark_catalog.py",
        "robots/libero/three_benchmark_adapter.py",
        "robots/libero/three_benchmark_env.py",
        "robots/libero/three_benchmark_service.py",
        "robots/libero/three_benchmark_serve.py",
        "robots/libero/lilo_task.py",
        "robots/libero/lilo_assistance.py",
        "benchmarks/libero_lilo/manifest.py",
    ),
}


# Optional PRO service bundle, explicitly selected by environment.native_bundle.
# Keep the default PRO file list identical to the released catalog.
_BENCHMARK["libero_pro_image_tools"] = (
    *_BENCHMARK["libero_pro"],
    "robots/libero/pro_protocol.py",
    "robots/libero/image_tools.py",
    "robots/libero/image_tools.lock.json",
    "robots/libero/image_tool_records.py",
    "robots/libero/image_tool_adapter.py",
    "robots/libero/image_vla_bridge.py",
)


def benchmark_source_files(benchmark: str) -> tuple[str, ...]:
    """List only protocol, HTTP, artifact and selected native wrapper sources."""
    if benchmark not in _BENCHMARK:
        raise ValueError(f"unsupported Benchmark source bundle: {benchmark}")
    return tuple(sorted((*_COMMON, *_BENCHMARK[benchmark])))


def benchmark_source_digest(root: Path, benchmark: str) -> str:
    """Hash the same named files in a checkout and in its standalone image."""
    digest = hashlib.sha256(b"RoboRSI-benchmark-source-v1\0")
    for name in benchmark_source_files(benchmark):
        path = root / name
        if (
            any(parent.is_symlink() for parent in (path, *path.parents))
            or not path.is_file()
        ):
            raise ValueError(f"Benchmark source is missing or linked: {name}")
        digest.update(
            name.encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest()
        )
    return digest.hexdigest()


def verify_benchmark_runtime(root: Path, benchmark: str) -> None:
    """Reject Agent packages inherited from a runtime base or stale source tree."""
    benchmark_source_digest(root, benchmark)
    allowed = set(benchmark_source_files(benchmark))
    for directory in ("roborsi", "heart_runtime", "rpent", "robots", "benchmarks"):
        for path in (root / directory).rglob("*"):
            if path.is_file() and path.relative_to(root).as_posix() not in allowed:
                raise ValueError(f"undeclared Benchmark image source: {path}")
    forbidden = {
        "roborsi",
        "heart-runtime",
        "heart",
        "rpent",  # Reject the previously published Agent distribution as well.
        "pydantic-ai",
        "pydantic-ai-slim",
        "openai",
        "anthropic",
        "claude-agent-sdk",
        "openai-codex",
        "openai-codex-cli-bin",
        "mcp",
    }
    installed = {
        package.metadata["Name"].lower().replace("_", "-")
        for package in distributions()
    }
    if overlap := forbidden & installed:
        raise ValueError(
            "Benchmark base must exclude Agent packages: " + ", ".join(sorted(overlap))
        )
