# Copyright 2026 The SenseNova-RoboRSI Authors.
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

"""Serve the RoboCasa365 Target50 cells through RoboRSI's shared HTTP API."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import sys
import threading
from pathlib import Path
from typing import Any

from roborsi.evaluation.messages import BenchmarkRuntimeInfo, Task
from robots.robocasa import tools as robocasa_tools

BENCHMARK = "robocasa365"
TARGET50 = json.loads(
    (Path(__file__).parent / "eval" / "target50.json").read_text(encoding="utf-8")
)
PROTOCOL = TARGET50["protocol_id"]
COMMIT = TARGET50["dependencies"]["robocasa"]["commit"]
# One action runs one RoboCasa tool. The Target50 protocol bounds an episode by
# 100 planner turns and the split's wall clock; this cap only stops a runaway
# episode and is never the binding limit in practice.
MAX_STEPS = 400
# Exploration-only tools never enter an evaluation episode.
_EXCLUDED_TOOLS = frozenset({"reset"})
_ADAPTER_FILES = ("benchmark_service.py", "roborsi_adapter.py", "tools.py")

# Target50 reports the mean of the 50 per-task success rates, so a cell
# without a valid result still counts against its task.
EVALUATION = {
    "success_metric": "success",
    "score_metric": "score",
    "aggregation": "task_weighted_success_rate",
    "error_denominator": "expected",
}


def _action_variant(spec: dict[str, Any]) -> dict[str, Any]:
    """Turn one RoboCasa tool schema into an action carrying its name.

    Only the tool name is enforced. A missing or unknown argument reaches the
    tool, which answers with its usual "bad arguments" result and lets the
    episode go on, as the Target50 runs did; rejecting the action at the
    schema instead would end the episode as an Agent error.
    """
    schema = spec["input_schema"]
    properties = {"primitive": {"const": spec["name"]}}
    properties.update(copy.deepcopy(schema.get("properties", {})))
    required = schema.get("required", [])
    description = f"{spec['name']}: {spec['description']}"
    if required:
        description += f" Required arguments: {', '.join(required)}."
    return {
        "type": "object",
        "description": description,
        "properties": properties,
        "required": ["primitive"],
    }


ACTION_SCHEMA = {
    "oneOf": [
        _action_variant(spec)
        for spec in robocasa_tools.TOOLS_SPEC
        if spec["name"] not in _EXCLUDED_TOOLS
    ]
}
PRIMITIVES = tuple(
    variant["properties"]["primitive"]["const"] for variant in ACTION_SCHEMA["oneOf"]
)
CAPABILITIES = {
    "action_schema": ACTION_SCHEMA,
    "observation_channels": ["agentview", "navview", "wrist", "state"],
}
# The primitives read the RLDX rollout limits from the environment; Target50
# fixes them, and an unset RLDX_MAX_CHUNKS would fall back to 70.
RLDX_ENVIRONMENT = {
    "RLDX_MAX_CHUNKS": str(TARGET50["runtime_protocol"]["rldx_max_chunks"]),
    "RLDX_SETTLE_PATIENCE": str(TARGET50["runtime_protocol"]["rldx_settle_patience"]),
    "RLDX_ACTION_STEPS_PER_CHUNK": str(
        TARGET50["runtime_protocol"]["rldx_action_steps_per_chunk"]
    ),
}


def split_of(task_name: str) -> tuple[str, dict[str, Any]]:
    """Return the Target50 split owning ``task_name``."""
    for name, split in TARGET50["splits"].items():
        if task_name in split["tasks"]:
            return name, split
    raise ValueError(f"{task_name} is not a Target50 task")


def runtime_info() -> BenchmarkRuntimeInfo:
    """Identify the adapter files independently of deployment claims."""
    digest = hashlib.sha256()
    for name in _ADAPTER_FILES:
        digest.update(
            name.encode() + b"\0" + Path(__file__).with_name(name).read_bytes()
        )
    return BenchmarkRuntimeInfo(
        benchmark=BENCHMARK,
        benchmark_version=PROTOCOL,
        benchmark_commit=COMMIT,
        adapter_version=f"robocasa365-adapter-sha256:{digest.hexdigest()}",
        evaluation=dict(EVALUATION),
        platform={
            "system": platform.system(),
            "machine": platform.machine(),
            "rendering": os.environ.get("MUJOCO_GL", "egl"),
        },
    )


def make_task(task_name: str, seed: int) -> Task:
    """Return one Target50 cell with its native split, seed and wall clock."""
    split, spec = split_of(task_name)
    if seed not in spec["seeds"]:
        raise ValueError(f"seed {seed} is outside the Target50 seeds of {split}")
    return Task(
        benchmark=BENCHMARK,
        task_id=task_name,
        seed=seed,
        instruction=(
            f"Solve the RoboCasa365 task {task_name} in the target kitchen "
            f"({split}, seed {seed}). The live task language, success "
            "criteria and camera views are in each observation."
        ),
        max_steps=MAX_STEPS,
        native={
            "protocol": PROTOCOL,
            "environment_split": TARGET50["environment_split"],
            "split": split,
            "timeout_s": spec["timeout_seconds"],
        },
        capabilities=CAPABILITIES,
    )


class RoboCasa365Benchmark:
    """Provide one RoboCasa kitchen per episode and a shared RLDX-1 service.

    The RLDX-1 service is either an existing endpoint or, when only a
    checkpoint is given, a ``vla_server`` this service starts on its first
    episode and keeps for later ones; the server exits with this process.
    """

    name = BENCHMARK
    supports_episode_stop = True

    def __init__(
        self,
        vla_endpoint: str | None,
        *,
        cuda_device: int | None = None,
        cells: str = "all",
        vla_checkpoint: str | None = None,
        spawn_vla=None,
    ) -> None:
        if cells not in ("all", "seed1"):
            raise ValueError("cells must be 'all' or 'seed1'")
        self.vla_endpoint = vla_endpoint
        self.cuda_device = cuda_device
        self.cells = cells
        self.vla_checkpoint = vla_checkpoint
        self._spawn_vla = spawn_vla or _spawn_vla_server
        self._vla_lock = threading.Lock()
        self._vla_daemon = None

    def _vla(self) -> str | None:
        """Return the RLDX-1 endpoint, starting the owned server once if needed."""
        with self._vla_lock:
            if self.vla_endpoint is None and self.vla_checkpoint:
                self._vla_daemon, self.vla_endpoint = self._spawn_vla(
                    self.vla_checkpoint, self.cuda_device
                )
            return self.vla_endpoint

    def runtime_info(self) -> BenchmarkRuntimeInfo:
        """Return source identity, native scoring rules and rendering platform."""
        return runtime_info()

    def tasks(self) -> list[Task]:
        """Return the 340 Target50 cells, or the 50 seed-1 cells."""
        return [
            make_task(name, seed)
            for split in TARGET50["splits"].values()
            for name in split["tasks"]
            for seed in split["seeds"]
            if self.cells == "all" or seed == 1
        ]

    def assets(self) -> list[dict]:
        """Kitchen assets ship inside the image; RLDX-1 runs as its own service."""
        return []

    def create_adapter(self, directory: Path, artifacts):
        """Use the framework's existing artifact and HTTP ownership."""
        from robots.robocasa.roborsi_adapter import RoboCasa365Adapter

        return RoboCasa365Adapter(
            directory,
            vla_endpoint=self._vla(),
            cuda_device=self.cuda_device,
            artifacts=artifacts,
        )


def _spawn_vla_server(checkpoint: str, cuda_device: int | None):
    """Start one RLDX-1 server owned by this process and wait until it serves."""
    from roborsi.utils.daemon import ProcessDaemon, pick_free_port
    from roborsi.utils.rpc.client_utils import wait_for_ready
    from roborsi.utils.rpc.http_rpc import HttpRpcClient

    port = pick_free_port()
    daemon = ProcessDaemon(
        name="vla_server",
        cmd=[
            sys.executable,
            str(Path(__file__).with_name("vla_server.py")),
            "--model-path",
            checkpoint,
            "--transport",
            "http",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--parent-watch",
            *(["--cuda-device", str(cuda_device)] if cuda_device is not None else []),
        ],
        prepend_rlinf_path=False,
    )
    daemon.start()
    endpoint = f"http://127.0.0.1:{port}"
    probe = HttpRpcClient(endpoint)
    try:
        wait_for_ready(probe, daemon=daemon, timeout_s=600.0)
    except BaseException:
        daemon.stop()
        raise
    finally:
        probe.close()
    return daemon, endpoint


def make_robocasa365_benchmark() -> RoboCasa365Benchmark:
    """Factory for roborsi.evaluation.service_host --kind benchmark.

    ``ROBORSI_ROBOCASA_VLA_ENDPOINT`` attaches an existing RLDX-1 service;
    without it the service starts its own from ``RLDX_CHECKPOINT``, which the
    image sets, so the Registry's single-container start needs no extra setup.
    """
    cuda = os.environ.get("ROBORSI_ROBOCASA_CUDA_DEVICE")
    return RoboCasa365Benchmark(
        os.environ.get("ROBORSI_ROBOCASA_VLA_ENDPOINT") or None,
        cuda_device=int(cuda) if cuda else None,
        cells=os.environ.get("ROBORSI_ROBOCASA_CELLS", "all"),
        vla_checkpoint=os.environ.get("RLDX_CHECKPOINT") or None,
    )
