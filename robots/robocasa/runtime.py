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

"""Own RoboCasa environment processes and attach RLDX sessions."""

from __future__ import annotations

import argparse
import os
import sys
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from roborsi.dashboard.events import DashboardEventSink
from roborsi.robots.runtime import try_spawn_server, try_wait_server
from roborsi.utils.config import get_repo_root
from roborsi.utils.daemon import ProcessDaemon, pick_free_port
from roborsi.utils.rpc import make_rpc_client
from roborsi.utils.rpc.http_rpc import HttpRpcClient

if TYPE_CHECKING:
    from roborsi.utils.rpc import RpcClient


def _spawn_env_server(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[ProcessDaemon | None, RpcClient]:
    """Spawn (or attach to) the RoboCasa env_server.

    Returns ``(daemon, rpc)`` — the daemon is ``None`` when an external
    endpoint was attached (the caller must not own it).
    """
    if args.env_endpoint is None:
        host, port = "127.0.0.1", pick_free_port()
        daemon = ProcessDaemon(
            name="env_server",
            cmd=[
                sys.executable,
                str(get_repo_root() / "robots" / "robocasa" / "env_server.py"),
                "--task-name",
                args.task_name,
                "--split",
                args.split,
                "--seed",
                str(args.seed),
                "--transport",
                "http",
                "--host",
                host,
                "--port",
                str(port),
                "--parent-watch",
                *(
                    ["--cuda-device", str(args.cuda_device)]
                    if args.cuda_device is not None
                    else []
                ),
            ],
            env_overrides={
                # EGL stays the default. A launcher on a host where EGL cannot
                # render offscreen sets MUJOCO_GL=osmesa (with
                # PYOPENGL_PLATFORM=osmesa) and the server inherits it.
                "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl"),
                "ROBOT_PLATFORM": "ROBOCASA",
                # Standard RoboRSI evaluation uses --seed directly. Do not let a
                # stale variable from legacy paired-scene runs change the reset.
                "RLDX_RESET_SEED": "",
            },
            log_path=str(Path(output_dir) / "env_server.log"),
        )
        daemon.start()
        return daemon, HttpRpcClient(f"http://{host}:{port}")
    return None, make_rpc_client(args.env_endpoint)


def _spawn_vla_server(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[ProcessDaemon | None, RpcClient]:
    """Spawn (or attach to) the RoboCasa vla_server.

    Returns ``(daemon, rpc)`` — the daemon is ``None`` when an external
    endpoint was attached (the caller must not own it).
    """
    if args.vla_endpoint is None:
        if not args.vla_model_path:
            raise ValueError(
                "--vla-model-path is required when spawning a local vla_server"
            )
        host, port = "127.0.0.1", pick_free_port()
        daemon = ProcessDaemon(
            name="vla_server",
            cmd=[
                sys.executable,
                str(get_repo_root() / "robots" / "robocasa" / "vla_server.py"),
                "--model-path",
                args.vla_model_path,
                "--transport",
                "http",
                "--host",
                host,
                "--port",
                str(port),
                "--parent-watch",
                *(
                    ["--cuda-device", str(args.cuda_device)]
                    if args.cuda_device is not None
                    else []
                ),
            ],
            log_path=str(Path(output_dir) / "vla_server.log"),
        )
        daemon.start()
        return daemon, HttpRpcClient(f"http://{host}:{port}", enable_sessions=True)
    return None, make_rpc_client(args.vla_endpoint, enable_sessions=True)


def _init_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
    components: set[str] | None,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    """Initialize every RoboCasa component, or only ``components`` when given.

    Each server can be spawned or attached-to independently: pass an
    endpoint to attach, or leave it unset to spawn a local subprocess.
    """
    from robots.robocasa.env_client import RoboCasaEnvClient
    from robots.robocasa.vla_client import RoboCasaVLAClient

    starters = {
        "env": lambda: _spawn_env_server(args, output_dir),
        "vla": lambda: _spawn_vla_server(args, output_dir),
    }
    connectors = {
        "env": lambda rpc: {
            "env_client": RoboCasaEnvClient(
                rpc,
                expected_meta={
                    "task_name": args.task_name,
                    "split": args.split,
                    "seed": args.seed,
                    "camera_h": 256,
                    "camera_w": 256,
                },
            ),
            "workdir": str(output_dir),
            "hi_res": args.hi_res or None,
        },
        "vla": lambda rpc: {"vla_client": RoboCasaVLAClient(rpc)},
    }
    # A cold pod builds the kitchen on CPU (OSMesa) while other cells do the
    # same; 120 s was not enough and those cells failed before the agent ran.
    timeouts = {"env": 600.0, "vla": 300.0}
    selected = set(starters) if components is None else set(components)
    enable_vla = getattr(args, "enable_vla", True)
    if not enable_vla:
        selected.discard("vla")
    unknown = selected.difference(starters)
    if unknown:
        raise ValueError(f"unknown RoboCasa runtime components: {sorted(unknown)}")

    pending: dict[str, tuple[ProcessDaemon | None, RpcClient]] = {}
    owned_daemons: dict[str, ProcessDaemon] = {}
    for component, starter in starters.items():
        if component in selected:
            pending[component] = try_spawn_server(
                owned_daemons,
                dashboard_events,
                component,
                starter,
            )

    runtime_kwargs: dict[str, Any] = {}
    if not enable_vla:
        runtime_kwargs["vla_client"] = None
    for component, (daemon, rpc) in pending.items():
        component_kwargs = try_wait_server(
            owned_daemons,
            dashboard_events,
            component,
            rpc,
            daemon,
            timeouts[component],
            post_fn=partial(connectors[component], rpc),
        )
        runtime_kwargs.update(component_kwargs)

    return list(owned_daemons.values()), runtime_kwargs
