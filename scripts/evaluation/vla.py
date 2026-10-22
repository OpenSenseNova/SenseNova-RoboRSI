# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Bind model-config VLA settings and own evaluation-scoped serving containers."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

from roborsi.agents.config import AgentConfig
from robots.libero.vla_client import (
    PROTOCOL,
    OpenPiProfile,
    endpoint_url,
    openpi_enabled,
)
from scripts.evaluation.downloads import prepare_openpi

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Deployment:
    profile: OpenPiProfile
    image: str | None
    cache_dir: Path
    endpoint: str | None = None


def _run(command, **kwargs):
    kwargs.setdefault("check", True)
    return subprocess.run(command, text=True, **kwargs)


def requests(presets, *, models=None, parameter_overrides=None) -> dict:
    """Validate every enabled profile before performing Docker or network work."""
    selected = {}
    for name, path in presets.items():
        if not name.startswith("libero"):
            continue
        overrides = [
            f"parameters.{key}={json.dumps(value)}"
            for key, value in (parameter_overrides or {}).get(name, {}).items()
        ]
        agent = AgentConfig.from_file(path, overrides=overrides)
        if not openpi_enabled(agent):
            continue
        if agent.factory not in {
            "robots.libero.vla_agent:make_agent",
            "robots.libero.three_benchmark_agent:make_three_benchmark_agent",
        }:
            raise ValueError(
                "OpenPI VLA requires a LIBERO Agent factory that translates VLA actions"
            )
        model = (models or {}).get(name)
        if model is None or model.vla is None:
            raise ValueError(
                f"{name}: enable_vla requires a vla section in the selected --model-config"
            )
        profile = model.vla
        url = (
            agent.parameters.get("vla_endpoint")
            or os.environ.get("ROBORSI_LIBERO_VLA_URL")
            or profile.endpoint
        )
        selected[name] = (profile, endpoint_url(url) if url else None)
    return selected


def plan(presets, *, models=None, parameter_overrides=None) -> dict:
    return {
        name: {
            **asdict(profile),
            "endpoint": url,
            "mode": "external" if url else "managed",
            "protocol": PROTOCOL,
        }
        for name, (profile, url) in requests(
            presets, models=models, parameter_overrides=parameter_overrides
        ).items()
    }


def _ensure_image(profile):
    dockerfile = ROOT / "scripts/deploy/openpi/Dockerfile"
    digest = hashlib.sha256(
        profile.openpi_revision.encode()
        + dockerfile.read_bytes()
        + dockerfile.with_name("Dockerfile.dockerignore").read_bytes()
        + (ROOT / "robots/libero/vla_server.py").read_bytes()
        + (ROOT / "scripts/evaluation/downloads.py").read_bytes()
    ).hexdigest()
    image = f"roborsi-openpi:{digest[:24]}"
    inspected = _run(
        ["docker", "image", "inspect", image], capture_output=True, check=False
    )
    if inspected.returncode:
        print(f"Building isolated OpenPI environment: {image}", file=sys.stderr)
        _run(
            [
                "docker",
                "build",
                "--platform",
                "linux/amd64",
                "--file",
                str(dockerfile),
                "--build-arg",
                f"OPENPI_REVISION={profile.openpi_revision}",
                "-t",
                image,
                str(ROOT),
            ]
        )
    return image


def _mounts(deployment):
    mounts = ["--mount", f"type=bind,src={deployment.cache_dir},dst=/cache"]
    checkpoint = deployment.profile.checkpoint
    if not checkpoint.startswith("gs://"):
        mounts += ["--mount", f"type=bind,src={checkpoint},dst=/checkpoint,readonly"]
        checkpoint = "/checkpoint"
    return mounts, checkpoint


def _remove(container):
    try:
        _run(["docker", "rm", "--force", container], capture_output=True, check=False)
    except OSError:
        print(f"Could not stop managed VLA container: {container}", file=sys.stderr)


def prepare_agents(presets, work_dir: Path, *, models=None) -> dict[str, Deployment]:
    selected = requests(presets, models=models)
    deployments, shared = {}, {}
    for name, (profile, url) in selected.items():
        identity = (profile, url)
        if identity not in shared:
            deployment = Deployment(
                profile, None, work_dir.resolve() / "vla/openpi-cache", url
            )
            if not url:
                image = _ensure_image(profile)
                deployment = Deployment(profile, image, deployment.cache_dir)
                prepare_openpi(image, deployment.cache_dir, profile.checkpoint)
            shared[identity] = deployment
        deployments[name] = shared[identity]
    return deployments


def _wait(url, container, profile):
    started = time.monotonic()
    next_status = started + 15
    deadline = time.monotonic() + profile.startup_timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url + "/v1/health", timeout=2) as response:
                health = json.loads(response.read(4096))
            if (
                not isinstance(health, dict)
                or health.get("protocol") != PROTOCOL
                or health.get("policy_config") != profile.policy_config
                or health.get("ready") is not True
            ):
                raise ValueError("VLA endpoint health protocol/policy mismatch")
            return
        except (OSError, urllib.error.URLError):
            pass
        if container:
            state = _run(
                ["docker", "inspect", "--format", "{{.State.Running}}", container],
                capture_output=True,
            )
            if state.stdout.strip() != "true":
                logs = _run(
                    ["docker", "logs", "--tail", "40", container],
                    capture_output=True,
                    check=False,
                )
                raise RuntimeError(
                    f"OpenPI VLA service exited:\n{logs.stdout}\n{logs.stderr}"
                )
        if time.monotonic() >= next_status:
            print(
                f"Waiting for VLA readiness: elapsed={time.monotonic() - started:.0f}s",
                file=sys.stderr,
                flush=True,
            )
            next_status = time.monotonic() + 15
        time.sleep(1)
    raise RuntimeError(
        f"OpenPI VLA did not become ready in {profile.startup_timeout_s:g}s"
    )


@contextmanager
def services(deployments, presets, work_dir, *, log_dir=None):
    owned, urls, bound = [], {}, {}
    previous_env = {}
    try:
        for name, deployment in deployments.items():
            if deployment not in urls:
                url = deployment.endpoint
                container = None
                if not url:
                    container = "roborsi-vla-" + uuid4().hex[:12]
                    owned.append(container)
                    mounts, checkpoint = _mounts(deployment)
                    _run(
                        [
                            "docker",
                            "run",
                            "-d",
                            "--platform",
                            "linux/amd64",
                            "--name",
                            container,
                            "--gpus",
                            f'"{deployment.profile.gpus}"'
                            if "," in deployment.profile.gpus
                            else deployment.profile.gpus,
                            "-p",
                            "127.0.0.1::8000",
                            *mounts,
                            deployment.image,
                            "--checkpoint",
                            checkpoint,
                            "--policy-config",
                            deployment.profile.policy_config,
                        ],
                        capture_output=True,
                    )
                    port = _run(
                        ["docker", "port", container, "8000/tcp"], capture_output=True
                    ).stdout.strip()
                    url = endpoint_url("http://" + port)
                print(f"Waiting for VLA model and warmup: {name}", file=sys.stderr)
                _wait(url, container, deployment.profile)
                urls[deployment] = url
            source = AgentConfig.from_file(presets[name])
            variable = f"ROBORSI_{name.upper()}_VLA_URL"
            previous_env[variable] = os.environ.get(variable)
            os.environ[variable] = urls[deployment]
            destination = work_dir / "presets" / f"{name}-vla.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "source_root": os.path.commonpath(
                            [str(source.source_root), str(work_dir.resolve())]
                        ),
                        "extends": str(source.path),
                        "parameters": {
                            "vla_endpoint_env": variable,
                            "vla_model": {
                                "backend": "openpi",
                                **asdict(deployment.profile),
                            },
                        },
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            bound[name] = str(destination)
        yield bound
    finally:
        for variable, value in previous_env.items():
            if value is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = value
        for container in reversed(owned):
            from scripts.evaluation.reporting import save_service_log

            save_service_log(container, log_dir or work_dir / "logs/services", _run)
            _remove(container)
