# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Lightweight OpenPI profile and LIBERO observation/action contracts."""

from __future__ import annotations

import base64
import json
import math
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

PROTOCOL = "roborsi-libero-openpi-v1"


@dataclass(frozen=True)
class OpenPiProfile:
    openpi_revision: str
    policy_config: str
    checkpoint: str
    gpus: str = "all"
    startup_timeout_s: float = 1800
    request_timeout_s: float = 120
    action_steps: int = 5
    endpoint: str | None = None


def parse_profile(document: dict, *, base_dir: Path, resolve=True) -> OpenPiProfile:
    """Validate model.yaml's VLA section, or an already resolved runtime binding."""
    from roborsi.utils.config_values import json_value, reject_secrets

    allowed = {"backend", *OpenPiProfile.__dataclass_fields__}
    required = {
        "backend",
        "openpi_revision",
        "policy_config",
        "checkpoint",
    }
    if (
        not isinstance(document, dict)
        or set(document) - allowed
        or required - set(document)
    ):
        raise ValueError("Unknown or missing VLA profile fields")
    json_value(document)
    reject_secrets(document)
    document = dict(document)
    if document["backend"] != "openpi":
        raise ValueError("model.vla requires backend: openpi")
    if document.get("endpoint") is not None:
        document["endpoint"] = endpoint_url(document["endpoint"])
    if not isinstance(document["openpi_revision"], str) or not re.fullmatch(
        r"[0-9a-f]{40}", document["openpi_revision"]
    ):
        raise ValueError("OpenPI revision must be a pinned 40-character commit SHA")
    if not isinstance(document["policy_config"], str) or not re.fullmatch(
        r"[A-Za-z0-9_]+", document["policy_config"]
    ):
        raise ValueError("Invalid OpenPI policy_config")
    checkpoint = (
        os.environ.get("ROBORSI_LIBERO_VLA_CHECKPOINT", document["checkpoint"])
        if resolve
        else document["checkpoint"]
    )
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError("VLA checkpoint must be a GCS URI or local directory")
    if checkpoint.startswith("gs://"):
        if not re.fullmatch(
            r"gs://[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+", checkpoint
        ) or ".." in checkpoint.split("/"):
            raise ValueError("Invalid VLA checkpoint GCS URI")
    elif resolve:
        checkpoint = str((base_dir / Path(checkpoint).expanduser()).resolve())
        if not Path(checkpoint).is_dir():
            raise ValueError("Local VLA checkpoint directory is missing")
    document["checkpoint"] = checkpoint
    document["gpus"] = (
        os.environ.get("ROBORSI_VLA_GPUS", document.get("gpus", "all"))
        if resolve
        else document.get("gpus", "all")
    )
    if not isinstance(document["gpus"], str) or not re.fullmatch(
        r"all|device=[A-Za-z0-9_,.-]+|[1-9][0-9]*", document["gpus"]
    ):
        raise ValueError("VLA gpus must be all, a count or device=ID[,ID]")
    for key in ("startup_timeout_s", "request_timeout_s"):
        value = document.get(key, 1800 if key == "startup_timeout_s" else 120)
        if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
            raise ValueError(f"VLA {key} must be finite and positive")
    steps = document.get("action_steps", 5)
    if type(steps) is not int or not 1 <= steps <= 50:
        raise ValueError("VLA action_steps must be an integer in [1, 50]")
    return OpenPiProfile(**{k: v for k, v in document.items() if k != "backend"})


def endpoint_url(value: str) -> str:
    """Require a credential-free HTTP service origin."""
    try:
        url = urlsplit(value)
        valid = (
            url.scheme in {"http", "https"}
            and url.hostname
            and url.username is None
            and url.password is None
            and not url.query
            and not url.fragment
            and url.path in {"", "/"}
            and (url.port is None or url.port > 0)
            and not any(c.isspace() or ord(c) < 32 for c in value)
            and "\\" not in value
        )
    except (ValueError, TypeError):
        valid = False
    if not valid:
        raise ValueError("VLA endpoint must be an HTTP(S) origin without credentials")
    return value.rstrip("/")


def openpi_enabled(config) -> bool:
    """Retain the existing external MCP mode; otherwise request model-config VLA."""
    enabled = config.parameters.get("enable_vla", False)
    if type(enabled) is not bool:
        raise ValueError("parameters.enable_vla must be a boolean")
    if (
        getattr(config, "factory", None) == "robots.libero.roborsi_27b_agent:make_agent"
        and config.parameters.get("tool_transport") == "openpi_sam3"
    ):
        return False
    if not enabled:
        return False
    variable = config.parameters.get("vla_url_env", "ROBORSI_VLA_MCP_URL")
    if not isinstance(variable, str) or not re.fullmatch(r"[A-Za-z_]\w*", variable):
        raise ValueError("parameters.vla_url_env must name an environment variable")
    if os.environ.get(variable):
        return False
    return True


def observation_payload(observation, case_dir: Path, instruction: str) -> dict:
    """Use only the current case's verified images and proprioception."""
    import numpy as np

    from roborsi.evaluation.media import CaseMediaChannel

    data = observation.data
    policy_oriented = "states" in data and "main_images" in data.get("media_fields", {})
    names = (
        ("main_images", "wrist_images")
        if policy_oriented
        else ("agentview_image", "robot0_eye_in_hand_image")
    )
    images = []
    media = CaseMediaChannel(case_dir / "media")
    for name in names:
        index = data.get("media_fields", {}).get(name)
        if type(index) is not int or not 0 <= index < len(observation.media):
            raise ValueError(f"VLA needs current LIBERO camera {name}")
        images.append(
            base64.b64encode(
                media.read(
                    observation.media[index],
                    observation.episode_id,
                    observation.step_index,
                )
            ).decode("ascii")
        )
    if policy_oriented:
        state = np.asarray(data["states"], dtype=np.float64)
    else:
        quat = np.asarray(data.get("robot0_eef_quat"), dtype=np.float64)
        if (
            quat.shape != (4,)
            or not np.isfinite(quat).all()
            or not np.isclose(np.linalg.norm(quat), 1, atol=1e-3)
        ):
            raise ValueError("VLA needs a finite unit xyzw end-effector quaternion")
        scalar = np.clip(quat[3], -1.0, 1.0)
        denominator = np.sqrt(max(0.0, 1 - scalar * scalar))
        angle = (
            np.zeros(3)
            if denominator < 1e-8
            else quat[:3] * (2 * np.arccos(scalar) / denominator)
        )
        state = np.concatenate(
            (
                np.asarray(data.get("robot0_eef_pos")),
                angle,
                np.asarray(data.get("robot0_gripper_qpos")),
            )
        )
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("VLA needs a finite eight-dimensional LIBERO state")
    return {
        "protocol": PROTOCOL,
        "images": images,
        "rotate_images": not policy_oriented,
        "state": state.tolist(),
        "instruction": instruction,
    }


def validate_actions(actions, steps: int) -> list[list[float]]:
    """Reject malformed policy output before submitting any native action."""
    if not isinstance(actions, list) or not 1 <= len(actions) <= 1000:
        raise ValueError("VLA must return a nonempty bounded action chunk")
    for vector in actions:
        if (
            not isinstance(vector, list)
            or len(vector) != 7
            or any(type(v) not in {float, int} or not math.isfinite(v) for v in vector)
        ):
            raise ValueError("VLA actions must be finite seven-dimensional vectors")
    return actions[:steps]


def predict(
    endpoint: str, profile: OpenPiProfile, observation, case_dir: Path, instruction: str
) -> list:
    payload = observation_payload(observation, case_dir, instruction)
    request = urllib.request.Request(
        endpoint_url(endpoint) + "/v1/infer",
        data=json.dumps(payload, allow_nan=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(
            request, timeout=profile.request_timeout_s
        ) as response:
            body = response.read(1024 * 1024 + 1)
        if len(body) > 1024 * 1024:
            raise ValueError("VLA response exceeds size limit")
        result = json.loads(body)
        if not isinstance(result, dict) or result.get("protocol") != PROTOCOL:
            raise ValueError("VLA response protocol mismatch")
        return validate_actions(result.get("actions"), profile.action_steps)
    except (OSError, urllib.error.URLError):
        raise RuntimeError(
            "VLA inference request failed; inspect the VLA service logs"
        ) from None
