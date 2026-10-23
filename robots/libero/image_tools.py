# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
"""Reuse pinned image tools without mixing legacy/current Python packages.

Only the three read-only handlers cross this boundary. Robot actions, episode
termination and VLA/SAM3 execution retain the evaluation adapter's budget.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

PROFILE = "libero-image-observation-v1"
READ_ONLY_NAMES = frozenset({"view_env_state", "view_camera_meta", "back_project"})
LOCK = Path(__file__).with_name("image_tools.lock.json")


def enabled(parameters):
    value = parameters.get("enable_observation_tools", False)
    if type(value) is not bool:
        raise ValueError("enable_observation_tools must be a boolean")
    return value


def configured(path):
    from roborsi.agents.config import AgentConfig

    return enabled(AgentConfig.from_file(path).parameters)


def observation_prompt(agent):
    from dataclasses import replace

    return replace(
        agent,
        system_prompt=agent.system_prompt
        + (
            "\nThe configured Benchmark also exposes view_env_state, view_camera_meta "
            "and back_project from the pinned image. Use them through roborsi_action. "
            "Choose pixels from the matching calibrated camera image: high=1024x1024, "
            "low=256x256, row first, col second, origin top-left. Match the recorded "
            "step and camera. These read-only tools consume a decision but zero physical "
            "steps. The Benchmark alone determines task success and stopping."
        ),
    )


def locked_manifest():
    return json.loads(LOCK.read_text())


def tool_action_schema(enabled=False):
    from robots.libero.three_benchmark_adapter import ACTION_SCHEMA

    schema = copy.deepcopy(ACTION_SCHEMA)
    if enabled:
        for spec in locked_manifest()["tools"]:
            if spec["name"] not in {"pi0_pick", "pi0_doubled", "segment"}:
                continue
            branch = copy.deepcopy(spec["input_schema"])
            branch.setdefault("properties", {})["primitive"] = {"const": spec["name"]}
            branch["required"] = ["primitive", *branch.get("required", [])]
            branch["additionalProperties"] = False
            branch["description"] = spec["description"]
            schema["oneOf"].append(branch)
    return schema


def action_schema(vla=False):
    schema = tool_action_schema(vla)
    for spec in locked_manifest()["tools"]:
        if spec["name"] not in READ_ONLY_NAMES:
            continue
        branch = copy.deepcopy(spec["input_schema"])
        branch.setdefault("properties", {})["primitive"] = {"const": spec["name"]}
        branch["required"] = ["primitive", *branch.get("required", [])]
        branch["additionalProperties"] = False
        branch["description"] = spec["description"]
        schema["oneOf"].append(branch)
    return schema


def image_manifest():
    root = Path(os.environ.get("ROBORSI_IMAGE_TOOLS_ROOT", "/opt/roborsi-image-tools"))
    if not root.exists() and "ROBORSI_IMAGE_TOOLS_ROOT" not in os.environ:
        root = Path("/opt/roborsi")
    lock = locked_manifest()
    source = root / lock["source_file"]
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != lock["sha256"]:
        raise ValueError("Image tool implementation differs from image_tools.lock.json")
    return {"root": str(root), "source_file": lock["source_file"], "sha256": digest}


def _encode(value):
    if isinstance(value, bytes):
        return {"__image_bytes__": base64.b64encode(value).decode("ascii")}
    raise TypeError(f"Unsupported image-tool return type: {type(value)}")


def _decode(value):
    if set(value) == {"__image_bytes__"}:
        return base64.b64decode(value["__image_bytes__"], validate=True)
    return value


def call_image_tool(name, arguments, directory, *, endpoint=None):
    if name not in READ_ONLY_NAMES | {"segment"}:
        raise ValueError("Unsupported image tool")
    manifest = image_manifest()
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}
    }
    env["PYTHONNOUSERSITE"] = "1"
    response = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker"],
        input=json.dumps(
            {
                "manifest": manifest,
                "name": name,
                "arguments": arguments,
                "directory": str(Path(directory).resolve()),
                "endpoint": endpoint,
            }
        ),
        text=True,
        capture_output=True,
        timeout=60,
        env=env,
        cwd="/tmp",
    )
    if response.returncode:
        raise RuntimeError(f"Image tool worker failed: {response.stderr[-2000:]}")
    return json.loads(response.stdout, object_hook=_decode)


class RecordedState:
    """Satisfy the legacy EnvState read interface using current saved frames."""

    def __init__(self, directory):
        self.directory = Path(directory)

    def get(self, step=-1):
        if type(step) is not int or step < -1:
            raise ValueError("step must be -1 or a recorded nonnegative state index")
        indices = [
            int(p.name)
            for p in self.directory.iterdir()
            if p.is_dir() and p.name.isdigit()
        ]
        index = max(indices) if step == -1 else step
        if index not in indices:
            raise ValueError(f"state step {step} not available")
        folder = self.directory / f"{index:06d}"
        data = json.loads((folder / "record.json").read_text())
        return SimpleNamespace(
            step_idx=index,
            artifacts={p.name for p in folder.iterdir()},
            terminated=data["terminated"],
            truncated=data["truncated"],
            state=data["state"],
            extras={"task_language": data["task_language"]},
            command=None,
            result=None,
            elapsed_s=None,
        )

    def load_bytes(self, name, *, step=-1):
        if name != Path(name).name:
            raise ValueError("Only recorded artifact basenames are supported")
        record = self.get(step)
        if name not in record.artifacts:
            raise FileNotFoundError(name)
        return (self.directory / f"{record.step_idx:06d}" / name).read_bytes()

    def load(self, name, *, step=-1):
        import io

        import numpy as np

        value = self.load_bytes(name, step=step)
        if name.endswith(".npz"):
            with np.load(io.BytesIO(value), allow_pickle=False) as data:
                return data["value"]
        if name.endswith(".png"):
            from PIL import Image

            return np.asarray(Image.open(io.BytesIO(value)).convert("RGB"))
        return json.loads(value)

    def exists(self, name, *, step=-1):
        try:
            self.load_bytes(name, step=step)
            return True
        except (OSError, ValueError):
            return False

    def save(self, name, value, *, step=-1):
        import io

        import numpy as np

        if name != Path(name).name or not name.startswith(
            ("segment_", "segment_overlay_")
        ):
            raise ValueError("Only segmentation artifacts can be written")
        record = self.get(step)
        target = self.directory / f"{record.step_idx:06d}" / name
        if target.exists():
            raise ValueError("Refusing to overwrite a recorded artifact")
        if name.endswith(".png"):
            from PIL import Image

            b = io.BytesIO()
            Image.fromarray(np.asarray(value)).save(b, format="PNG")
            content = b.getvalue()
        else:
            content = json.dumps(value).encode()
        with target.open("xb") as f:
            f.write(content)
        return name


def worker():
    # Import legacy packages in a fresh interpreter: the current checkout's
    # identically named robots/roborsi packages never enter this interpreter.
    request = json.load(sys.stdin)
    source = Path(request["manifest"]["root"]) / request["manifest"]["source_file"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != locked_manifest()["sha256"]:
        raise ValueError("Image tools changed before worker import")
    sys.path.insert(0, request["manifest"]["root"])
    import contextlib

    with contextlib.redirect_stdout(sys.stderr):
        from robots.libero import tools

        if Path(tools.__file__).resolve() != source.resolve():
            raise ValueError("Image tool module resolved to unexpected source")
        state = RecordedState(request["directory"])
        if request["name"] == "segment":
            from roborsi.robots.components.sam3_client import Sam3Client
            from roborsi.utils.rpc.http_rpc import HttpRpcClient

            obj = SimpleNamespace(
                _sam3_client=Sam3Client(HttpRpcClient(request["endpoint"]))
            )
            result = tools.LiberoPrimitives.segment(
                obj, **request["arguments"], state=state
            )
        elif request["name"] in READ_ONLY_NAMES:
            result = getattr(tools, request["name"])(
                **request["arguments"], state=state
            )
        else:
            raise ValueError("Unsupported image tool")
    print(json.dumps(result, default=_encode))


def _expose_action_property_types(schema: dict) -> dict:
    """Expose parameter types to parsers that inspect only root properties.

    The oneOf branches remain authoritative for each primitive's required
    fields and constraints; this adds no coercion or new legal actions.
    """
    import copy

    properties = {}
    for branch in schema["oneOf"]:
        for name, field in branch["properties"].items():
            if "type" in field:
                # Keep array element types visible to the serving parser too.
                typed = {"type": field["type"]}
                if "items" in field:
                    typed["items"] = copy.deepcopy(field["items"])
            else:
                # These action discriminators are string const/enum fields.
                values = field.get("enum", [field.get("const")])
                if not all(isinstance(value, str) for value in values):
                    raise ValueError(f"action property {name} needs an explicit type")
                typed = {"type": "string"}
            if name in properties and properties[name] != typed:
                raise ValueError(f"inconsistent types for action property {name}")
            properties[name] = typed
    schema.update(type="object", properties=properties)
    return schema


def tool_endpoints() -> dict[str, str]:
    """Read explicit service settings; never infer an HTTP RPC endpoint from MCP."""
    import os

    enabled = os.environ.get("ROBORSI_LIBERO_TOOL_MODE", "no_vla_no_sam3")
    if enabled == "no_vla_no_sam3":
        return {}
    if enabled != "vla_sam3":
        raise ValueError("unknown ROBORSI_LIBERO_TOOL_MODE")
    return validate_tool_urls(
        {
            name: os.environ.get(f"ROBORSI_{name.upper()}_URL", "")
            for name in ("vla", "sam3")
        }
    )


def validate_tool_urls(endpoints: dict[str, str]) -> dict[str, str]:
    from urllib.parse import urlsplit

    if set(endpoints) != {"vla", "sam3"}:
        raise ValueError("both VLA and SAM3 service URLs are required")
    for name, value in endpoints.items():
        url = urlsplit(value)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or "<" in value
            or ">" in value
            or url.path not in {"", "/"}
            or url.query
            or url.fragment
            or url.username
            or url.password
        ):
            raise ValueError(
                f"set ROBORSI_{name.upper()}_URL to the HTTP RPC service origin"
            )
    return {name: value.rstrip("/") for name, value in endpoints.items()}


class OpenPiExecutionClient:
    """Use the OpenPI HTTP wire contract for benchmark-owned VLA execution."""

    def __init__(self, endpoint):
        self.endpoint = validate_tool_urls({"vla": endpoint, "sam3": endpoint})["vla"]

    def call(self, method, args=(), *, timeout_s=120):
        import base64
        import json
        import urllib.request

        import numpy as np
        from PIL import Image

        if method == "healthz":
            request = self.endpoint + "/v1/health"
        elif method == "vla.predict":
            obs, options = args
            if options != {"mode": "eval"}:
                raise ValueError("OpenPI execution requires eval mode")
            images = []
            for name in ("main_images", "wrist_images"):
                image = np.asarray(obs[name])
                if image.ndim != 4 or image.shape[0] != 1 or image.dtype != np.uint8:
                    raise ValueError("OpenPI execution needs batched uint8 images")
                buffer = io.BytesIO()
                Image.fromarray(image[0]).save(buffer, format="PNG")
                images.append(base64.b64encode(buffer.getvalue()).decode())
            state = np.asarray(obs["states"])
            if state.shape != (1, 8) or not np.isfinite(state).all():
                raise ValueError("OpenPI execution needs a finite [1,8] state")
            request = urllib.request.Request(
                self.endpoint + "/v1/infer",
                data=json.dumps(
                    {
                        "protocol": "roborsi-libero-openpi-v1",
                        "images": images,
                        "rotate_images": False,
                        "state": state[0].tolist(),
                        "instruction": obs["task_descriptions"][0],
                    },
                    allow_nan=False,
                ).encode(),
                headers={"Content-Type": "application/json"},
            )
        else:
            raise ValueError("unsupported OpenPI execution method")
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = response.read(1024 * 1024 + 1)
        if len(body) > 1024 * 1024:
            raise ValueError("OpenPI response exceeds size limit")
        result = json.loads(body)
        if (
            not isinstance(result, dict)
            or result.get("protocol") != "roborsi-libero-openpi-v1"
        ):
            raise ValueError("OpenPI response protocol mismatch")
        if method == "healthz":
            if result.get("ready") is not True:
                raise ValueError("OpenPI service is not ready")
            return result
        actions = result.get("actions")
        if not isinstance(actions, list) or not 1 <= len(actions) <= 1000:
            raise ValueError("VLA must return a nonempty bounded action chunk")
        for vector in actions:
            if (
                not isinstance(vector, list)
                or len(vector) != 7
                or any(
                    type(v) not in {float, int} or not math.isfinite(v) for v in vector
                )
            ):
                raise ValueError("VLA actions must be finite seven-dimensional vectors")
        return np.asarray(actions)[None]


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("Internal image-tool worker")
    worker()
