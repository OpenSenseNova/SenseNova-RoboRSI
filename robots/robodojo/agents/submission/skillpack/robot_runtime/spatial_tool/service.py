"""Host-only tool facade. Starts a fresh worker and discards its call context."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from .contracts import failure, public_result, tool_spec
from .validation import Limits


def configuration(overrides=None):
    config = json.loads(Path(__file__).with_name("config.json").read_text())
    config.update(deepcopy(overrides or {}))
    if set(config)-{"mode", "timeout_s", "support_planes", "limits"}:
        raise ValueError("Unknown spatial configuration field")
    tool_spec(config["mode"])
    Limits.parse(config["limits"])
    if type(config["timeout_s"]) not in (int, float) or not 0 < config["timeout_s"] <= 600:
        raise ValueError("Invalid spatial timeout")
    if not isinstance(config["support_planes"], dict):
        raise ValueError("Named support planes must be a host configuration object")
    # Plane validation does not initialize a model or touch images/memory.
    from .geometry import plane_parameters
    for name, plane in config["support_planes"].items():
        if not isinstance(name, str) or not name:
            raise ValueError("Support planes need names")
        plane_parameters(plane, "environment_origin")
    return config


class SpatialLocalizationTool:
    def __init__(self, observe, image_root, *, config=None, worker_python=None):
        self.observe = observe
        self.image_root = str(Path(image_root).resolve()) if image_root is not None else None
        self.config = configuration(config)
        self.mode = self.config["mode"]
        self.worker_python = worker_python or sys.executable

    @property
    def spec(self):
        return tool_spec(self.mode)

    def locate(self, request):
        from jsonschema import Draft202012Validator, ValidationError
        try:
            json.dumps(request, allow_nan=False)
            Draft202012Validator(self.spec["input_schema"]).validate(request)
        except (ValidationError, ValueError, TypeError):
            return failure(request if isinstance(request, dict) else {}, "invalid_geometry", "invalid_request_parameters")
        if self.image_root is None:
            return failure(request, "invalid_calibration", "capture_root_unavailable")
        try:
            # Private observe: its RGB/state payload is never returned to the caller.
            latest = self.observe()
            descriptor = latest.get("observation", {}).get("spatial_localization")
            if not descriptor:
                return failure(request, "invalid_calibration", "current_calibrated_snapshot_unavailable")
            payload = dict(mode=self.mode, request=request, descriptor=descriptor,
                           image_root=self.image_root, config=self.config)
            runtime = str(Path(__file__).resolve().parent.parent)
            env = {k: v for k, v in os.environ.items() if k in (
                "PATH", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "SYSTEMROOT", "SSL_CERT_FILE", "SSL_CERT_DIR")}
            env.update(PYTHONPATH=runtime, PYTHONDONTWRITEBYTECODE="1")
            # A private, empty cwd; no inherited /workspace, files, history or
            # tools. The read-only snapshot is host-owned. No output artifacts.
            with tempfile.TemporaryDirectory(prefix="roborsi-spatial-") as scratch:
                child = subprocess.run([self.worker_python, "-B", "-m", "spatial_tool.worker"],
                    input=json.dumps(payload, allow_nan=False), text=True, stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL, cwd=scratch, env=env,
                    timeout=self.config["timeout_s"], start_new_session=True)
            if child.returncode or len(child.stdout) > 1024*1024:
                return failure(request, "low_confidence", "localization_worker_failed")
            return public_result(json.loads(child.stdout), request)
        except subprocess.TimeoutExpired:
            return failure(request, "low_confidence", "localization_timeout")
        except (OSError, ValueError, KeyError, TypeError):
            return failure(request, "invalid_calibration", "snapshot_or_worker_unavailable")
