# Copyright 2026 The RoboRSI Contributors.
# Licensed under the Apache License, Version 2.0.

"""Private native-process cuRobo execution and its one-step acknowledgement.

The policy process supplies validated EEF targets. The simulator process owns
the real solver, measured joints, previous controls and original joint step.
This module adds no model-facing tool or simulator reset/scoring implementation.
"""

import hashlib
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from robots.robodojo.contracts import get_robot_action_dim_info

FORMAT = "roborsi_native_curobo_v1"


def action_channels():
    """Use the shared native robot metadata, as at the Model boundary."""
    dimensions = get_robot_action_dim_info("arx_x5")
    if len(dimensions["arm_dim"]) != 2 or len(dimensions["ee_dim"]) != 2:
        raise ValueError("Native cuRobo requires the registered dual-arm layout")
    channels = {}
    for index, arm in enumerate(("left", "right")):
        channels[arm + "_arm_joint_state"] = dimensions["arm_dim"][index]
        channels[arm + "_ee_joint_state"] = dimensions["ee_dim"][index]
    if any(type(size) is not int or size <= 0 for size in channels.values()):
        raise ValueError("Invalid registered native action dimensions")
    return channels


def digest(action):
    return hashlib.sha256(
        json.dumps(
            action, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def envelope(request_id, step, action):
    return {
        "_roborsi_control": "bounded_curobo",
        "format": FORMAT,
        "request_id": request_id,
        "control_step": step,
        "action": deepcopy(action),
        "action_sha256": digest(action),
    }


def _load(name, path, expected):
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Native cuRobo source differs: " + path.name)
    if name in sys.modules:
        if Path(sys.modules[name].__file__).resolve() != path.resolve():
            raise ValueError("Native cuRobo module belongs to another snapshot")
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def runtime():
    root = Path(__file__).with_name("control")
    hashes = json.loads((root / "manifest.json").read_text())
    package = _load(
        "policylab", root / "policylab/__init__.py", hashes["policylab/__init__.py"]
    )
    package.api = _load(
        "policylab.api", root / "policylab/api.py", hashes["policylab/api.py"]
    )
    kin = _load(
        "_roborsi_native_curobo_fk",
        root / "eef_kinematics.py",
        hashes["eef_kinematics.py"],
    )
    _load("bounded_curobo", root / "bounded_curobo.py", hashes["bounded_curobo.py"])
    execute = _load(
        "curobo_execution", root / "curobo_execution.py", hashes["curobo_execution.py"]
    )
    return kin.DualKinematics, execute.execute


def execute_control(env, request, *, step_joint, initial_observation, record_dir=None):
    """Execute one private request through an existing native joint owner."""
    required = {
        "_roborsi_control",
        "format",
        "request_id",
        "control_step",
        "action",
        "action_sha256",
    }
    if (
        not isinstance(request, dict)
        or set(request) != required
        or request["_roborsi_control"] != "bounded_curobo"
        or request["format"] != FORMAT
        or type(request["request_id"]) is not int
        or request["request_id"] <= 0
        or type(request["control_step"]) is not int
        or request["action_sha256"] != digest(request["action"])
    ):
        raise ValueError("Invalid private native cuRobo request")
    if int(env.take_action_cnt[0]) != request["control_step"] or bool(env.end_flag[0]):
        raise ValueError("Native cuRobo request is stale or terminal")
    kinematics, execute = runtime()
    kin = getattr(env, "_roborsi_curobo_kinematics", None)
    if kin is None:
        kin = kinematics(env)
        env._roborsi_curobo_kinematics = kin

    class StateView:
        def __getattr__(self, name):
            return getattr(env, name)

        def get_obs(self):
            # The current native RGB observation is already captured. Solving
            # must not acquire another frame or change the VLA history.
            return initial_observation

    receipts = []

    def retain(receipt):
        receipts.append(deepcopy(receipt))
        if record_dir is not None:
            with (Path(record_dir) / "bounded_curobo_execution.jsonl").open(
                "a"
            ) as stream:
                stream.write(json.dumps(receipt, allow_nan=False) + "\n")

    adapter = SimpleNamespace(env=StateView(), kinematics=kin, _step_joint=step_joint)
    result = execute(adapter, request["action"], receipt_sink=retain)
    if len(receipts) != 1 or receipts[0]["status"] != "completed":
        raise RuntimeError("Missing completed native cuRobo receipt; do not resend")
    result["roborsi_execution"] = {
        "format": FORMAT,
        "request_id": request["request_id"],
        "action_sha256": request["action_sha256"],
        "receipt": receipts[0],
        "applied_action": deepcopy(result["applied_action"]),
    }
    validate_ack(
        request["request_id"],
        request["control_step"],
        request["action"],
        result["roborsi_execution"],
    )
    return result


def validate_ack(request_id, step, action, ack):
    """Bind actual native execution evidence to the current pending motion."""
    import math

    if (
        not isinstance(ack, dict)
        or ack.get("format") != FORMAT
        or ack.get("request_id") != request_id
        or ack.get("action_sha256") != digest(action)
    ):
        raise ValueError(
            "Native cuRobo acknowledgement differs from the pending request"
        )
    receipt, applied = ack.get("receipt", {}), ack.get("applied_action", {})
    if (
        receipt.get("status") != "completed"
        or receipt.get("backend") != "harness_bounded_curobo"
        or receipt.get("control_step_before") != step
        or receipt.get("control_step_after") != step + 1
        or type(receipt.get("physics_steps")) is not int
        or receipt["physics_steps"] <= 0
        or applied.get("execution_backend") != "harness_bounded_curobo"
        or applied.get("joint_command") != receipt.get("joint_command")
    ):
        raise ValueError(
            "Native cuRobo acknowledgement lacks one-step physics evidence"
        )
    command = applied["joint_command"]
    channels = action_channels()
    if not isinstance(command, dict) or set(command) != set(channels):
        raise ValueError("Native cuRobo acknowledgement lacks the actual joint command")
    for name, size in channels.items():
        values = command[name]
        if (
            not isinstance(values, list)
            or len(values) != size
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in values)
            or (size == 1 and not 0 <= values[0] <= 1)
        ):
            raise ValueError("Invalid native cuRobo executed joints")
    return deepcopy(applied)
