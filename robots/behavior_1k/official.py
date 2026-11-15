# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Official R1Pro 21D control configuration."""

import copy
import math

from .contracts import validate_action

EEF21 = "r1pro_eef21_v1"
JOINT23 = "r1pro_joint23_v1"
ACTION_LAYOUT = (
    ("base", 0, 3),
    ("torso", 3, 7),
    ("left_eef", 7, 13),
    ("left_gripper", 13, 14),
    ("right_eef", 14, 20),
    ("right_gripper", 20, 21),
)


def action_dimension(profile=EEF21):
    if profile != EEF21:
        raise ValueError(f"Unknown R1Pro action_profile: {profile!r}")
    return 21


def validate_include_depth(profile, include_depth):
    """Keep official evaluation on the official full-resolution RGB-D wrapper."""
    if type(include_depth) is not bool:
        raise ValueError("include_depth must be boolean")
    if not include_depth:
        raise ValueError("official requires RGBDFullResWrapper (include_depth=true)")
    return include_depth


def robot_config_for_profile(config, profile=EEF21):
    """Replace complete arm blocks: JointController-only arguments are invalid for IK."""
    action_dimension(profile)
    result = copy.deepcopy(config)
    result["action_normalize"] = False
    for arm in ("left", "right"):
        result["controller_config"][f"arm_{arm}"] = {
            "name": "InverseKinematicsController",
            "mode": "absolute_pose",
            "command_input_limits": None,
            "command_output_limits": None,
            "use_impedances": False,
            "pos_kp": 150,
            "smoothing_filter_size": 2,
        }
    return result


def quat_to_rotvec(quaternion):
    """Convert public xyzw quaternions to a canonical rotation vector for FK."""
    values = validate_action(quaternion, action_dim=4)
    norm = math.sqrt(sum(value * value for value in values))
    if norm < 1e-12:
        raise ValueError("Cannot convert a zero quaternion")
    x, y, z, w = (value / norm for value in values)
    if w < 0:
        x, y, z, w = -x, -y, -z, -w
    sine = math.sqrt(x * x + y * y + z * z)
    scale = 2.0 if sine < 1e-12 else 2 * math.atan2(sine, w) / sine
    return [x * scale, y * scale, z * scale]
