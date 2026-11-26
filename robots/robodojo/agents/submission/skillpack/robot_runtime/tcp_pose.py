"""Translate X5 link6 poses and grasp-centre TCP poses at the tool boundary.

Use the LIBERO-transfer convention: TCP at 0.1501 m along native link6 +X,
and public pitch/roll/yaw degrees relative to a fixed downward orientation.
"""

from __future__ import annotations

import math


GRASP_POINT_OFFSET_M = 0.1501
TOP_DOWN_QUAT_WXYZ = (0.5, -0.5, 0.5, 0.5)
POSE_CONTRACT = {
    "reference_point": "TCP/GRASP_POINT",
    "coordinate_frame": "environment_origin",
    "position_units": "metres",
    "orientation_reference": "vertical_down",
    "orientation_units": "degrees",
    "orientation_axes": {"pitch_deg": "+X", "roll_deg": "+Y", "yaw_deg": "+Z"},
    "orientation_order": ["pitch_deg", "roll_deg", "yaw_deg"],
}


def _numbers(value, count: int) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(f"Expected {count} finite pose values")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in value):
        raise ValueError("Pose values must be finite numbers")
    return list(value)


def _shift(position, quaternion, direction: int) -> list[float]:
    xyz = _numbers(position, 3)
    q = _numbers(quaternion, 4)
    norm = math.sqrt(sum(x * x for x in q))
    if abs(norm - 1) > 1e-4:
        raise ValueError("Expected a unit wxyz quaternion")
    w, x, y, z = (value / norm for value in q)
    # First column of R(q): link6's local +X expressed in the environment frame.
    axis = (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y))
    return [p + direction * GRASP_POINT_OFFSET_M * a for p, a in zip(xyz, axis)]


def _multiply(left, right) -> list[float]:
    w, x, y, z = left
    a, b, c, d = right
    return [w*a-x*b-y*c-z*d, w*b+x*a+y*d-z*c,
            w*c-x*d+y*a+z*b, w*d+x*c-y*b+z*a]


def angles_to_quat(pitch_deg: float, roll_deg: float, yaw_deg: float) -> list[float]:
    """Apply extrinsic +X pitch, +Y roll, +Z yaw to the downward reference."""
    values = _numbers([pitch_deg, roll_deg, yaw_deg], 3)
    for value, limit in zip(values, (180, 90, 180)):
        if not -limit <= value <= limit:
            raise ValueError("Angles exceed pitch/yaw +/-180 or roll +/-90 degrees")
    rotation = [1.0, 0.0, 0.0, 0.0]
    for axis, value in enumerate(values):
        half = math.radians(value) / 2
        q = [math.cos(half), 0.0, 0.0, 0.0]
        q[axis + 1] = math.sin(half)
        rotation = _multiply(q, rotation)
    return _multiply(rotation, TOP_DOWN_QUAT_WXYZ)


def quat_to_angles(quaternion) -> tuple[float, float, float]:
    """Recover the old LIBERO-transfer canonical angles, including gimbal lock."""
    q = _numbers(quaternion, 4)
    norm = math.sqrt(sum(v*v for v in q))
    if abs(norm - 1) > 1e-4:
        raise ValueError("Expected a native unit quaternion")
    reference_inverse = [TOP_DOWN_QUAT_WXYZ[0], *(-v for v in TOP_DOWN_QUAT_WXYZ[1:])]
    w, x, y, z = _multiply([v/norm for v in q], reference_inverse)
    # Rz(yaw) Ry(roll) Rx(pitch), matching the frozen transfer harness.
    m20 = 2 * (x*z-w*y)
    roll = math.asin(max(-1.0, min(1.0, -m20)))
    if abs(math.cos(roll)) > 1e-7:
        pitch = math.atan2(2*(y*z+w*x), 1-2*(x*x+y*y))
        yaw = math.atan2(2*(x*y+w*z), 1-2*(y*y+z*z))
    else:
        yaw = 0.0
        sign = 1.0 if m20 < 0 else -1.0
        pitch = math.atan2(sign*2*(x*y-w*z), 1-2*(x*x+z*z))
    return tuple(math.degrees(v) + 0.0 for v in (pitch, roll, yaw))


def link6_to_tcp_pose(pose) -> list[float]:
    """Translate a native seven-value pose; preserve its orientation."""
    values = _numbers(pose, 7)
    return _shift(values[:3], values[3:], 1) + values[3:]


def tcp_to_link6_pose(pose) -> list[float]:
    """Use the target orientation to recover the flange position for IK."""
    values = _numbers(pose, 7)
    return _shift(values[:3], values[3:], -1) + values[3:]


def link6_to_public_pose(pose) -> dict:
    """Return TCP XYZ and reference-relative degree angles to the model."""
    native = _numbers(pose, 7)
    pitch, roll, yaw = quat_to_angles(native[3:])
    return dict(position=_shift(native[:3], native[3:], 1),
                pitch_deg=pitch, roll_deg=roll, yaw_deg=yaw)


def validate_gripper_opening(value):
    """Normalized opening: zero fully closed, one fully open."""
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("gripper_opening must be a finite number in [0, 1]")
    return value


def link6_to_tcp_target(target: dict) -> dict:
    """Serialize a native target as public TCP plus reference-relative angles."""
    return {**link6_to_public_pose([*target["position"], *target["quaternion_wxyz"]]),
            "gripper_opening": validate_gripper_opening(target["gripper_opening"])}


def tcp_to_link6_target(target: dict) -> dict:
    """Convert a public angle target to the native unit-quaternion IK target."""
    q = angles_to_quat(target["pitch_deg"], target["roll_deg"], target["yaw_deg"])
    return dict(position=_shift(target["position"], q, -1),
                quaternion_wxyz=q,
                gripper_opening=validate_gripper_opening(target["gripper_opening"]))
