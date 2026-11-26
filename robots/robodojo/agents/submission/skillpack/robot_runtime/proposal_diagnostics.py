"""Read-only numerical evidence for reviewing a completed VLA proposal.

No inference, observation, random sampling, execution or semantic decisions live
here. All step numbers are one-based, including the first measured-to-target step.
"""
from __future__ import annotations

import hashlib
import json
import math

from tcp_pose import link6_to_tcp_pose


PREVIEW_STEPS = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50)
PREFIX_FIELDS = ("delta_cm", "path_cm", "max_step_cm", "max_rotation_step_deg",
                 "max_joint_step_rad", "grip_delta", "grip_range")


def _finite(values):
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError("Non-finite diagnostic input")
    return result


def _distance(a, b):
    return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))


def _angle(a, b):
    a, b = _finite(a), _finite(b)
    norms = math.sqrt(sum(x*x for x in a)*sum(x*x for x in b))
    if norms == 0:
        raise ValueError("Zero quaternion")
    return math.degrees(2*math.acos(min(1., abs(sum(x*y for x, y in zip(a, b))/norms))))


def build_review_aid(proposal, joint_actions, observation, *, dimensions):
    """Compute facts from all original targets and a matching measured state.

    Joint-channel widths are supplied by the shared action-dimension helper and
    checked against measured state. No new robot dimension table is introduced.
    The proposal and the execution buffer are never edited.
    """
    if observation["control_step"] != proposal["control_step"]:
        raise ValueError("Measured state is not from the proposal control step")
    state = observation["state"]
    targets = proposal["targets"]
    horizon = proposal["horizon"]
    if not targets or len(targets) != horizon or len(joint_actions) != horizon:
        raise ValueError("Full trajectory required for diagnostics")
    layout, offset = {}, 0
    if len(dimensions["arm_dim"]) != 2 or len(dimensions["ee_dim"]) != 2:
        raise ValueError("Expected the native dual-arm dimension contract")
    for arm, width, grip_width in zip(("left", "right"), dimensions["arm_dim"], dimensions["ee_dim"]):
        joints = _finite(state[arm+"_arm_joint_state"])
        grip = _finite(state[arm+"_ee_joint_state"])
        if len(joints) != width or len(grip) != grip_width or grip_width != 1 or width < 1:
            raise ValueError("Measured state differs from the shared action dimensions")
        layout[arm] = (offset, joints, grip[0])
        offset += width + grip_width
    actions = [_finite(row) for row in joint_actions]
    if any(len(row) != offset for row in actions):
        raise ValueError("Joint proposal width differs from measured state")
    bounds = proposal["out_of_bounds"]
    if len(bounds) != horizon:
        raise ValueError("Full existing bound checks required")
    arms, prefixes = {}, {str(n): {} for n in PREVIEW_STEPS if n <= horizon}
    for arm, (start, measured_joints, measured_grip) in layout.items():
        initial = link6_to_tcp_pose(state[arm+"_ee_pose"])
        poses = [initial] + [link6_to_tcp_pose([
            *target[arm]["position"], *target[arm]["quaternion_wxyz"]]) for target in targets]
        joints = [measured_joints] + [row[start:start+len(measured_joints)] for row in actions]
        grips = [measured_grip] + [row[start+len(measured_joints)] for row in actions]
        displacement = [100*_distance(initial[:3], pose[:3]) for pose in poses[1:]]
        distances = [100*_distance(a[:3], b[:3]) for a, b in zip(poses, poses[1:])]
        joint_steps = [max(abs(x-y) for x, y in zip(a, b)) for a, b in zip(joints, joints[1:])]
        rotation_steps = [_angle(a[3:], b[3:]) for a, b in zip(poses, poses[1:])]
        arms[arm] = dict(
            start_tcp_cm=[100*x for x in initial[:3]],
            end_tcp_cm=[100*x for x in poses[-1][:3]],
            end_delta_cm=[100*(y-x) for x, y in zip(initial[:3], poses[-1][:3])],
            max_displacement_cm=max(displacement), path_length_cm=sum(distances),
            max_tcp_step_cm=max(distances),
            end_rotation_deg=_angle(initial[3:], poses[-1][3:]),
            max_rotation_step_deg=max(rotation_steps),
            max_joint_step_rad=max(joint_steps),
            max_joint_step_at=joint_steps.index(max(joint_steps))+1,
            gripper=dict(start=grips[0], end=grips[-1], minimum=min(grips[1:]), maximum=max(grips[1:]),
                         change_threshold=.05,
                         changes=[dict(step=i, before=a, after=b) for i, (a, b) in
                                  enumerate(zip(grips, grips[1:]), 1) if abs(b-a) >= .05]))
        for number, summary in prefixes.items():
            n = int(number)
            summary[arm] = dict(delta_cm=[100*(y-x) for x, y in zip(initial[:3], poses[n][:3])],
                                path_cm=sum(distances[:n]), max_step_cm=max(distances[:n]),
                                max_rotation_step_deg=max(rotation_steps[:n]),
                                max_joint_step_rad=max(joint_steps[:n]),
                                grip_delta=grips[n]-grips[0],
                                grip_range=[min(grips[:n+1]), max(grips[:n+1])])
    return dict(status="available", advisory_only=True, proposal_id=proposal["proposal_id"],
                control_step=proposal["control_step"], horizon=horizon, step_numbering="one-based",
                evidence="Predicted full trajectory versus measured start; no motion executed",
                arms=arms, prefixes=prefixes,
                existing_out_of_bounds=[dict(step=i+1, flags=value) for i, value in enumerate(bounds) if value],
                joint_buffer_sha256=hashlib.sha256(json.dumps(joint_actions, separators=(",", ":"), allow_nan=False).encode()).hexdigest())
