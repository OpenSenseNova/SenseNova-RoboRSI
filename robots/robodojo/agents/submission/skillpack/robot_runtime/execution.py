"""Private execution policy: stored VLA joints or measured-state TCP tracking.

Every physical update goes through the owner's original request/recording path.
No provider calls, simulator imports, motion retries or cloud lifecycle actions.
"""
from copy import deepcopy
import math
import time

from tcp_pose import link6_to_tcp_pose, tcp_to_link6_pose, validate_gripper_opening
from gripper_monitor import monitor_for


DEFAULT_MAX_NATIVE_STEPS = 80
MAX_NATIVE_STEPS = 160
POSITION_TOLERANCE_M = .003
ROTATION_TOLERANCE_RAD = math.radians(3)
FAST_APPROACH_DISTANCE_M = .04
FAST_TRANSLATION_STEP_M = .045
FINE_TRANSLATION_STEP_M = .02
FAST_ROTATION_STEP_RAD = .3
FINE_ROTATION_STEP_RAD = .1


def _action_dimensions():
    from XPolicyLab.utils.process_data import get_robot_action_dim_info
    # This native adapter's reset contract requires arx_x5.
    return get_robot_action_dim_info("arx_x5")


def stage_owned_adapter(source, config, modules):
    """Keep the configured Simate session/history wrapper when extending IK routing."""
    from pathlib import Path
    import shutil
    source, modules = Path(source), Path(modules)
    original_file, class_name = config["adapter"].rsplit(":", 1)
    entry = source / Path(original_file).name
    if not entry.is_file():
        raise ValueError("Configured native adapter entry is absent from owned source")
    for name in ("execution.py", "tcp_pose.py", "proposal_diagnostics.py", "gripper_monitor.py", "bounded_curobo.py", "curobo_execution.py"):
        shutil.copy2(modules/name, source/name)
    hybrid = source/"hybrid_adapter.py"
    suffix = "\nfrom execution import adapter_class\nAdapter = adapter_class(Adapter)\n"
    if suffix in hybrid.read_text():
        raise ValueError("Native adapter was already extended")
    hybrid.write_text(hybrid.read_text()+suffix)
    config["adapter"] = str(entry)+":"+class_name


def tracking_geometry(state, native_goal):
    """Return a bounded link6 setpoint and errors measured at the public TCP."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    targets, errors = {}, {}
    for arm in ("left", "right"):
        actual = state[arm + "_ee_pose"]
        current = link6_to_tcp_pose(actual)
        goal = native_goal[arm]
        goal_tcp = link6_to_tcp_pose([*goal["position"], *goal["quaternion_wxyz"]])
        rotation = Rotation.from_quat(np.asarray(actual[3:])[[1, 2, 3, 0]])
        desired = Rotation.from_quat(np.asarray(goal["quaternion_wxyz"])[[1, 2, 3, 0]])
        dp = np.asarray(goal_tcp[:3]) - current[:3]
        dr = (desired * rotation.inv()).as_rotvec()
        # An open gripper can approach a distant, model-chosen pregrasp pose
        # quickly. Keep the final approach, closed-gripper motion and contact
        # corrections at the original task-space resolution.
        fast_approach = (goal["gripper_opening"] >= .9 and
                         np.linalg.norm(dp) > FAST_APPROACH_DISTANCE_M)
        translation_cap = FAST_TRANSLATION_STEP_M if fast_approach else FINE_TRANSLATION_STEP_M
        rotation_cap = FAST_ROTATION_STEP_RAD if fast_approach else FINE_ROTATION_STEP_RAD
        errors[arm] = dict(position_error_m=float(np.linalg.norm(dp)),
                           orientation_error_rad=float(np.linalg.norm(dr)))
        # TCP interpolation and quaternion rotation: never interpolate Euler angles.
        # The resulting link6 displacement includes the TCP offset's rotation.
        # Shrink both deltas if needed to stay inside the original 5 cm bound.
        scale = 1.
        for _ in range(16):
            fraction_p = min(1., translation_cap / max(np.linalg.norm(dp), 1e-12)) * scale
            fraction_r = min(1., rotation_cap / max(np.linalg.norm(dr), 1e-12)) * scale
            q = (Rotation.from_rotvec(dr * fraction_r) * rotation).as_quat()[[3, 0, 1, 2]]
            pose = tcp_to_link6_pose([*(np.asarray(current[:3]) + dp * fraction_p).tolist(), *q.tolist()])
            candidate = dict(position=pose[:3], quaternion_wxyz=pose[3:],
                             gripper_opening=goal["gripper_opening"])
            if np.linalg.norm(np.asarray(candidate["position"]) - actual[:3]) <= .05 + 1e-9:
                break
            scale *= .5
        targets[arm] = candidate
    return targets, errors


def _at_goal(errors):
    return all(e["position_error_m"] <= POSITION_TOLERANCE_M and
               e["orientation_error_rad"] <= ROTATION_TOLERANCE_RAD for e in errors.values())


def execute_tcp(owner, req, native_request, *, invalid_action_type):
    """Follow each supplied TCP waypoint using actual feedback; do not skip failures."""
    before = owner.latest["observation"]["control_step"]
    budget = req.get("max_native_steps", DEFAULT_MAX_NATIVE_STEPS)
    if type(budget) is not int or not 1 <= budget <= MAX_NATIVE_STEPS:
        raise invalid_action_type("max_native_steps must be an integer from 1 to 160")
    completed, status, errors = 0, "reached", {}
    monitor, gripper_event = monitor_for(owner), None
    for goal in req["targets"]:
        waypoint_start = owner.latest["observation"]["control_step"]
        state = owner.public["observation"]["state"]
        minimum_steps = 12 if any(abs(float(state[arm+"_ee_joint_state"][0]) -
                                       goal[arm]["gripper_opening"]) > .02
                                  for arm in ("left", "right")) else 2
        previous_cost, last_progress = None, waypoint_start
        # At least one update applies the requested gripper command even when
        # the arm pose already matches. Pose convergence does not verify grasp.
        while True:
            step = owner.latest["observation"]["control_step"]
            if owner.latest.get("done"):
                status = "native_done"; break
            if owner.public["observation"].get("review_due"):
                status = "review_due"; break
            if time.monotonic() >= owner.deadline:
                status = "wall_limit"; break
            target, errors = tracking_geometry(owner.public["observation"]["state"], goal)
            if step - waypoint_start >= minimum_steps and _at_goal(errors):
                completed += 1; break
            if step - before >= budget:
                status = "native_step_budget_exhausted"; break
            cost = sum(e["position_error_m"] + .15 * e["orientation_error_rad"]
                       for e in errors.values())
            if previous_cost is None or previous_cost - cost > .0002:
                previous_cost, last_progress = cost, step
            elif step - last_progress >= max(8, minimum_steps):
                status = "stalled_reobserve"; break
            monitor.prepare(owner.latest["observation"], target)
            native_request(owner, dict(op="act", repeat=1, action=dict(
                mode="eef", target=target, reason=req["reason"], assessment=req["assessment"])))
            if owner.latest["observation"]["control_step"] <= step:
                status = "no_progress_reobserve"; break
            gripper_event = monitor.sample(owner.latest["observation"])
            if gripper_event is not None:
                status = "gripper_change_reobserve"; break
        if status != "reached":
            break
    # Report the latest actual residual even when the final update terminated.
    if completed < len(req["targets"]):
        _, errors = tracking_geometry(owner.public["observation"]["state"], req["targets"][completed])
    return dict(status=status, action_source="model_tcp_closed_loop",
                executed_steps=owner.latest["observation"]["control_step"] - before,
                completed_targets=completed, requested_targets=len(req["targets"]),
                max_native_steps=budget, pose_errors=errors,
                position_tolerance_m=POSITION_TOLERANCE_M,
                orientation_tolerance_rad=ROTATION_TOLERANCE_RAD,
                grasp_verification="Requires RGB evidence; pose convergence alone does not verify grasp",
                **({"gripper_event": gripper_event} if gripper_event else {}),
                **deepcopy(owner.public))


def execute_vla(owner, req, native_request, *, invalid_action_type):
    """Accept the exact private joint prefix, preserving the VLA's timing."""
    from policylab.common import event
    pending = getattr(owner, "vla_pending", None)
    if pending is None or req.get("proposal_id") != pending["proposal_id"]:
        raise invalid_action_type("Current proposal_id required")
    if pending["control_step"] != owner.latest["observation"]["control_step"]:
        raise invalid_action_type("Stale proposal cannot execute")
    if time.monotonic() >= owner.deadline or owner.segment_done or owner.fault or owner.latest["done"]:
        raise invalid_action_type("Episode ended or environment unavailable")
    if owner.public["observation"].get("review_due"):
        raise invalid_action_type("Checkpoint review due")
    span = req.get("approved_steps", 5)
    if type(span) is not int or not 5 <= span <= 50 or req.get("targets") is not None:
        raise invalid_action_type("Accept requires a stored 5..50-step prefix without replacement targets")
    if not isinstance(req.get("reason"), str) or not req["reason"].strip():
        raise invalid_action_type("Decision reason required")
    assessment = req.get("assessment")
    if not isinstance(assessment, dict) or set(assessment) != {"current_subgoal"} or not isinstance(assessment["current_subgoal"], str) or not assessment["current_subgoal"].strip():
        raise invalid_action_type("assessment.current_subgoal required")
    if any(pending["out_of_bounds"][:span]):
        raise invalid_action_type("Cannot accept an out-of-bounds proposal")
    if pending.get("execution_mode") != "stored_vla_joint":
        raise invalid_action_type("This proposal has no private joint execution buffer")
    before = owner.latest["observation"]["control_step"]
    log = owner.active["record_dir"] / "vla_review.jsonl"
    event(log, event="review", proposal_id=pending["proposal_id"], decision="accept",
          approved_steps=span, reason=req["reason"], assessment=assessment,
          execution_mode="stored_vla_joint", selected_targets=pending["targets"][:span])
    monitor, gripper_event, status = monitor_for(owner), None, "executed"
    try:
        for index in range(span):
            step = owner.latest["observation"]["control_step"]
            monitor.prepare(owner.latest["observation"], pending["targets"][index])
            native_request(owner, dict(op="act", repeat=1, action=dict(
                mode="stored_vla_joint", proposal_id=pending["proposal_id"], index=index,
                reason=req["reason"], assessment=assessment)))
            if owner.latest["observation"]["control_step"] <= step:
                status = "no_progress_reobserve"; break
            gripper_event = monitor.sample(owner.latest["observation"])
            if gripper_event is not None:
                status = "gripper_change_reobserve"; break
            if owner.latest["done"] or owner.public["observation"].get("review_due"):
                break
    finally:
        moved = owner.latest["observation"]["control_step"] - before
        if moved:
            owner.vla_pending = None
        event(log, event="execution", proposal_id=pending["proposal_id"],
              execution_mode="stored_vla_joint", approved_steps=span,
              executed_steps=moved, remaining_queue_cancelled=moved < span,
              status=status, **({"gripper_event": gripper_event} if gripper_event else {}))
    owner.vla_pending = None
    return dict(status=status, action_source="vla_original_joints", executed_steps=moved,
                **({"gripper_event": gripper_event} if gripper_event else {}),
                **deepcopy(owner.public))


def adapter_class(base):
    """Extend an owned runtime copy, without changing frozen benchmark sources."""
    class Adapter(base):
        def __init__(self, environment_dir, options):
            backend = options.get("eef_execution_backend", "legacy_dls")
            if backend not in {"legacy_dls", "bounded_curobo"}:
                raise ValueError("eef_execution_backend must be legacy_dls or bounded_curobo")
            self._eef_execution_backend = backend
            super().__init__(environment_dir, options)

        def _cache_review_observation(self, observation):
            self._review_aid_observation = deepcopy({key: observation[key] for key in
                                                   ("control_step", "state") if key in observation})

        def _transition(self, *args, **kwargs):
            result = super()._transition(*args, **kwargs)
            # Reuse an observation the original adapter already emitted. No new
            # camera read, physics step or policy history entry is introduced.
            self._cache_review_observation(result.get("observation", {}))
            return result

        def vla_propose(self, subgoal):
            import json
            path = self.record_dir / "vla_proposals_private.jsonl"
            offset = path.stat().st_size if path.exists() else 0
            proposal = super().vla_propose(subgoal)
            with path.open() as stream:
                stream.seek(offset)
                record = json.loads(stream.readline())
            if record["proposal_id"] != proposal["proposal_id"]:
                raise RuntimeError("Private joint buffer provenance mismatch")
            self._joint_proposal = dict(id=proposal["proposal_id"], cursor=0,
                                       control_step=proposal["control_step"],
                                       actions=record["clipped_joint_actions"][:proposal["horizon"]])
            # Preview the same continuous openings that acceptance executes.
            for target, values in zip(proposal["targets"], self._joint_proposal["actions"]):
                for arm, index in (("left", 6), ("right", 13)):
                    target[arm].pop("gripper_closed", None)
                    target[arm]["gripper_opening"] = validate_gripper_opening(values[index])
            proposal["execution_mode"] = "stored_vla_joint"
            proposal["gripper_conversion"] = "Continuous opening in [0,1]: 0 fully closed, 1 fully open; preview and execution use the same stored values"
            try:
                from proposal_diagnostics import build_review_aid
                proposal["review_aid"] = build_review_aid(
                    proposal, self._joint_proposal["actions"], self._review_aid_observation,
                    dimensions=_action_dimensions())
            except Exception as error:
                # Diagnostics must never invalidate an otherwise usable proposal.
                proposal["review_aid"] = dict(status="unavailable", advisory_only=True,
                                              error_type=type(error).__name__, reason=str(error))
            return proposal

        def _step_eef(self, action):
            """Route manual EEF targets; stored VLA joints bypass the solver."""
            if getattr(self, "_eef_execution_backend", "legacy_dls") == "bounded_curobo":
                from curobo_execution import execute
                return execute(self, action)
            import numpy as np
            from scipy.spatial.transform import Rotation
            from policylab.api import InvalidAction
            if not isinstance(action, dict) or set(action) != {"mode", "target", "reason", "assessment"} or action["mode"] != "eef":
                raise InvalidAction("Provide mode=eef, dual-arm target, assessment and reason")
            assessment = action["assessment"]
            if (not isinstance(assessment, dict) or set(assessment) != {"current_subgoal"}
                    or not isinstance(assessment["current_subgoal"], str)
                    or not assessment["current_subgoal"].strip()):
                raise InvalidAction("assessment.current_subgoal is required")
            if not isinstance(action["reason"], str) or not action["reason"].strip():
                raise InvalidAction("Decision reason is required")
            targets = action["target"]
            if not isinstance(targets, dict) or set(targets) != {"left", "right"}:
                raise InvalidAction("EEF target needs left and right arms")
            state = self.env.get_obs()["state"]
            ik_targets = {}
            for arm in ("left", "right"):
                target = targets[arm]
                if not isinstance(target, dict) or set(target) != {"position", "quaternion_wxyz", "gripper_opening"}:
                    raise InvalidAction("EEF targets need position, quaternion_wxyz and gripper_opening")
                try:
                    p = np.asarray(target["position"], float)
                    q = np.asarray(target["quaternion_wxyz"], float)
                    opening = validate_gripper_opening(target["gripper_opening"])
                except (TypeError, ValueError) as error:
                    raise InvalidAction(str(error)) from error
                if p.shape != (3,) or q.shape != (4,) or not np.isfinite(np.r_[p, q]).all() or abs(np.linalg.norm(q)-1) > 1e-4:
                    raise InvalidAction("Finite EEF position and unit quaternion required")
                old = np.asarray(state[arm+"_ee_pose"], float)
                rotation = Rotation.from_quat(q[[1,2,3,0]]) * Rotation.from_quat(old[[4,5,6,3]]).inv()
                if np.linalg.norm(p-old[:3]) > .05+1e-9 or rotation.magnitude() > .35+1e-9:
                    raise InvalidAction("EEF target exceeds measured-state 5cm/0.35rad bounds")
                # The frozen IK signature requires this private compatibility
                # flag. The actual actuator command below always uses opening.
                ik_targets[arm] = dict(target, gripper_closed=opening < .5)
            proposal = self.kinematics.target(ik_targets)
            values = list(proposal["action"])
            values[6], values[13] = targets["left"]["gripper_opening"], targets["right"]["gripper_opening"]
            command = dict(left_arm_joint_state=values[:6], left_ee_joint_state=values[6:7],
                           right_arm_joint_state=values[7:13], right_ee_joint_state=values[13:14])
            result = self._step_joint(command)
            result["applied_action"] = dict(joint_command=command, ik_diagnostics=proposal["diagnostics"],
                                           execution_mode="model_tcp_ik", ik_used=True)
            return result

        def step(self, action):
            if isinstance(action, dict) and action.get("mode") == "eef":
                return self._step_eef(action)
            if not isinstance(action, dict) or action.get("mode") != "stored_vla_joint":
                return super().step(action)
            import numpy as np
            from scipy.spatial.transform import Rotation
            from policylab.api import InvalidAction
            pending = getattr(self, "_joint_proposal", None)
            index = action.get("index")
            if (pending is None or action.get("proposal_id") != pending["id"] or
                    type(index) is not int or index != pending["cursor"] or
                    int(self.env.take_action_cnt[0]) != pending["control_step"] + index or
                    not 0 <= index < len(pending["actions"])):
                raise InvalidAction("Stale or out-of-order private VLA joint step")
            values = np.asarray(pending["actions"][index], float)
            if values.shape != (14,) or not np.isfinite(values).all():
                raise InvalidAction("Invalid stored VLA joint action")
            state = self.env.get_obs()["state"]
            for arm, offset in (("left", 0), ("right", 7)):
                robot = self.kinematics.robots[arm]
                key = self.kinematics.manager.robot_key[self.kinematics.manager.robot_list.index(robot)]
                limits = key.data.soft_joint_pos_limits[0, robot.arm_joint_indices].cpu().numpy()
                joints = values[offset:offset+6]
                if np.any(joints < limits[:, 0]) or np.any(joints > limits[:, 1]):
                    raise InvalidAction("Stored VLA target exceeds joint limits")
                goal = self.kinematics.root(arm) @ self.kinematics.fk[arm].matrix(joints)
                actual = np.asarray(state[arm+"_ee_pose"])
                rotation = Rotation.from_matrix(goal[:3, :3]) * Rotation.from_quat(actual[[4,5,6,3]]).inv()
                if np.linalg.norm(goal[:3, 3] - actual[:3]) > .05 + 1e-9 or rotation.magnitude() > .35 + 1e-9:
                    raise InvalidAction("Stored VLA step exceeds measured-state 5cm/0.35rad bounds")
            command = dict(left_arm_joint_state=values[:6].tolist(), left_ee_joint_state=values[6:7].tolist(),
                           right_arm_joint_state=values[7:13].tolist(), right_ee_joint_state=values[13:14].tolist())
            result = self._step_joint(command)
            pending["cursor"] += 1
            result["applied_action"] = dict(joint_command=command, execution_mode="stored_vla_joint",
                                           proposal_id=pending["id"], index=index, ik_used=False)
            return result
    return Adapter
