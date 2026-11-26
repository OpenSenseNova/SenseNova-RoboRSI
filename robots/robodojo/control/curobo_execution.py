"""Native cuRobo with wsl4-scale bounds and no joint-target EMA.

The original simulator and joint executor are retained. Predicted FK keeps
the wider joint allowance inside the native 5 cm / 0.35 rad pose bounds.
"""
from __future__ import annotations
import json
import numpy as np
from scipy.spatial.transform import Rotation
from policylab.api import InvalidAction
from bounded_curobo import (bounded_goal, canonical_solution, bounded_joints, pose_residual,
                           TRANSLATION_ERROR_CAP_M, ROTATION_ERROR_CAP_RAD,
                           JOINT_DELTA_CAP_RAD)

def _command(action, state):
    """Apply the existing EEF contract before entering native execution."""
    if (
        not isinstance(action, dict)
        or set(action) != {"mode", "target", "reason", "assessment"}
        or action["mode"] != "eef"
    ):
        raise InvalidAction("Provide mode=eef, dual-arm target, assessment and reason")
    assessment = action["assessment"]
    if (
        not isinstance(assessment, dict)
        or set(assessment) != {"current_subgoal"}
        or not isinstance(assessment["current_subgoal"], str)
        or not assessment["current_subgoal"].strip()
        or not isinstance(action["reason"], str)
        or not action["reason"].strip()
    ):
        raise InvalidAction("Current subgoal and decision reason are required")
    targets = action["target"]
    if not isinstance(targets, dict) or set(targets) != {"left", "right"}:
        raise InvalidAction("EEF target needs left and right arms")
    command = {}
    for arm in ("left", "right"):
        target = targets[arm]
        if not isinstance(target, dict) or set(target) != {
            "position",
            "quaternion_wxyz",
            "gripper_opening",
        }:
            raise InvalidAction("Position, unit wxyz and continuous gripper opening required")
        try:
            p = np.asarray(target["position"], dtype=float)
            q = np.asarray(target["quaternion_wxyz"], dtype=float)
        except (TypeError, ValueError) as error:
            raise InvalidAction(
                "Numeric EEF position and quaternion required"
            ) from error
        if p.shape != (3,) or q.shape != (4,) or not np.isfinite(np.r_[p, q]).all():
            raise InvalidAction("Finite EEF position and quaternion required")
        if (
            abs(np.linalg.norm(q) - 1) > 1e-4
            or type(target["gripper_opening"]) not in (int, float)
            or not np.isfinite(target["gripper_opening"])
            or not 0 <= target["gripper_opening"] <= 1
        ):
            raise InvalidAction("Unit quaternion and continuous gripper opening in [0,1] required")
        old = np.asarray(state[f"{arm}_ee_pose"], dtype=float)
        if np.linalg.norm(p - old[:3]) > 0.05 + 1e-9:
            raise InvalidAction("EEF target exceeds 5 cm")
        turn = (
            Rotation.from_quat(q[[1, 2, 3, 0]])
            * Rotation.from_quat(old[[4, 5, 6, 3]]).inv()
        )
        if turn.magnitude() > 0.35 + 1e-9:
            raise InvalidAction("EEF target exceeds 0.35 rad")
        # This is the measured/FK ee_link pose, not delta_ee_pose or TCP pose.
        # RoboDojo's single-env EE API uses the same environment-origin frame.
        command[f"{arm}_ee_pose"] = np.r_[p, q].tolist()
        command[f"{arm}_ee_joint_state"] = [float(target["gripper_opening"])]
    return command


def execute(adapter, action, *, receipt_sink=None):
    env=adapter.env;state=env.get_obs()["state"];requested=_command(action,state)
    kin=adapter.kinematics;manager=env.robot_manager;check=kin.check()
    before=int(env.take_action_cnt[0]);physics_before=int(env.sim._sim_step_counter)
    receipt={"backend":"harness_bounded_curobo","control_step_before":before,
             "limits":{"translation_error_vector_m":TRANSLATION_ERROR_CAP_M,"rotation_error_vector_rad":ROTATION_ERROR_CAP_RAD,"joint_delta_from_measured_rad":JOINT_DELTA_CAP_RAD,"joint_target_ema_enabled":False,"joint_target_ema_alpha":1.0,"predicted_link6_step_m":.05,"predicted_rotation_step_rad":.35,"zero_error_translation_m":1e-6,"zero_error_rotation_rad":1e-5},
             "requested_native_action":requested,"fk_validation":check,"arms":{},"status":"solving"}
    command={};diagnostics={}
    try:
        for arm in ("left","right"):
            robot=kin.robots[arm]
            if manager.ik_solver[robot.robot_name].ee_link!=robot.ee_link_name:raise RuntimeError("cuRobo and harness tool frames differ")
            current=np.asarray(manager.get_joint(robot,[0])[0],float)
            key=manager.robot_key[manager.robot_list.index(robot)]
            limits=key.data.soft_joint_pos_limits[0,robot.arm_joint_indices].cpu().numpy()
            root=kin.root(arm);now=root@kin.fk[arm].matrix(current)
            target=np.asarray(requested[arm+"_ee_pose"],float)
            effective,delta=bounded_goal(now,target)
            zero_error=bool(np.linalg.norm(delta[:3])<=1e-6 and np.linalg.norm(delta[3:])<=1e-5)
            raw={"status":"ZeroErrorHold","joint_names":list(robot.arm_joints_name),"joint_value":current.tolist()}if zero_error else manager.solve_ik(effective.tolist(),env_idx=0,robot=robot,trans="world")
            if int(env.take_action_cnt[0])!=before or int(env.sim._sim_step_counter)!=physics_before:raise RuntimeError("IK solving advanced native state")
            info={"method":"harness_bounded_curobo","solver_status":"not_called_zero_error"if zero_error else raw.get("status"),"solver_invoked":not zero_error,"measured_joint_state":current.tolist(),"bounded_ee_target":effective.tolist(),"bounded_task_delta":delta.tolist(),"raw_joint_value":np.asarray(raw["joint_value"],float).tolist()if not zero_error and raw.get("joint_value")is not None else None,"physical_steps_during_solve":0}
            previous=None
            controls=getattr(manager.control_manager,"prev_control",None)
            if controls and len(controls)>0:
                record=controls[0].get(manager.process_name(robot.arm_name),{})
                if "position"in record:previous=np.asarray(record["position"],float)
            info["previous_joint_command"]=previous.tolist()if previous is not None else None
            if raw.get("status")in("Success","ZeroErrorHold"):
                if list(raw.get("joint_names",[]))!=list(robot.arm_joints_name):raise RuntimeError("Unexpected native IK joint order")
                names=kin.fk[arm].names;types={n:k for n,k,_,_ in kin.fk[arm].chain}
                canonical,turns=canonical_solution(raw.get("joint_value"),current,limits,[types[n]in("revolute","continuous")for n in names])
                if canonical is not None:
                    raw_pose=root@kin.fk[arm].matrix(np.asarray(raw["joint_value"],float));canonical_pose=root@kin.fk[arm].matrix(canonical)
                    equivalence=pose_residual(raw_pose,canonical_pose)
                    if equivalence["link6_position_error_m"]>1e-5 or equivalence["rotation_error_rad"]>1e-4:raise RuntimeError("Selected periodic equivalent changed FK")
                    candidate=canonical;info.update(status="Hold"if zero_error else"Success",canonical_joint_value=canonical.tolist(),equivalent_turns=turns,equivalence_check=equivalence)
                else:
                    candidate=previous if previous is not None else current;info.update(status="GuardRejected",fallback="bounded_previous_command"if previous is not None else"bounded_measured_hold",guard_reason="no_equivalent_solution_within_soft_limits")
            elif raw.get("status")=="Fail":
                candidate=previous if previous is not None else current;info.update(status="Fail",fallback="bounded_previous_command"if previous is not None else"bounded_measured_hold")
            else:raise RuntimeError("Unexpected native IK solver status")
            # Disable EMA on every manual move, including fine/contact moves.
            # Previous control is consulted only for the existing IK fallback.
            actual=bounded_joints(candidate,current,limits)
            info["smoothing"]={"method":"disabled","enabled":False,"alpha":1.0}
            # Independent joint clipping can change the IK solution's pose.
            # Backtrack the actual command, retaining the original native bound.
            for scale in (1., .5, .25, .125, .0625, .03125, 0.):
                trial=np.asarray(current+scale*(actual-current),dtype=np.float32)
                trial_pose=root@kin.fk[arm].matrix(trial)
                trial_rotation=Rotation.from_matrix(trial_pose[:3,:3]@now[:3,:3].T).magnitude()
                if (np.linalg.norm(trial_pose[:3,3]-now[:3,3])<=.05+1e-9 and
                        trial_rotation<=.35+1e-9):
                    actual=trial
                    info["predicted_step_limit_scale"]=scale
                    break
            else:
                raise RuntimeError("No joint command satisfies native predicted pose bounds")
            actual_pose=root@kin.fk[arm].matrix(actual)
            info.update(joint_value=actual.tolist(),proposed_joint_delta=(actual-current).tolist(),joint_limited=bool(np.any(np.abs(actual-np.asarray(candidate))>1e-6)),predicted_residual_to_bounded_goal=pose_residual(actual_pose,effective),predicted_residual_to_requested_goal=pose_residual(actual_pose,target),previous_command_change=(actual-previous).tolist()if previous is not None else None)
            # Error caps are incremental task-vector limits, not false claims
            # that this clipped joint update has reached the full pose target.
            command[arm+"_arm_joint_state"]=actual.tolist();command[arm+"_ee_joint_state"]=requested[arm+"_ee_joint_state"]
            diagnostics[arm]=info;receipt["arms"][arm]=info
        receipt["joint_command"]=command;receipt["status"]="executing"
        # Exactly the original harness joint interpolation/physics recorder.
        result=adapter._step_joint(command)
        if int(env.take_action_cnt[0])!=before+1:raise RuntimeError("Bounded cuRobo did not execute exactly one native action")
        if int(env.sim._sim_step_counter)<=physics_before:raise RuntimeError("Joint action has no physics evidence")
        post=result["observation"]["state"]
        for arm in ("left","right"):
            info=diagnostics[arm];pose=np.asarray(post[arm+"_ee_pose"],float)
            info["actual_residual_to_bounded_goal"]=pose_residual(pose,info["bounded_ee_target"])
            info["actual_residual_to_requested_goal"]=pose_residual(pose,requested[arm+"_ee_pose"])
        result["applied_action"]={"execution_backend":"harness_bounded_curobo","joint_command":command,"ik_diagnostics":diagnostics,"execution_mode":"model_tcp_ik","ik_used":any(x["solver_invoked"]for x in diagnostics.values()),"ik_fallback_arms":[a for a,x in diagnostics.items()if x["status"]not in("Success","Hold")]}
        receipt["status"]="completed";return result
    except BaseException as error:
        receipt.update(status="fault",error=type(error).__name__+": "+str(error))
        if int(env.take_action_cnt[0])!=before or int(env.sim._sim_step_counter)!=physics_before:raise RuntimeError("Bounded cuRobo action fault after execution; do not resend")from error
        raise
    finally:
        receipt.update(control_step_after=int(env.take_action_cnt[0]),physics_steps=int(env.sim._sim_step_counter)-physics_before)
        if receipt_sink is not None:
            receipt_sink(receipt)
        else:
            with(adapter.record_dir/"bounded_curobo_execution.jsonl").open("a")as stream:stream.write(json.dumps(receipt)+"\n")
