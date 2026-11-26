"""Geometric helpers for cuRobo with wsl4-scale incremental bounds."""
from __future__ import annotations
import math
import numpy as np
from scipy.spatial.transform import Rotation

TRANSLATION_ERROR_CAP_M=.045
ROTATION_ERROR_CAP_RAD=.3
JOINT_DELTA_CAP_RAD=.4

def matrix(pose):
    if np.asarray(pose).shape==(4,4):return np.asarray(pose,float)
    pose=np.asarray(pose,float)
    if pose.shape!=(7,)or not np.isfinite(pose).all()or abs(np.linalg.norm(pose[3:])-1)>1e-4:raise ValueError('Finite position / unit wxyz pose required')
    t=np.eye(4);t[:3,3]=pose[:3];t[:3,:3]=Rotation.from_quat(pose[[4,5,6,3]]).as_matrix();return t

def bounded_goal(now,target):
    now=matrix(now);goal=matrix(target);dp=goal[:3,3]-now[:3,3];dr=Rotation.from_matrix(goal[:3,:3]@now[:3,:3].T).as_rotvec()
    dp*=min(1.,TRANSLATION_ERROR_CAP_M/max(float(np.linalg.norm(dp)),1e-12));dr*=min(1.,ROTATION_ERROR_CAP_RAD/max(float(np.linalg.norm(dr)),1e-12))
    q=Rotation.from_matrix(Rotation.from_rotvec(dr).as_matrix()@now[:3,:3]).as_quat();return np.r_[now[:3,3]+dp,q[[3,0,1,2]]],np.r_[dp,dr]

def canonical_solution(raw,current,limits,periodic):
    raw=np.asarray(raw,float);current=np.asarray(current,float);limits=np.asarray(limits,float)
    if raw.shape!=(6,)or current.shape!=(6,)or limits.shape!=(6,2)or len(periodic)!=6 or not np.isfinite(np.r_[raw,current,limits.flatten()]).all():raise ValueError('Invalid solver / measured / limits vector')
    result=[];turns=[]
    for value,q,(lo,hi),wrap in zip(raw,current,limits,periodic):
        if lo>hi:raise ValueError('Reversed joint soft limits')
        if not wrap:
            if not lo<=value<=hi:return None,None
            result.append(float(value));turns.append(0);continue
        a=math.ceil((lo-value)/(2*math.pi));b=math.floor((hi-value)/(2*math.pi))
        if a>b:return None,None
        k=min(b,max(a,round((q-value)/(2*math.pi))));result.append(float(value+2*math.pi*k));turns.append(int(k))
    return np.asarray(result),turns

def bounded_joints(candidate,current,limits):
    candidate=np.asarray(candidate,float);current=np.asarray(current,float);limits=np.asarray(limits,float)
    if candidate.shape!=(6,)or current.shape!=(6,)or limits.shape!=(6,2)or not np.isfinite(np.r_[candidate,current,limits.flatten()]).all():raise ValueError('Invalid joint command')
    low=np.maximum(limits[:,0],current-JOINT_DELTA_CAP_RAD);high=np.minimum(limits[:,1],current+JOINT_DELTA_CAP_RAD)
    if np.any(low>high):raise ValueError('Joint outside bounded valid interval')
    return np.clip(candidate,low,high).astype(np.float32)

def pose_residual(actual,target):
    a=matrix(actual);b=matrix(target)
    return {'link6_position_error_m':float(np.linalg.norm(a[:3,3]-b[:3,3])),'tcp_position_error_m':float(np.linalg.norm(a[:3,3]+.1501*a[:3,0]-b[:3,3]-.1501*b[:3,0])),'rotation_error_rad':float(Rotation.from_matrix(b[:3,:3]@a[:3,:3].T).magnitude())}


def smooth_bounded_joints(candidate,current,limits,previous=None,alpha=.5):
    if type(alpha)not in (int,float)or not math.isfinite(alpha)or not 0<alpha<=1:raise ValueError('EMA alpha must be in (0,1]')
    bounded=bounded_joints(candidate,current,limits).astype(float)
    origin=np.asarray(current if previous is None else previous,float)
    if origin.shape!=(6,)or not np.isfinite(origin).all():raise ValueError('Invalid previous applied joint command')
    filtered=origin+alpha*(bounded-origin)
    actual=bounded_joints(filtered,current,limits)
    return actual,{'method':'joint_target_ema_then_original_bounds','alpha':float(alpha),'history_source':'actual_previous_control'if previous is not None else'measured_state_initialization','before_smoothing_joint_target':bounded.tolist(),'ema_joint_target':filtered.tolist(),'post_smoothing_bound_active':bool(np.any(np.abs(actual-filtered)>1e-6))}
