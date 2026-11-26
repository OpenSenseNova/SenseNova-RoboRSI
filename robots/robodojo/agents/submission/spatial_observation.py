"""Validate public calibration and stage a ready-to-use skill input file."""
from copy import deepcopy
import json
import hashlib
from pathlib import Path
import numpy as np


def validate(value, vision, *, step=None):
    if value is None:return None
    value=deepcopy(value)
    if (value.get("schema_version")!=1 or value.get("frame")!="environment_origin"
            or value.get("units")!="meters" or value.get("pixel_order")!="u,v"
            or type(value.get("control_step")) is not int or value["control_step"]<0
            or (step is not None and value["control_step"]!=step)):
        raise ValueError("Spatial calibration identity or control step differs")
    names=("cam_head","cam_left_wrist","cam_right_wrist")
    if set(value.get("cameras",{}))!=set(names):raise ValueError("Three calibrated cameras required")
    for name in names:
        c=value["cameras"][name];k=np.asarray(c["K"],float);t=np.asarray(c["T_base_camera_cv"],float)
        if (k.shape!=(3,3) or t.shape!=(4,4) or not np.isfinite(k).all() or not np.isfinite(t).all()
                or not np.allclose(k[2],[0,0,1]) or min(k[0,0],k[1,1])<=0
                or not np.allclose(t[3],[0,0,0,1])
                or not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-5)
                or not np.isclose(np.linalg.det(t[:3,:3]),1,atol=1e-5)
                or list(c["image_size"])!=list(vision[name]["color"].shape[:2])):
            raise ValueError("Spatial calibration matrix or RGB dimensions differ")
        c["K"]=k.tolist();c["T_base_camera_cv"]=t.tolist()
    return value


def stage(calibration, directory, frame, *, step):
    if calibration is None:return None
    if calibration["control_step"]!=step:raise ValueError("Refusing stale camera calibration")
    directory=Path(directory)
    cameras={name:{"image":f"{frame:05d}_{name}.jpg","K":c["K"],
                   "T_base_camera_cv":c["T_base_camera_cv"],"distortion":c.get("distortion",[])}
             for name,c in calibration["cameras"].items()}
    case={"frame":"environment_origin","units":"meters","cameras":cameras}
    path=directory/f"{frame:05d}_spatial_case.json"
    with path.open("x") as stream:json.dump(case,stream,allow_nan=False)
    identity=directory.name+":"+path.stem
    receipt={"observation_id":identity, "capture_source":calibration.get("source"),
             "camera_bindings":{name:{"observation_id":identity,
                "image_control_step":step,"pose_control_step":step,"calibration_control_step":step}
                for name in cameras},
             "control_step":step,"case_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
             "images":{name:hashlib.sha256((directory/c["image"]).read_bytes()).hexdigest()
                       for name,c in cameras.items()}}
    with path.with_suffix(".meta.json").open("x") as stream:json.dump(receipt,stream,allow_nan=False)
    latest=directory/"latest_spatial.json"
    from .runtime_records import write
    write(latest,{"case_file":path.name,"control_step":step})
    return {"case_path":"/observations/"+directory.name+"/"+path.name,
            "control_step":step,"frame":"environment_origin","units":"meters",
            "pixel_order":"u,v","camera_names":list(cameras),
            "image_sizes":{name:{"height":int(c["image_size"][0]),"width":int(c["image_size"][1])}
                           for name,c in calibration["cameras"].items()},
            "usage":"Use the registered stateless spatial localization tool; image and calibration are snapshot-bound."}
