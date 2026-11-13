# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Offline target-label conversion using the pinned OmniGibson FKSolver (Lula).

Only robot assets are read. No simulator, scene, or task state is instantiated.
R1Pro ships a URDF and robot definition but no Lula descriptor, so the minimal
descriptor is generated from those assets. Evaluation uses the official IK
controller, independently of this offline FK dependency.
"""

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from robots.behavior_1k.contracts import validate_action
from robots.behavior_1k.official import EEF21, quat_to_rotvec


def _load_lula():
    # Isaac exposes this CPU extension through Kit in simulator runs. For
    # offline preparation, expose the same bundled extension without launching Kit.
    if importlib.util.find_spec("lula") is None:
        spec = importlib.util.find_spec("isaacsim")
        roots = list(spec.submodule_search_locations or []) if spec else []
        for root in roots:
            bundle = Path(root) / "exts/isaacsim.robot_motion.lula/pip_prebundle"
            if bundle.is_dir():
                sys.path.insert(0, str(bundle))
                break
    if importlib.util.find_spec("lula") is None:
        raise RuntimeError("official preparation requires the Isaac Sim Lula extension")


class R1ProFK:
    def __init__(self, *, repo_root: Path, data_root: Path):
        import yaml

        self.repo_root = Path(repo_root).resolve()
        model = Path(data_root) / "omnigibson-robot-assets/models/r1pro"
        definition_path = model / "r1pro.yaml"
        urdf_path = model / "urdf/r1pro_with_meta_links.urdf"
        source_config_path = model / "urdf/r1pro_source_cfg.yaml"
        definition = yaml.safe_load(definition_path.read_text())
        urdf = ET.parse(urdf_path).getroot()
        joints = {j.attrib["name"]: j for j in urdf.findall("joint")}
        links = {link.attrib["name"] for link in urdf.findall("link")}
        self.base_link = definition["base_footprint_link_name"]
        self.eef_links = definition["manipulation"]["eef_link_names"]
        # The released URDF omits the virtual EEF links added to USD. Apply
        # their official source-config transforms to the gripper-link FK.
        offsets = {
            item["link"]: item
            for item in yaml.safe_load(source_config_path.read_text())["eef_vis_links"]
        }
        self.eef_frames = {
            arm: (
                {
                    "parent_link": link,
                    "offset": {
                        "position": [0, 0, 0],
                        "orientation": [0, 0, 0, 1],
                    },
                }
                if link in links
                else offsets[link]
            )
            for arm, link in self.eef_links.items()
        }
        self.joint_names = definition["articulated_trunk"]["trunk_joint_names"] + [
            name
            for arm in ("left", "right")
            for name in definition["manipulation"]["arm_joint_names"][arm]
        ]
        if len(self.joint_names) != 18 or set(self.eef_links) != {"left", "right"}:
            raise ValueError("Unexpected R1Pro kinematic definition")
        self.limits = []
        for name in self.joint_names:
            joint = joints[name]
            limit = joint.find("limit")
            if joint.attrib["type"] != "revolute" or limit is None:
                raise ValueError(f"Missing revolute joint limits: {name}")
            self.limits.append(
                (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
            )
        descriptor = {
            "api_version": 1.0,
            "cspace": self.joint_names,
            "default_q": [max(low, min(high, 0.0)) for low, high in self.limits],
            "cspace_to_urdf_rules": [
                {"name": name, "rule": "fixed", "value": 0.0}
                for name, joint in joints.items()
                if name not in self.joint_names and joint.attrib["type"] != "fixed"
            ],
        }
        descriptor_text = yaml.safe_dump(descriptor, sort_keys=False)
        _load_lula()
        sys.path.insert(0, str(self.repo_root / "OmniGibson"))
        os.environ.setdefault("OMNIGIBSON_DATA_PATH", str(data_root))
        from omnigibson.utils.control_utils import FKSolver

        with tempfile.TemporaryDirectory(prefix="r1pro-fk-") as directory:
            path = Path(directory) / "descriptor.yaml"
            path.write_text(descriptor_text)
            self.solver = FKSolver(str(path), str(urdf_path))
        self.provenance = {
            "action_profile": EEF21,
            "method": "omnigibson.utils.control_utils.FKSolver",
            "label_semantics": "FK of joint command targets; not achieved observation poses",
            "frame": self.base_link,
            "eef_links": self.eef_links,
            "eef_frames": self.eef_frames,
            "orientation": "axis_angle_rotation_vector_radians",
            "position_units": "metres",
            "joint_names": self.joint_names,
            "joint_limits": self.limits,
            "joint_limit_policy": "clip offline source targets to official URDF joint limits",
            "robot_definition_sha256": hashlib.sha256(
                definition_path.read_bytes()
            ).hexdigest(),
            "urdf_sha256": hashlib.sha256(urdf_path.read_bytes()).hexdigest(),
            "source_config_sha256": hashlib.sha256(
                source_config_path.read_bytes()
            ).hexdigest(),
            "lula_descriptor_sha256": hashlib.sha256(
                descriptor_text.encode()
            ).hexdigest(),
            "fk_source_sha256": hashlib.sha256(
                (
                    self.repo_root / "OmniGibson/omnigibson/utils/control_utils.py"
                ).read_bytes()
            ).hexdigest(),
        }

    @classmethod
    def from_config(cls, cfg):
        options = cfg["benchmark_options"]
        return cls(repo_root=options["repo_root"], data_root=options["data_root"])

    def poses(self, joint_positions):
        import numpy as np
        import torch
        from omnigibson.utils.transform_utils import (
            pose_transform,
            relative_pose_transform,
        )

        poses = self.solver.get_link_poses(
            np.asarray(joint_positions, dtype=np.float64),
            [self.base_link]
            + [self.eef_frames[arm]["parent_link"] for arm in ("left", "right")],
        )
        result = {}
        for arm in ("left", "right"):
            frame = self.eef_frames[arm]
            pose = pose_transform(
                *poses[frame["parent_link"]],
                torch.tensor(frame["offset"]["position"], dtype=torch.float32),
                torch.tensor(frame["offset"]["orientation"], dtype=torch.float32),
            )
            position, quaternion = relative_pose_transform(
                *pose, *poses[self.base_link]
            )
            result[arm] = position.tolist() + quat_to_rotvec(quaternion.tolist())
        return result

    def convert(self, action):
        action = validate_action(action, action_dim=23)
        targets = action[3:14] + action[15:22]
        clipped = [
            max(low, min(high, q))
            for q, (low, high) in zip(targets, self.limits, strict=True)
        ]
        poses = self.poses(clipped)
        result = action[:3] + clipped[:4] + poses["left"] + [action[14]]
        result += poses["right"] + [action[22]]
        return validate_action(result, action_dim=21)

    def descriptor(self):
        # Convert tuples to JSON-native types for stable manifest comparisons.
        return json.loads(json.dumps(self.provenance))
