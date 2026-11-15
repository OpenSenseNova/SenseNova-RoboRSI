# Copyright 2026 The RoboRSI Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Versioned DexJoCo task, action and observation contracts."""

from __future__ import annotations

import math

COMMIT = "8d23b0fab23b17a58c4b55f3942e17013aaf8267"
DATASET_REVISION = "125df5c1019e97503929ef1a0ad8f90373436afa"
PROTOCOL = "dexjoco-roborsi-compat-v1"
LIMITS = {
    "bimanual_assembly": 1500,
    "bimanual_hanoi": 1500,
    "bimanual_microwave_cook": 1100,
    "bimanual_photograph": 1000,
    "bimanual_unlock_ipad": 1200,
    "click_mouse": 1200,
    "fold_glasses": 1200,
    "hammer_nail": 1000,
    "pick_bucket": 1200,
    "pinch_tongs": 1000,
    "water_plant": 1000,
}
SEEDS = tuple(range(100000, 100005))
ALIGNMENT = [
    -4.4294e-01,
    1.3729e-06,
    1.5170,
    -3.14156462,
    -6.91584035e-05,
    -1.40317984e-03,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0.263,
    0,
    0,
    0,
]
EVALUATION = {
    "success_metric": "native_success",
    "score_metric": "native_success",
    "aggregation": "task_weighted_success_rate",
    "error_denominator": "expected",
}


def action_schema(dual: bool) -> dict:
    """Describe one absolute 30 Hz row; policy chunks are queued by the Agent."""
    width = 44 if dual else 22
    return {
        "type": "object",
        "properties": {
            "vector": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": width,
                "maxItems": width,
                "description": "Absolute xyz metres, rotation vector radians, 16 hand targets; right then left for two arms. No clipping or interpolation.",
            }
        },
        "required": ["vector"],
        "additionalProperties": False,
    }


CAPABILITIES = {
    "action_schema": {"oneOf": [action_schema(False), action_schema(True)]},
    "observation_channels": ["rgb", "state"],
}


def validate_vector(data: dict, dual: bool) -> list[float]:
    """Reject malformed actions before any simulator mutation."""
    width = 44 if dual else 22
    values = data.get("vector")
    if (
        set(data) != {"vector"}
        or not isinstance(values, list)
        or len(values) != width
        or any(type(x) not in (int, float) or not math.isfinite(x) for x in values)
    ):
        raise ValueError(f"DexJoCo requires exactly {width} finite numeric targets")
    return values


def environment_action(row: list[float], dual: bool):
    """Match the pinned evaluator's scalar-first quaternion and hand order."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    q = np.asarray(validate_vector({"vector": row}, dual), dtype=np.float64)
    right = Rotation.from_rotvec(q[3:6]).as_quat(scalar_first=True)
    if dual:
        left = Rotation.from_rotvec(q[25:28]).as_quat(scalar_first=True)
        return np.concatenate((q[:3], right, q[22:25], left, q[6:22], q[28:44]))
    return np.concatenate((q[:3], right, q[6:22]))


def public_observation(raw: dict, cameras: dict, dual: bool, prompt: str):
    """Copy only official RGB cameras, instruction and robot proprioception."""
    import numpy as np

    width = 46 if dual else 23
    state = np.asarray(raw["state"])
    if state.ndim != 1 or len(state) < width or not np.isfinite(state[:width]).all():
        raise ValueError("Invalid DexJoCo robot state")
    images = {}
    for name in dict.fromkeys(cameras.values()):
        rgb = np.asarray(raw[name])
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError("Invalid DexJoCo camera RGB")
        images[name] = np.ascontiguousarray(rgb).copy()
    return {"prompt": prompt, "state": state[:width].astype(float).tolist()}, images
