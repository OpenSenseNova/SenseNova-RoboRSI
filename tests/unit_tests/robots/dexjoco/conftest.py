# Copyright 2026 The SenseNova-RoboRSI Authors.
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

"""Small offline DexJoCo fixtures without native simulator imports."""

import numpy as np
import pytest
import yaml


@pytest.fixture
def upstream(tmp_path):
    root = tmp_path / "upstream"
    config = root / "configs/rand_full"
    config.mkdir(parents=True)
    for name in ("water_plant", "bimanual_hanoi", "click_mouse"):
        (config / f"{name}.yaml").write_text(
            yaml.safe_dump(
                {
                    "env_name": name,
                    "robot_type": "dual_arm"
                    if name.startswith("bimanual")
                    else "single_arm",
                    "camera_mapping": {"base": "ego"},
                    "prompt": "Complete " + name,
                }
            )
        )
    return root


class FakeNative:
    horizon = None

    def __init__(self, root, task, seed, multiplier):
        self.steps = 0
        self.closed = 0
        self.width = 46 if task.startswith("bimanual") else 23
        self.end_at = 3
        self.succeed = False

    def raw(self):
        return {
            "state": np.arange(self.width + 10, dtype=float),
            "ego": np.zeros((8, 8, 3), dtype=np.uint8),
            "object_pose": [999],
        }

    def reset(self):
        return self.raw(), {"private_reset": True}

    def step(self, row):
        self.steps += 1
        ended = self.steps == self.end_at
        return (
            self.raw(),
            9.0,
            ended,
            False,
            {"succeed": ended and self.succeed, "private": 42},
        )

    def close(self):
        self.closed += 1
