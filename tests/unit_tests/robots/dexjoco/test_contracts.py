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

"""Benchmark action and observation contracts without historical policy assets."""

import math

import numpy as np
import pytest

from robots.dexjoco.contracts import (
    LIMITS,
    environment_action,
    public_observation,
    validate_vector,
)


@pytest.mark.parametrize("dual", [False, True])
def test_action_conversion_preserves_scalar_first_quaternions_and_hand_order(dual):
    row = [1.0, 2.0, 3.0, 0.0, 0.0, math.pi, *range(10, 26)]
    expected = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, *range(10, 26)]
    if dual:
        row += [4.0, 5.0, 6.0, math.pi / 2, 0.0, 0.0, *range(40, 56)]
        expected = [
            1.0,
            2.0,
            3.0,
            0.0,
            0.0,
            0.0,
            1.0,
            4.0,
            5.0,
            6.0,
            math.sqrt(0.5),
            math.sqrt(0.5),
            0.0,
            0.0,
            *range(10, 26),
            *range(40, 56),
        ]
    np.testing.assert_allclose(
        environment_action(row, dual), expected, atol=1e-15, rtol=0
    )


@pytest.mark.parametrize("bad", [True, "1", float("nan"), float("inf")])
def test_bad_row_fails_before_conversion(bad):
    row = [0.0] * 22
    row[7] = bad
    with pytest.raises(ValueError):
        validate_vector({"vector": row}, False)


@pytest.mark.parametrize("dual,width", [(False, 23), (True, 46)])
def test_no_privileged_state_or_camera_leaks(dual, width):
    raw = {
        "state": np.arange(100),
        "ego": np.arange(8 * 9 * 3, dtype=np.uint8).reshape(8, 9, 3),
        "segmentation": np.ones((8, 9)),
        "seed": 100000,
        "succeed": True,
    }
    public, images = public_observation(raw, {"base": "ego"}, dual, "Task")
    assert public == {"prompt": "Task", "state": list(range(width))}
    assert set(images) == {"ego"}
    np.testing.assert_array_equal(images["ego"], raw["ego"])
    assert not np.shares_memory(images["ego"], raw["ego"])
    assert LIMITS["water_plant"] == 1000
