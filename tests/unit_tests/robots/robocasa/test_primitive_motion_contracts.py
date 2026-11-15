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

"""Contracts for what the RoboCasa motion primitives report back."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robots.robocasa.primitives import RoboCasaPrimitives


class _PlanarBase:
    """A base that moves 1 cm per step along its heading unless it is blocked."""

    def __init__(
        self,
        *,
        blocked: bool = False,
        finger_creep: float = 0.0,
        turn_rate: float = 0.1,
    ) -> None:
        self.pos = np.zeros(3)
        self.yaw = 0.0
        self.blocked = blocked
        self.turn_rate = turn_rate
        self.gripper_qpos = np.array([0.02])
        # Width lost per step, as when a held object slips and the fingers close in.
        self.finger_creep = finger_creep
        self.grip_commands: list[float] = []

    @property
    def current_raw_obs(self) -> dict:
        quat = Rotation.from_euler("z", self.yaw).as_quat()
        return {"robot0_base_pos": self.pos.copy(), "robot0_base_quat": quat}

    def step(self, action) -> None:
        self.grip_commands.append(float(action[6]))
        self.gripper_qpos = self.gripper_qpos - self.finger_creep
        self.yaw += self.turn_rate * float(action[9])
        if self.blocked:
            return
        heading = np.array([np.cos(self.yaw), np.sin(self.yaw)])
        left = np.array([-np.sin(self.yaw), np.cos(self.yaw)])
        self.pos[:2] += 0.01 * (float(action[7]) * heading + float(action[8]) * left)


def _primitives(env: _PlanarBase) -> RoboCasaPrimitives:
    primitives = RoboCasaPrimitives.__new__(RoboCasaPrimitives)
    primitives.env = env
    primitives._check_cancelled = None
    primitives._recording = False
    primitives._vla_desync = False
    primitives._pos_jac = None
    primitives._fwd_offset = 0.0
    return primitives


def test_blocked_drive_is_reported_as_not_ok():
    result = _primitives(_PlanarBase(blocked=True)).move_base(forward=1.0, steps=10)

    assert result["ok"] is False
    assert result["distance_m"] == 0.0


def test_free_drive_reports_distance_and_keeps_base_moved():
    result = _primitives(_PlanarBase()).move_base(forward=1.0, steps=10)

    assert result["ok"] is True
    assert result["distance_m"] == 0.1
    assert np.allclose(result["base_moved"], [0.1, 0.0, 0.0])


def test_turn_in_place_stays_ok():
    result = _primitives(_PlanarBase()).move_base(turn=1.0, steps=10)

    assert result["ok"] is True
    assert result["distance_m"] == 0.0


def test_navigate_to_turns_toward_face_xy_after_arriving():
    env = _PlanarBase()

    result = _primitives(env).navigate_to([0.5, 0.0], tol=0.05, face_xy=[0.5, 1.0])

    facing = result["facing"]
    assert facing["faced"] is True
    assert abs(facing["heading_err_rad"]) <= 0.12
    to_point = np.array([0.5, 1.0]) - env.pos[:2]
    assert abs(np.arctan2(to_point[1], to_point[0]) - env.yaw) <= 0.12


def test_navigate_to_without_face_xy_keeps_its_result_shape():
    result = _primitives(_PlanarBase()).navigate_to([0.5, 0.0], tol=0.05)

    assert result["ok"] is True
    assert "facing" not in result


def test_face_reports_failure_when_the_base_cannot_turn():
    primitives = _primitives(_PlanarBase())
    primitives.env.step = lambda action: None

    facing = primitives._face([0.0, 1.0], "hold", max_steps=5)

    assert facing == {"faced": False, "turn_steps": 5, "heading_err_rad": 1.5708}


@pytest.mark.parametrize("face_xy", [[], [1.0], [1.0, 2.0, 3.0], [float("nan"), 1.0]])
def test_navigate_to_rejects_a_malformed_face_xy_before_moving(face_xy):
    env = _PlanarBase()

    with pytest.raises(ValueError, match="face_xy"):
        _primitives(env).navigate_to([0.5, 0.0], tol=0.05, face_xy=face_xy)

    assert env.grip_commands == []
    assert np.allclose(env.pos, 0.0)


def test_face_counts_a_turn_that_reaches_tolerance_on_its_last_step():
    # At 0.01 rad per step from yaw 0 toward 1.315 rad, the 120th and last
    # step leaves 0.115 rad, inside the 0.12 rad tolerance.
    env = _PlanarBase(turn_rate=0.01)
    point = [np.cos(1.315), np.sin(1.315)]

    result = _primitives(env).navigate_to([0.0, 0.0], face_xy=point)

    assert result["facing"] == {
        "faced": True,
        "turn_steps": 120,
        "heading_err_rad": 0.115,
    }


def test_face_on_the_point_itself_needs_no_turn():
    facing = _primitives(_PlanarBase())._face([0.0, 0.0], "hold")

    assert facing == {"faced": True, "turn_steps": 0, "heading_err_rad": 0.0}


def test_hold_never_opens_while_the_fingers_close_in():
    # The gripper action only counts by its sign, so any opening command here
    # would back the finger target off a full step from a slipping object.
    env = _PlanarBase(finger_creep=0.001)

    _primitives(env).move_base(forward=1.0, steps=10)

    assert env.grip_commands == [0.0] * 10


def test_hold_is_the_default_for_every_carry_motion():
    env = _PlanarBase()
    primitives = _primitives(env)

    primitives.navigate_to([0.5, 0.0], tol=0.05, face_xy=[0.5, 1.0])
    primitives.move_base(turn=1.0, steps=5)

    assert env.grip_commands
    assert set(env.grip_commands) == {0.0}


def test_explicit_gripper_commands_pass_through_clipped():
    primitives = _primitives(_PlanarBase())

    assert primitives._resolve_grip(1) == 1.0
    assert primitives._resolve_grip(-1.0) == -1.0
    assert primitives._resolve_grip(3.0) == 1.0
    assert primitives._resolve_grip(None) == 0.0
