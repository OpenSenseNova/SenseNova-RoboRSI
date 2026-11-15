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

"""Contracts for the RoboCasa env client's RPC budgets."""

from __future__ import annotations

from roborsi.robots.components.env_client_base import BaseEnvClient
from robots.robocasa.env_client import RoboCasaEnvClient

META = {"camera_h": 4, "camera_w": 4}


class RecordingRpc:
    def __init__(self):
        self.calls = []

    def call(self, method, args=(), timeout_s=None, **kwargs):
        self.calls.append((method, timeout_s))
        return META if method == "env.get_env_meta" else {"step": 0}


def test_connect_reset_uses_the_kitchen_scene_budget():
    rpc = RecordingRpc()

    RoboCasaEnvClient(rpc, expected_meta=META)

    assert ("env.reset", 600.0) in rpc.calls


def test_explicit_reset_uses_the_kitchen_scene_budget():
    rpc = RecordingRpc()
    client = RoboCasaEnvClient(rpc, expected_meta=META)
    rpc.calls.clear()

    client.reset()

    assert rpc.calls == [("env.reset", 600.0)]


def test_kitchen_budget_leaves_the_shared_default_alone():
    shared, kitchen = BaseEnvClient._TIMEOUT_S, RoboCasaEnvClient._TIMEOUT_S

    assert shared["env.reset"] == 120.0
    assert kitchen["env.step"] == shared["env.step"]
