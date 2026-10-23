# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
"""Image-tool contracts, namespace isolation and unchanged physical budgets."""

from pathlib import Path

import numpy as np
import pytest
from jsonschema import ValidationError, validate

from roborsi.agents.config import AgentConfig
from roborsi.evaluation.messages import Action
from robots.libero import image_tools
from robots.libero.image_tool_adapter import ObservationAdapter
from robots.libero.image_tools import tool_action_schema
from robots.libero.three_benchmark_adapter import ThreeBenchmarkAdapter
from tests.unit_tests.robots.libero.test_three_benchmark_adapter import FakeEnv, _task

ROOT = Path(__file__).resolve().parents[4]


def test_generic_switch_and_unchanged_default():
    assert image_tools.enabled({}) is False
    with pytest.raises(ValueError):
        image_tools.enabled({"enable_observation_tools": "true"})
    for name in ["pro_no_vla_no_sam3", "pro_vla_sam3"]:
        config = AgentConfig.from_file(ROOT / f"configs/agents/libero/{name}.yaml")
        assert image_tools.enabled(config.parameters)
        assert "observation" not in config.factory
    config = AgentConfig.from_file(ROOT / "configs/agents/libero/pro.yaml")
    assert not image_tools.enabled(config.parameters)


@pytest.mark.parametrize("vla", [False, True])
def test_image_schema_keeps_actions_and_excludes_reset(vla):
    schema = image_tools.action_schema(vla)
    original = tool_action_schema(vla)
    for branch, baseline in zip(schema["oneOf"], original["oneOf"]):
        if (
            baseline.get("properties", {}).get("primitive", {}).get("const")
            != "segment"
        ):
            assert branch == baseline
    for name in image_tools.READ_ONLY_NAMES:
        data = {"primitive": name}
        if name == "back_project":
            data.update(row=1, col=2)
        validate(data, schema)
    with pytest.raises(ValidationError):
        validate({"primitive": "reset"}, schema)


def test_image_mismatch_fails_without_fallback(tmp_path, monkeypatch):
    source = tmp_path / image_tools.locked_manifest()["source_file"]
    source.parent.mkdir(parents=True)
    source.write_text("# wrong image")
    monkeypatch.setenv("ROBORSI_IMAGE_TOOLS_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="differs"):
        image_tools.image_manifest()
    with pytest.raises(ValueError, match="Unsupported image tool"):
        image_tools.call_image_tool("reset", {}, tmp_path)


def test_bridge_readonly_and_history_preserve_physics(tmp_path, monkeypatch):
    class CameraEnv(FakeEnv):
        def camera_artifacts(self):
            return {
                "agentview.png": np.zeros((4, 4, 3), dtype=np.uint8),
                "agentview_high.png": np.zeros((4, 4, 3), dtype=np.uint8),
                "wrist_high.png": np.zeros((4, 4, 3), dtype=np.uint8),
            }

    import robots.libero.image_tool_adapter as module

    monkeypatch.setattr(module, "image_manifest", lambda: {"sha256": "test"})
    seen = []

    def call(name, args, directory):
        seen.append((name, args))
        return {
            "step": image_tools.RecordedState(directory)
            .get(args.get("step", -1))
            .step_idx
        }

    monkeypatch.setattr(module, "call_image_tool", call)
    monkeypatch.setattr(ThreeBenchmarkAdapter, "_video_frame", lambda *_: None)
    task = _task(plus9=True)
    native = dict(task.native)
    native.pop("skill_count")
    native.update(
        suite="libero_spatial_task",
        init_state_id=0,
        score_protocol="final_goal_success_at_1x",
    )
    task = task.model_copy(
        update={
            "benchmark": "libero_pro",
            "native": native,
            "capabilities": {
                "action_schema": image_tools.action_schema(),
                "image_observation_tools": image_tools.PROFILE,
            },
        }
    )
    env = CameraEnv(task, set(), [0] * 20)
    adapter = ObservationAdapter(tmp_path, env_factory=lambda _: env)
    try:
        adapter.start(task, "episode")

        def step(data):
            return adapter.step(
                Action(
                    benchmark="libero_pro",
                    episode_id="episode",
                    step_index=adapter._decision_steps,
                    data=data,
                )
            )

        step({"primitive": "view_env_state", "step": 0})
        assert adapter._env_steps == 0
        step({"vector": [0] * 7})
        before = adapter._env_steps
        result = step({"primitive": "view_env_state", "step": 0})
        assert result.observation.data["camera_state"]["native_env_steps"] == 0
        assert adapter._env_steps == env.steps == before == 1
        assert adapter._decision_steps == 3
        assert seen == [
            ("view_env_state", {"step": 0}),
            ("view_env_state", {"step": 0}),
        ]
    finally:
        adapter.close()


@pytest.mark.parametrize("success_at,expected_steps", [(set(), 5), ({2}, 2)])
def test_image_vla_uses_original_physical_budget(
    tmp_path, monkeypatch, success_at, expected_steps
):
    import robots.libero.image_tool_adapter as module
    import robots.libero.image_vla_bridge as bridge

    monkeypatch.setattr(module, "image_manifest", lambda: {"sha256": "test"})
    monkeypatch.setattr(ThreeBenchmarkAdapter, "_video_frame", lambda *_: None)
    task = _task(plus9=False)
    native = dict(task.native)
    native.pop("skill_count")
    native.update(
        suite="libero_spatial_task",
        init_state_id=0,
        score_protocol="final_goal_success_at_1x",
        max_env_steps=5,
        official_1x_steps=5,
    )
    task = task.model_copy(
        update={
            "benchmark": "libero_pro",
            "max_steps": 5,
            "native": native,
            "capabilities": {
                "action_schema": image_tools.action_schema(True),
                "image_observation_tools": image_tools.PROFILE,
            },
        }
    )
    env = FakeEnv(task, success_at, [0] * 10)
    adapter = ObservationAdapter(
        tmp_path,
        env_factory=lambda _: env,
        tool_urls={"vla": "http://localhost:1", "sam3": "http://localhost:2"},
    )
    monkeypatch.setattr(
        adapter,
        "_observation",
        lambda values: ThreeBenchmarkAdapter._observation(adapter, values),
    )

    class Session:
        def __init__(self, a, data):
            self.a = a
            self.closed = False

        def events(self):
            for _ in range(20):
                yield [0] * 7

        def finish(self):
            assert self.a._ended
            self.a._tool_feedback = {
                "terminated": self.a._primary_success,
                "truncated": not self.a._primary_success,
            }

        def close(self):
            self.closed = True

    monkeypatch.setattr(bridge, "ImageSkillSession", Session)
    try:
        adapter.start(task, "episode")
        result = adapter.step(
            Action(
                benchmark="libero_pro",
                episode_id="episode",
                step_index=0,
                data={"primitive": "pi0_pick", "prompt": "pick", "max_chunks": 1},
            )
        )
        assert adapter._env_steps == env.steps == expected_steps
        assert result.terminated == bool(success_at)
        assert result.truncated == (not bool(success_at))
        assert result.observation.data["vla_result"]["task_success"] == bool(success_at)
        assert adapter._decision_steps == 1
    finally:
        adapter.close()


def test_openpi_action_response_validation(monkeypatch):
    import io
    import json
    import urllib.request

    payload = {"protocol": "roborsi-libero-openpi-v1", "actions": [[0.0] * 7]}

    def respond(*args, **kwargs):
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(urllib.request, "urlopen", respond)
    obs = {
        "main_images": np.zeros((1, 4, 4, 3), dtype=np.uint8),
        "wrist_images": np.zeros((1, 4, 4, 3), dtype=np.uint8),
        "states": np.zeros((1, 8)),
        "task_descriptions": ["pick"],
    }
    client = image_tools.OpenPiExecutionClient("http://localhost:1")
    assert client.call("vla.predict", args=(obs, {"mode": "eval"})).shape == (1, 1, 7)
    payload["actions"][0][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        client.call("vla.predict", args=(obs, {"mode": "eval"}))


@pytest.mark.parametrize(
    "data",
    [
        {"primitive": "move_to", "xyz": [0.2, 0, 0], "max_steps": 3},
        {
            "primitive": "move_pose",
            "xyz": [0.2, 0, 0],
            "target_pitch": 0.4,
            "target_yaw": 0.6,
            "max_steps": 3,
        },
        {"primitive": "rotate_pitch", "delta": 0.4, "max_steps": 3},
        {"primitive": "rotate_wrist", "target": 2 * np.pi + 0.4, "max_steps": 3},
        {"primitive": "release", "max_steps": 3},
        {"primitive": "set_gripper", "gripper": 1, "steps": 3},
        {"primitive": "move_to", "xyz": [0, 0, 0], "max_steps": 3},
    ],
)
@pytest.mark.parametrize("success_at,budget", [(set(), 8), ({2}, 8), (set(), 2)])
def test_image_direct_actions_preserve_physical_stops_and_feedback(
    tmp_path, monkeypatch, data, success_at, budget
):
    import robots.libero.image_tool_adapter as module

    monkeypatch.setattr(module, "image_manifest", lambda: {"sha256": "test"})
    monkeypatch.setattr(ThreeBenchmarkAdapter, "_video_frame", lambda *_: None)

    class RecordedEnv(FakeEnv):
        def __init__(self, task):
            super().__init__(task, success_at, [0] * 10)
            self.vectors = []

        def reset(self):
            values, info = super().reset()
            values["states"][3] = np.pi
            values["states"][6:] = [0.02, -0.02]
            return values, info

        def step(self, vector):
            self.vectors.append(vector)
            values, reward, terminated, truncated, info = super().step(vector)
            values["states"][3] = np.pi
            opening = [0.02, 0.035, 0.01][min(self.steps - 1, 2)]
            values["states"][6:] = [opening, -opening]
            return values, reward, terminated, truncated, info

    task = _task(plus9=False)
    native = dict(task.native)
    native.pop("skill_count")
    native.update(
        suite="libero_spatial_task",
        init_state_id=0,
        score_protocol="final_goal_success_at_1x",
        max_env_steps=budget,
        official_1x_steps=budget,
    )
    task = task.model_copy(
        update={
            "benchmark": "libero_pro",
            "max_steps": budget,
            "native": native,
            "capabilities": {
                "action_schema": image_tools.action_schema(),
                "image_observation_tools": image_tools.PROFILE,
            },
        }
    )
    env = RecordedEnv(task)
    import robots.libero.image_vla_bridge as bridge

    sessions = []

    class ImageSession:
        def __init__(self, a, command):
            self.a = a
            self.closed = False
            self.command = command
            sessions.append(self)

        def events(self):
            n = (
                0
                if self.command.get("xyz") == [0, 0, 0]
                else self.command.get("max_steps", self.command.get("steps", 3))
            )
            for _ in range(n):
                # Distinct vectors prove dispatch through the image session.
                yield [0.123, 0, 0, 0, 0, 0, -1]
            self.a._tool_feedback = {"image_feedback": "preserved", "steps_used": 999}

        def finish(self):
            self.a._tool_feedback = {"image_feedback": "preserved", "steps_used": 999}

        def close(self):
            self.closed = True

    monkeypatch.setattr(bridge, "ImageSkillSession", ImageSession)
    adapter = ObservationAdapter(tmp_path / "image", env_factory=lambda _: env)
    monkeypatch.setattr(
        adapter,
        "_observation",
        lambda values: ThreeBenchmarkAdapter._observation(adapter, values),
    )
    try:
        adapter.start(task, "episode")
        action = Action(
            benchmark="libero_pro", episode_id="episode", step_index=0, data=data
        )
        actual = adapter.step(action)
        zero = data.get("xyz") == [0, 0, 0]
        expected_steps = 0 if zero else min(3, budget, 2 if success_at else 3)
        assert env.steps == expected_steps
        assert all(v[0] == 0.123 for v in env.vectors)
        assert actual.terminated == (bool(success_at) and not zero)
        assert actual.truncated == (budget == 2 and not success_at and not zero)
        assert sessions[0].closed
        assert sessions[0].command == data
        feedback = actual.observation.data["action_result"]
        assert feedback["steps_used"] == env.steps
        assert feedback["image_feedback"] == "preserved"
        assert feedback["name"] == data["primitive"]
        assert feedback["terminated"] == actual.terminated
        assert feedback["truncated"] == actual.truncated
        if data["primitive"] in {"move_to", "move_pose"}:
            assert feedback["final_dist_m"] == pytest.approx(
                np.linalg.norm(data["xyz"]), abs=1e-4
            )
            assert feedback["target_xyz"] == data["xyz"]
            assert feedback["final_eef_pos"] == [0, 0, 0]
        if data["primitive"] == "move_pose":
            assert feedback["final_pitch_err"] == pytest.approx(0.4, abs=1e-4)
            assert feedback["final_yaw_err"] == pytest.approx(0.6, abs=1e-4)
        if data["primitive"].startswith("rotate_"):
            assert feedback["final_err"] == pytest.approx(0.4, abs=1e-4)
        if data["primitive"] == "release":
            assert feedback["start_gripper_opening"] == pytest.approx(0.04)
            assert feedback["peak_gripper_opening"] == pytest.approx(0.07)
            assert feedback["final_gripper_opening"] == pytest.approx(
                0.07 if env.steps == 2 else 0.02
            )
        assert (
            actual.model_dump(mode="json")["observation"]["data"]["action_result"]
            == feedback
        )
        if not adapter._ended:
            following = adapter.step(
                Action(
                    benchmark="libero_pro",
                    episode_id="episode",
                    step_index=1,
                    data={"vector": [0] * 7},
                )
            )
            assert "action_result" not in following.observation.data
    finally:
        adapter.close()


def test_image_rotation_boundary_preserves_native_parameter_names():
    from robots.libero.image_vla_bridge import image_arguments

    assert image_arguments({"primitive": "rotate_pitch", "delta": 0.4}) == (
        "rotate_pitch",
        {"delta_pitch": 0.4},
    )
    assert image_arguments({"primitive": "rotate_wrist", "target": 0.4}) == (
        "rotate_wrist",
        {"target_yaw": 0.4},
    )
    assert image_arguments({"primitive": "move_pose", "xyz": [0, 0, 0]}) == (
        "move_pose",
        {"xyz": [0, 0, 0]},
    )
