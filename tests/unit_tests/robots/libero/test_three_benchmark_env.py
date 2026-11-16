# Copyright 2026 The SenseNova-RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

import json

import numpy as np
import pytest

from roborsi.evaluation.messages import Task
from robots.libero import env_server
from robots.libero.three_benchmark_env import _policy_observation, _ProEnvironment
from robots.libero.three_benchmark_service import _ensure_libero_config


def test_container_libero_config_is_materialized_once(tmp_path, monkeypatch):
    config = tmp_path / "config"
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(config))
    expected = {}
    for field, environment in {
        "assets": "ROBORSI_LIBERO_ASSETS_ROOT",
        "benchmark_root": "ROBORSI_LIBERO_BENCHMARK_ROOT",
        "datasets": "ROBORSI_LIBERO_DATASETS_ROOT",
        "bddl_files": "ROBORSI_LIBERO_BDDL_ROOT",
        "init_states": "ROBORSI_LIBERO_INIT_ROOT",
    }.items():
        expected[field] = f"/frozen/{field}"
        monkeypatch.setenv(environment, expected[field])

    _ensure_libero_config()
    assert json.loads((config / "config.yaml").read_text()) == expected
    (config / "config.yaml").write_text("existing\n")
    _ensure_libero_config()
    assert (config / "config.yaml").read_text() == "existing\n"


def test_container_libero_config_rejects_partial_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(tmp_path / "config"))
    monkeypatch.setenv("ROBORSI_LIBERO_ASSETS_ROOT", "/frozen/assets")
    with pytest.raises(ValueError, match="incomplete container LIBERO paths"):
        _ensure_libero_config()


def test_ordered_policy_observation_preserves_named_legacy_robot_state():
    raw = {
        "agentview_image": np.zeros((4, 5, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((3, 2, 3), dtype=np.uint8),
        "robot0_eef_pos": np.asarray([0.1, -0.2, 0.9]),
        "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.asarray([0.03, -0.03]),
    }
    result = _policy_observation(raw, "do the task")

    assert result["states"].shape == (8,)
    assert result["robot_state_feedback"] == {
        "contract": "libero-robot-state-feedback-v1",
        "end_effector_position": [0.1, -0.2, 0.9],
        "end_effector_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "gripper_qpos": [0.03, -0.03],
    }


def test_pro_keeps_rng_seed_separate_from_fixed_init_state(monkeypatch):
    received = {}

    def fake_make_env(task_index, seed, **kwargs):
        received.update(task_index=task_index, seed=seed, **kwargs)
        return object()

    class FakeFacade:
        def __init__(self, raw, *, meta):
            received["meta"] = meta

        def close(self):
            pass

    monkeypatch.setattr(env_server, "make_env", fake_make_env)
    monkeypatch.setattr(env_server, "LiberoEnvFacade", FakeFacade)
    task = Task(
        benchmark="libero_pro",
        task_id="libero_spatial_task-t00-i03",
        seed=17,
        instruction="pick the object",
        max_steps=10_000,
        native={
            "suite": "libero_spatial_task",
            "task_index": 0,
            "reset_protocol": "fixed_init_state",
            "init_state_id": 3,
            "official_1x_steps": 10_000,
            "max_env_steps": 10_000,
            "plus9_on_failure": False,
            "score_protocol": "final_goal_success_at_1x",
        },
        capabilities={},
    )
    environment = _ProEnvironment(task)
    try:
        assert received["seed"] == 17
        assert received["init_state_id"] == 3
        assert received["meta"]["seed"] == 17
        assert received["meta"]["init_state_id"] == 3
    finally:
        environment.close()
