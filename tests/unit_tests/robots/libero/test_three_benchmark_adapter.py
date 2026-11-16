# Copyright 2026 The SenseNova-RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Three-Benchmark Adapter preserves 1x metrics across continuous recovery."""

from pathlib import Path

import numpy as np
import pytest

from roborsi.evaluation.messages import Action, Task
from robots.libero.three_benchmark_adapter import ACTION_SCHEMA, ThreeBenchmarkAdapter


class FakeEnv:
    def __init__(self, task: Task, successes: set[int], progress: list[float]):
        self.task = task
        self.successes = successes
        self.progress = progress
        self.steps = 0
        self.closed = False

    def identity(self):
        return self.task.native

    def instruction(self):
        return self.task.instruction

    def reset(self):
        return {
            "main_images": np.zeros((8, 8, 3), dtype=np.uint8),
            "states": np.zeros(8, dtype=np.float32),
        }, {}

    def step(self, action):
        self.steps += 1
        success = self.steps in self.successes
        return (
            {
                "main_images": np.full((8, 8, 3), self.steps, dtype=np.uint8),
                "states": np.zeros(8, dtype=np.float32),
            },
            float(success),
            success,
            False,
            {
                "ordered_success": success,
                "prefix_progress": self.progress[self.steps - 1],
            },
        )

    def close(self):
        self.closed = True


def _task(*, plus9: bool) -> Task:
    native = {
        "suite": "ultra_long_all",
        "task_index": 0,
        "reset_protocol": "fixed_init_state",
        "init_state_id": 1,
        "official_1x_steps": 2,
        "max_env_steps": 20 if plus9 else 2,
        "plus9_on_failure": plus9,
        "score_protocol": "ordered_prefix_at_1x",
        "skill_count": 2,
    }
    return Task(
        benchmark="libero_ultra_long",
        task_id="t00-i01",
        seed=0,
        instruction="do two skills",
        max_steps=native["max_env_steps"],
        native=native,
        capabilities={"action_schema": ACTION_SCHEMA},
    )


def _action(index: int, vectors=1) -> Action:
    data = {"vectors": [[0.0] * 7 for _ in range(vectors)]}
    return Action(
        benchmark="libero_ultra_long",
        episode_id="episode",
        step_index=index,
        data=data,
    )


def test_default_stops_at_1x_and_scores_ordered_prefix(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        ThreeBenchmarkAdapter, "_video_frame", lambda self, values: None
    )
    task = _task(plus9=False)
    env = FakeEnv(task, set(), [0.5, 0.5])
    adapter = ThreeBenchmarkAdapter(tmp_path, env_factory=lambda _: env)
    adapter.start(task, "episode")
    step = adapter.step(_action(0, vectors=2))
    assert step.truncated and step.termination_reason == "official_1x_limit"
    result = adapter.finish()
    assert result.success is False and result.score == 0.5
    assert result.details["completed_env_steps"] == 2


def test_plus9_continues_same_env_but_keeps_failed_1x_primary(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(
        ThreeBenchmarkAdapter, "_video_frame", lambda self, values: None
    )
    task = _task(plus9=True)
    env = FakeEnv(task, {4}, [0.5, 0.5, 0.5, 1.0])
    adapter = ThreeBenchmarkAdapter(tmp_path, env_factory=lambda _: env)
    adapter.start(task, "episode")
    first = adapter.step(_action(0, vectors=2))
    assert not first.terminated and not first.truncated
    final = adapter.step(_action(1, vectors=3))
    assert final.truncated and final.termination_reason == "recovered_after_1x"
    result = adapter.finish()
    assert result.success is False and result.score == 0.5
    assert result.details["cumulative_success"] is True
    assert result.details["completed_env_steps"] == 4


def test_success_inside_chunk_stops_immediately_and_scores_one(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(
        ThreeBenchmarkAdapter, "_video_frame", lambda self, values: None
    )
    task = _task(plus9=True)
    env = FakeEnv(task, {1}, [1.0])
    adapter = ThreeBenchmarkAdapter(tmp_path, env_factory=lambda _: env)
    adapter.start(task, "episode")
    step = adapter.step(_action(0, vectors=5))
    assert step.terminated and env.steps == 1
    assert adapter.finish().score == 1.0


def test_step_does_not_publish_undeclared_native_metadata(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        ThreeBenchmarkAdapter, "_video_frame", lambda self, values: None
    )
    task = _task(plus9=False)
    env = FakeEnv(task, set(), [0.0])
    original_step = env.step

    def step_with_non_json_metadata(action):
        values, reward, terminated, truncated, info = original_step(action)
        info["simulator_internal"] = np.asarray([np.nan])
        return values, reward, terminated, truncated, info

    env.step = step_with_non_json_metadata
    adapter = ThreeBenchmarkAdapter(tmp_path, env_factory=lambda _: env)
    adapter.start(task, "episode")
    result = adapter.step(_action(0))
    assert "simulator_internal" not in result.native
    assert result.native["prefix_progress"] == 0.0


def test_feedback_primitive_compiles_to_budgeted_physical_actions(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(
        ThreeBenchmarkAdapter, "_video_frame", lambda self, values: None
    )
    task = _task(plus9=False)
    env = FakeEnv(task, set(), [0.0, 0.0])
    position = np.zeros(3, dtype=np.float32)

    def servo_step(vector):
        env.steps += 1
        position[:] += np.asarray(vector[:3], dtype=np.float32) * 0.05
        state = np.zeros(8, dtype=np.float32)
        state[:3] = position
        return (
            {"main_images": np.zeros((8, 8, 3), dtype=np.uint8), "states": state},
            0.0,
            False,
            False,
            {"ordered_success": False, "prefix_progress": 0.0},
        )

    env.step = servo_step
    adapter = ThreeBenchmarkAdapter(tmp_path, env_factory=lambda _: env)
    adapter.start(task, "episode")
    result = adapter.step(
        Action(
            benchmark=task.benchmark,
            episode_id="episode",
            step_index=0,
            data={
                "primitive": "move_to",
                "xyz": [0.1, 0.0, 0.0],
                "max_steps": 10,
            },
        )
    )
    assert result.truncated
    assert result.native["action_kind"] == "move_to"
    assert result.native["native_env_steps"] == 2
    # The servo is cut at the immutable two-step official budget.
    assert position.tolist() == pytest.approx([0.05, 0.0, 0.0])
