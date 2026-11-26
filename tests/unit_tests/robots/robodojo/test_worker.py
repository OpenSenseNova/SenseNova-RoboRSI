# Copyright 2026 The RoboRSI Contributors.
# Licensed under the Apache License, Version 2.0.

"""Run the native worker protocol against a deterministic physical owner."""

import hashlib
import json
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from roborsi.evaluation.messages import Action, NativeResult, Task


def envelope():
    from robots.robodojo.native_curobo import envelope

    return envelope(
        1,
        0,
        {
            "mode": "eef",
            "target": {},
            "reason": "check",
            "assessment": {"current_subgoal": "hold"},
        },
    )


def test_action_schema_accepts_submitted_curobo_envelope():
    from robots.robodojo.roborsi_adapter import ACTION_SCHEMA

    Draft202012Validator(ACTION_SCHEMA).validate(envelope())


@pytest.mark.parametrize("bounded", [False, True])
def test_worker_preserves_one_native_step_and_saved_score(tmp_path, bounded):
    from robots.robodojo.roborsi_adapter import ACTION_SCHEMA
    from robots.robodojo.worker import EpisodeWorker

    task = Task(
        benchmark="robodojo",
        task_id="build_tower",
        seed=3,
        instruction="Build the tower",
        max_steps=1,
        native={
            "config": "arx_x5",
            "eval_seed": 0,
            "native_step_limit": 1,
            "layout_sha256": "a" * 64,
        },
        capabilities={
            "action_schema": ACTION_SCHEMA,
            "observation_channels": ["cam_head"],
        },
    )

    class Environment:
        take_action_cnt = [0]
        end_flag = [False]
        env_seeds = [3]
        success = [True]
        unstable_envs = []

        def get_obs(self):
            return {
                "instruction": task.instruction,
                "state": {},
                "vision": {"cam_head": {"color": np.zeros((2, 3, 3), dtype=np.uint8)}},
            }

        def take_action(self, action):
            self.take_action_cnt = [1]
            self.end_flag = [True]

    env = Environment()
    native_step_calls = []

    class Runtime:
        def run(self, spec, interact):
            interact(env, task, env.get_obs())
            source = tmp_path / "native/_result.json"
            source.parent.mkdir(exist_ok=True)
            source.write_text(
                json.dumps(
                    {
                        "eval_time": 1,
                        "details": {
                            "case": {"layout_id": 3, "score": 0.37, "success": True}
                        },
                    }
                )
            )
            return {"source": str(source)}

        def step(self, environment, action, *, initial_observation):
            assert "vision" in initial_observation
            native_step_calls.append(action)
            environment.take_action(action)
            raw = environment.get_obs()
            raw["roborsi_execution"] = {"request_id": 1}
            return raw

    worker = EpisodeWorker(tmp_path, Runtime())
    data = (
        envelope()
        if bounded
        else {
            f"{arm}_{name}": [0.0] * length
            for arm in ("left", "right")
            for name, length in (("arm_joint_state", 6), ("ee_joint_state", 1))
        }
    )
    action = Action(benchmark="robodojo", episode_id="case", step_index=0, data=data)
    stepped, finished, initial = Future(), Future(), Future()
    worker.requests.put(("step", action.as_dict(), stepped))
    worker.requests.put(("finish", {}, finished))
    worker._episode("start", {"task": task.as_dict(), "episode_id": "case"}, initial)
    assert initial.result(timeout=1)["observation"]["step_index"] == 0
    result = stepped.result(timeout=1)
    assert result["observation"]["step_index"] == 1
    assert env.take_action_cnt == [1]
    if bounded:
        assert native_step_calls == [data]
        assert result["observation"]["data"]["roborsi_execution"] == {"request_id": 1}
    native = NativeResult.from_dict(finished.result(timeout=1))
    assert native.score == 0.37 and native.success
    assert (
        native.source_sha256
        == hashlib.sha256((tmp_path / native.source).read_bytes()).hexdigest()
    )


def test_native_curobo_dispatch_uses_existing_joint_executor(monkeypatch):
    from robots.robodojo import native_curobo
    from robots.robodojo.native import NativeRuntime

    calls = []
    reads = []
    initial = {"state": {"measured": True}}

    def get_obs():
        reads.append(True)
        return {"state": {"after_step": True}}

    env = SimpleNamespace(
        get_obs=get_obs,
        take_action=lambda data: calls.append(data),
    )

    def execute(environment, request, *, step_joint, initial_observation, **kwargs):
        assert initial_observation is initial
        result = step_joint({"actual_joint": [0.0]})
        result["roborsi_execution"] = {"request_id": request["request_id"]}
        return result

    monkeypatch.setattr(native_curobo, "execute_control", execute)
    result = NativeRuntime().step(env, envelope(), initial_observation=initial)
    assert calls == [{"actual_joint": [0.0]}]
    assert result["roborsi_execution"] == {"request_id": 1}
    assert reads == [True]
