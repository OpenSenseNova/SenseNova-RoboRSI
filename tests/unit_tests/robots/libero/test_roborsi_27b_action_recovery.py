# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Invalid model calls must get feedback without entering the simulator."""

import copy
import json
from types import SimpleNamespace

import pytest
from jsonschema import ValidationError

from roborsi.evaluation.embodied_bridge import (
    _EmbodiedToolConnector,
    roborsi_action_tool,
)
from roborsi.evaluation.messages import Observation, StepResult, Task
from robots.libero.image_tools import _expose_action_property_types, action_schema
from robots.libero.roborsi_27b_agent import _enable_action_recovery, _validation_retries

PAIRS = [
    (
        {"primitive": "vector", "vector": [0, 0, 0, -0.787, 0, 0, 1]},
        {"vector": [0, 0, 0, -0.787, 0, 0, 1]},
    ),
    ({"vector": [0] * 7, "vectors": [[0] * 7] * 2}, {"vectors": [[0] * 7] * 2}),
    ({"primitive": "release", "steps": 20}, {"primitive": "release", "max_steps": 20}),
]


def task():
    return Task(
        benchmark="libero_pro",
        task_id="test",
        seed=0,
        instruction="Move the object.",
        max_steps=2000,
        capabilities={
            "action_schema": _expose_action_property_types(action_schema(False))
        },
    )


def observation(step=0):
    return Observation(
        benchmark="libero_pro",
        episode_id="e",
        step_index=step,
        data={"native_env_steps": step * 20},
    )


class Episode:
    def __init__(self, calls):
        self.calls = iter(
            SimpleNamespace(name="roborsi_action", arguments=x) for x in calls
        )
        self.feedback = []
        self.closed = False

    def next_call(self, **kwargs):
        return next(self.calls)

    def complete(self, call, result):
        self.feedback.append((call, result))

    def wait(self, **kwargs):
        return SimpleNamespace(error=None)

    def close(self, **kwargs):
        self.closed = True


def connector(tmp_path, calls, retries=3):
    episode = Episode(calls)
    t = task()
    planner = SimpleNamespace(
        local_tools=[roborsi_action_tool(t.capabilities["action_schema"])],
        start_episode=lambda *a, **kw: episode,
    )
    c = _EmbodiedToolConnector(
        planner,
        media_dir=tmp_path / "media",
        output_dir=tmp_path / "agent",
        system_prompt="Follow the action schema.",
    )
    c.reset(t)
    _enable_action_recovery(SimpleNamespace(_evaluation_connector=c), retries)
    return c, episode


@pytest.mark.parametrize("bad,good", PAIRS)
def test_rejected_action_not_forwarded_and_lifecycle_preserved(tmp_path, bad, good):
    original = copy.deepcopy(bad)
    c, e = connector(tmp_path, [bad, good, bad, good])
    try:
        action = c.act(observation())
        assert action.data == good and action.step_index == 0
        assert bad == original  # Never normalize or rewrite model arguments.
        assert len(e.feedback) == 1
        feedback = e.feedback[0][1].result
        assert feedback["executed"] is False and feedback["physical_steps_used"] == 0
        assert (
            feedback["step_index"] == 0
            and feedback["action_schema"] == task().capabilities["action_schema"]
        )
        assert c._last_step == 0 and c._pending.arguments == good
        # A fresh per-step retry budget; native feedback reaches the accepted call once.
        assert c.act(observation(1)).step_index == 1
        assert e.feedback[1][1].result["observation"]["native_env_steps"] == 20
        c.observe_end(
            StepResult(
                observation=observation(2),
                terminated=True,
                truncated=False,
                termination_reason="success",
            )
        )
        assert e.feedback[-1][1].is_finish
        rows = [
            json.loads(x)
            for x in (tmp_path / "agent/action_recovery.jsonl").read_text().splitlines()
        ]
        assert [r["event"] for r in rows] == [
            "rejected",
            "accepted",
            "rejected",
            "accepted",
        ]
        assert [r["step_index"] for r in rows] == [0, 0, 1, 1]
    finally:
        c.close()
    assert e.closed


def test_exhaustion_allows_three_corrections_not_four(tmp_path):
    bad = PAIRS[2][0]
    c, e = connector(tmp_path, [bad] * 4 + [PAIRS[2][1]])
    try:
        with pytest.raises(ValidationError):
            c.act(observation())
        assert len(e.feedback) == 3
        assert c._last_step == -1 and c._pending is None
        rows = [
            json.loads(x)
            for x in (tmp_path / "agent/action_recovery.jsonl").read_text().splitlines()
        ]
        assert [r["event"] for r in rows] == ["rejected"] * 3 + ["exhausted"]
    finally:
        c.close()


def test_disabled_default_still_fails_without_feedback(tmp_path):
    c, e = connector(tmp_path, [PAIRS[0][0]], retries=0)
    try:
        with pytest.raises(ValidationError):
            c.act(observation())
        assert not e.feedback
        assert not (tmp_path / "agent/action_recovery.jsonl").exists()
    finally:
        c.close()


def test_transport_failure_is_not_retried(tmp_path):
    c, e = connector(tmp_path, [])

    def fail(**kw):
        raise TimeoutError("transport timeout")

    e.next_call = fail
    try:
        with pytest.raises(TimeoutError):
            c.act(observation())
        assert not e.feedback
    finally:
        c.close()


@pytest.mark.parametrize("value", [True, -1, 4, "3", None, 1.5])
def test_invalid_config_rejected(value):
    with pytest.raises(ValueError):
        _validation_retries({"action_validation_max_retries": value})


@pytest.mark.parametrize("bad,good", PAIRS)
def test_real_dialogue_receives_error_and_self_corrects(
    tmp_path, monkeypatch, bad, good
):
    from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart
    from pydantic_ai.models.function import FunctionModel

    from roborsi.embodied_agent import EmbodiedAgent
    from roborsi.llm import LLMConfig

    responses = []

    def model(messages, info):
        returns = [
            p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)
        ]
        if returns:
            content = str(returns[-1].content)
            assert "invalid_action" in content and "executed" in content
            assert "environment is unchanged" in content
        responses.append(len(returns))
        return ModelResponse(
            parts=[ToolCallPart("roborsi_action", good if returns else bad)]
        )

    monkeypatch.setattr(LLMConfig, "build_model", lambda self: FunctionModel(model))
    monkeypatch.setenv("ROBORSI_MEDIA_DIR", str(tmp_path / "media"))
    a = EmbodiedAgent(
        mcp_servers=[],
        local_tools=[],
        output_dir=tmp_path / "agent",
        system_prompt="Follow the action schema.",
        llm=LLMConfig("openai", "offline", api_key="test"),
        max_turns=5,
    )
    a.reset(task())
    _enable_action_recovery(a, 3)
    try:
        action = a.act(observation())
        assert action.data == good
        assert responses == [0, 1]
    finally:
        a.close()
