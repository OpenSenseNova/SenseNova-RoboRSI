# Copyright 2026 The RoboRSI Contributors.
# Licensed under the Apache License, Version 2.0.

"""The deployed preset must reach the policy through the official factory."""

from pathlib import Path

from roborsi.agents.config import AgentConfig
from robots.robodojo.agents.factory import make_agent
from robots.robodojo.agents.submission.runtime_config import resolve_vla_config


def test_simate_sampling_preset_reaches_policy_without_starting_model(
    tmp_path, monkeypatch
):
    root = Path(__file__).resolve().parents[4]
    config = AgentConfig.from_file(root / "robots/robodojo/agent-cci.json")
    monkeypatch.setenv("ROBORSI_MEDIA_DIR", str(tmp_path / "media"))
    agent = make_agent(tmp_path / "case", config=config)
    assert agent.model is None
    assert agent.parameters["provider_kind"] == "cci"
    assert agent.parameters["reasoning_effort"] == "medium"
    assert agent.parameters["sampling"] == {
        "temperature": 1.0,
        "top_k": 20,
        "top_p": 0.95,
    }
    assert resolve_vla_config(agent.parameters) == {
        "backend": "simate_beta",
        "checkpoint_path": "checkpoints/simate-beta-robodojo-eval",
        "setting": "simate_vla",
    }
