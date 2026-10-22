# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from roborsi.evaluation.media import CaseMediaChannel
from roborsi.evaluation.messages import Action, Observation, Task


def runtime_model():
    from dataclasses import asdict
    from pathlib import Path

    from scripts.evaluation.configuration import resolve_model

    root = Path(__file__).resolve().parents[4]
    return {
        "backend": "openpi",
        **asdict(resolve_model(root / "configs/models/default.yaml").vla),
    }


def observation(tmp_path, *, enhanced=True):
    media = CaseMediaChannel(tmp_path / "media")
    pixels = np.array(
        [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], dtype=np.uint8
    )
    output = io.BytesIO()
    Image.fromarray(pixels).save(output, format="PNG")
    reference = media.write("episode", 0, output.getvalue(), "image/png")
    data = (
        {"states": [0.0] * 8, "media_fields": {"main_images": 0, "wrist_images": 0}}
        if enhanced
        else {
            "robot0_eef_pos": [0.1, 0.2, 0.3],
            "robot0_eef_quat": [0, 0, 0, 1],
            "robot0_gripper_qpos": [0.04, -0.04],
            "media_fields": {"agentview_image": 0, "robot0_eye_in_hand_image": 0},
        }
    )
    return Observation(
        benchmark="libero_pro" if enhanced else "libero",
        episode_id="episode",
        step_index=0,
        data=data,
        media=(reference,),
    ), pixels


@pytest.mark.parametrize("enhanced", [True, False])
def test_observation_conversion_rotates_only_raw_opengl_images(tmp_path, enhanced):
    from robots.libero.vla_client import observation_payload
    from robots.libero.vla_server import decode_observation

    obs, pixels = observation(tmp_path, enhanced=enhanced)
    payload = observation_payload(obs, tmp_path, "pick the cup")
    decoded = decode_observation(payload)
    np.testing.assert_array_equal(
        decoded["observation/image"], pixels if enhanced else pixels[::-1, ::-1]
    )
    np.testing.assert_array_equal(
        decoded["observation/wrist_image"], pixels if enhanced else pixels[::-1, ::-1]
    )
    assert decoded["prompt"] == "pick the cup"
    np.testing.assert_allclose(
        decoded["observation/state"],
        [0.0] * 8 if enhanced else [0.1, 0.2, 0.3, 0, 0, 0, 0.04, -0.04],
    )


@pytest.mark.parametrize(
    "actions", [[], [[0] * 6], [[float("nan")] * 7], [[0] * 8], [[True] * 7]]
)
def test_invalid_policy_actions_are_rejected(actions):
    from robots.libero.vla_client import validate_actions

    with pytest.raises(ValueError):
        validate_actions(actions, 5)


@pytest.mark.parametrize(
    "benchmark,expected",
    [
        ("libero", {"vector": [0.1] * 7}),
        ("libero_pro", {"vectors": [[0.1] * 7, [0.2] * 7]}),
    ],
)
def test_vla_action_is_translated_to_native_protocol(
    tmp_path, monkeypatch, benchmark, expected
):
    from robots.libero import vla_agent

    calls = []

    class Inner:
        def reset(self, task):
            self.task = task

        def act(self, obs):
            return Action(
                benchmark=obs.benchmark,
                episode_id=obs.episode_id,
                step_index=obs.step_index,
                data={"primitive": "vla", "instruction": "pick cup"},
            )

        def close(self):
            calls.append("close")

    config = SimpleNamespace(
        parameters={
            "vla_endpoint": "http://localhost:1234",
            "vla_model": runtime_model(),
        },
    )
    inner = Inner()
    monkeypatch.setattr(
        vla_agent, "predict", lambda *args, **kwargs: [[0.1] * 7, [0.2] * 7]
    )
    agent = vla_agent.LiberoVlaAgent(inner, config, tmp_path)
    task = Task(
        benchmark=benchmark,
        task_id="pick",
        seed=0,
        instruction="pick",
        max_steps=100,
        capabilities={"action_schema": {"type": "object"}},
    )
    agent.reset(task)
    obs = Observation(benchmark=benchmark, episode_id="episode", step_index=0)
    assert agent.act(obs).data == expected
    assert inner.task.capabilities["action_schema"]["oneOf"][1]["properties"][
        "primitive"
    ] == {"const": "vla"}
    assert (tmp_path / "vla.jsonl").is_file()
    agent.close()
    assert calls == ["close"]


def test_native_action_does_not_call_vla(tmp_path, monkeypatch):
    from robots.libero import vla_agent

    action = Action(
        benchmark="libero",
        episode_id="episode",
        step_index=0,
        data={"vector": [0.0] * 7},
    )
    inner = SimpleNamespace(act=lambda _: action)
    config = SimpleNamespace(
        parameters={
            "vla_endpoint": "http://localhost:1234",
            "vla_model": runtime_model(),
        },
    )
    monkeypatch.setattr(
        vla_agent, "predict", lambda *args: pytest.fail("native action used VLA")
    )
    assert (
        vla_agent.LiberoVlaAgent(inner, config, tmp_path).act(
            Observation(benchmark="libero", episode_id="episode", step_index=0)
        )
        is action
    )


def test_real_http_client_uses_verified_observation_and_bounded_native_actions(
    tmp_path,
):
    import threading

    from robots.libero.vla_client import OpenPiProfile, predict
    from robots.libero.vla_server import make_server

    observed = []

    class Policy:
        def infer(self, data):
            observed.append(data)
            return {"actions": np.array([[0.1] * 7, [0.2] * 7, [0.3] * 7])}

    server = make_server(Policy(), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    obs, pixels = observation(tmp_path)
    try:
        result = predict(
            f"http://127.0.0.1:{server.server_port}",
            OpenPiProfile(
                "a" * 40, "pi05_libero", "gs://fixture/checkpoint", action_steps=2
            ),
            obs,
            tmp_path,
            "pick cup",
        )
        assert result == [[0.1] * 7, [0.2] * 7]
        assert observed[0]["prompt"] == "pick cup"
        np.testing.assert_array_equal(observed[0]["observation/image"], pixels)
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_prepare_only_downloads_without_importing_policy_or_loading_gpu(
    tmp_path, monkeypatch
):
    import sys
    from types import ModuleType

    from scripts.evaluation.downloads import main

    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "assets").mkdir(parents=True)
    (checkpoint / "params").mkdir()
    tokenizer = tmp_path / "tokenizer.model"
    tokenizer.write_bytes(b"fixture-tokenizer")
    calls = []

    def download(url, **kwargs):
        calls.append((url, kwargs))
        return tokenizer if url.endswith(".model") else checkpoint

    package = ModuleType("openpi")
    package.__path__ = []
    shared = ModuleType("openpi.shared")
    shared.download = SimpleNamespace(maybe_download=download)
    monkeypatch.setitem(sys.modules, "openpi", package)
    monkeypatch.setitem(sys.modules, "openpi.shared", shared)
    assert (
        main(
            [
                "openpi",
                "--checkpoint",
                "gs://openpi-assets/checkpoints/pi05_libero",
            ]
        )
        == 0
    )
    assert calls == [
        ("gs://openpi-assets/checkpoints/pi05_libero", {"gs": {"token": "anon"}}),
        ("gs://big_vision/paligemma_tokenizer.model", {"gs": {"token": "anon"}}),
    ]


def test_real_standard_factory_enables_vla_without_calling_main_model(
    tmp_path, monkeypatch
):
    import json

    from roborsi.agents.config import AgentConfig
    from robots.libero.vla_agent import ROOT, LiberoVlaAgent

    config = AgentConfig.from_file(
        ROOT / "configs/agents/libero/standard.yaml",
        overrides=[
            "parameters.enable_vla=true",
            'parameters.vla_endpoint="http://127.0.0.1:32100"',
            "parameters.vla_model=" + json.dumps(runtime_model()),
        ],
    )
    agent = config.create_agent(tmp_path)
    assert isinstance(agent, LiberoVlaAgent)
    assert "VLA is enabled for this run" in agent.inner.system_prompt
    monkeypatch.setenv("ROBORSI_MEDIA_DIR", str(tmp_path / "media"))
    task = Task(
        benchmark="libero",
        task_id="pick",
        seed=0,
        instruction="pick cup",
        max_steps=5,
        capabilities={
            "action_schema": {
                "type": "object",
                "properties": {"vector": {"type": "array"}},
                "required": ["vector"],
                "additionalProperties": False,
            }
        },
    )
    agent.reset(task)
    agent.close()
