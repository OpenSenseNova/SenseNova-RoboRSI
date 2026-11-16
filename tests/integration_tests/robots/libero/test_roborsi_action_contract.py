# Copyright 2026 The SenseNova-RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check LIBERO task identity, native clipping and rejected actions over HTTP."""

import json
import os
import subprocess
import uuid
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from roborsi.benchmarks.registry import BenchmarkRegistry
from roborsi.evaluation.docker_runtime import DockerExecutor
from roborsi.evaluation.messages import Action, Task, from_envelope, to_envelope
from roborsi.evaluation.protocol import InteractionEnvelope
from roborsi.utils.config import get_project_env, require_project_env
from roborsi.utils.rpc import RpcError

pytestmark = [
    pytest.mark.skipif(
        get_project_env("ROBORSI_RUN_LIBERO_BENCHMARK") != "1",
        reason="requires the current LIBERO image and assets",
    ),
    pytest.mark.timeout(300),
]


def _lock():
    lock = BenchmarkRegistry([require_project_env("ROBORSI_LIBERO_REGISTRY")]).lock(
        "libero"
    )
    assert lock.task_message("0", 0).max_steps == 3
    return lock


def _start(client, task, episode):
    body = to_envelope(
        task, benchmark="libero", episode_id=episode, request_id="start"
    ).as_dict()
    _, wire = client.transport.request("POST", "/v1/episodes", body=body, timeout_s=120)
    return from_envelope(InteractionEnvelope.from_dict(wire))


def _step(client, episode, index, data):
    body = to_envelope(
        Action(benchmark="libero", episode_id=episode, step_index=index, data=data),
        benchmark="libero",
        episode_id=episode,
        request_id=f"step-{index}",
    ).as_dict()
    _, wire = client.transport.request(
        "POST", f"/v1/episodes/{episode}/actions", body=body, timeout_s=30
    )
    return from_envelope(InteractionEnvelope.from_dict(wire))


def _removed(container):
    assert not subprocess.check_output(
        ["docker", "ps", "-aq", "--filter", f"id={container.container_id}"],
        text=True,
        timeout=10,
    ).strip()


def _reference(lock, task, actions):
    asset = lock.as_dict()["assets"][0]
    assets = (lock.manifest_path.parent / asset["path"]).resolve()
    name = f"roborsi-action-reference-{uuid.uuid4().hex}"
    try:
        process = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--name",
                name,
                "--read-only",
                "--network",
                "none",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--cpus",
                "2",
                "--memory",
                "4096m",
                "--tmpfs",
                "/tmp:rw,nosuid,nodev,size=1g",
                "--mount",
                f"type=bind,src={assets},dst=/roborsi/assets/libero,readonly",
                "--mount",
                f"type=bind,src={Path(__file__).parent},dst=/checks,readonly",
                lock.image_reference,
                "python",
                "/checks/native_reference.py",
                "--suite",
                task.native["suite"],
                "--task",
                task.task_id,
                "--seed",
                str(task.seed),
                "--max-steps",
                str(task.max_steps),
                "--actions",
                json.dumps(actions),
            ],
            check=True,
            timeout=120,
            capture_output=True,
            text=True,
        )
    finally:
        subprocess.run(
            ["docker", "rm", "--force", name], timeout=10, capture_output=True
        )
    return json.loads(
        next(
            line.removeprefix("ROBORSI_NATIVE_REFERENCE=")
            for line in process.stdout.splitlines()
            if line.startswith("ROBORSI_NATIVE_REFERENCE=")
        )
    )


def test_native_action_limits_clipping_and_task_seed_match_http(tmp_path):
    lock = _lock()
    task = lock.task_message("1", 0).model_copy(update={"seed": 7})
    actions = [
        [2.0, -2.0, 0.0, 2.0, -2.0, 0.0, -2.0],
        [-1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 1.0],
        [0.0] * 7,
    ]
    states = []
    with DockerExecutor().start_adapter(lock, tmp_path) as container:
        client = container.client
        catalog = client.tasks()
        assert catalog[1].capabilities == task.capabilities
        started = _start(client, task, "native-actions")
        assert started.task == task
        assert started.capabilities["reset_state_id"] is not None
        validator = Draft202012Validator(started.capabilities["action_schema"])
        for index, vector in enumerate(actions):
            validator.validate({"vector": vector})
            response = _step(client, "native-actions", index, {"vector": vector})
            states.append(response.observation.data["states"])
        assert response.truncated and not response.terminated
        before = client.status("native-actions")
        with pytest.raises(RpcError, match="ended"):
            _step(client, "native-actions", 3, {"vector": [0.0] * 7})
        assert client.status("native-actions") == before
        _, wire = client.transport.request(
            "POST", "/v1/episodes/native-actions/finish", body={"request_id": "finish"}
        )
        result = from_envelope(InteractionEnvelope.from_dict(wire))
        for info in client.artifacts(episode_id="native-actions"):
            client.download(info, tmp_path / "download")
        assert (
            json.loads((tmp_path / "download/native/score.json").read_text())["steps"]
            == 3
        )
    _removed(container)
    # Run the same seed directly in LIBERO, with out-of-range inputs preclipped.
    native = _reference(lock, task, np.clip(actions, -1, 1).tolist())
    assert native["control"] == {
        "name": "OSC_POSE",
        "input_type": "delta",
        "input_ref_frame": "base",
        "input_min": [-1.0] * 6,
        "input_max": [1.0] * 6,
        "output_min": [-0.05] * 3 + [-0.5] * 3,
        "output_max": [0.05] * 3 + [0.5] * 3,
        "frequency_hz": 20,
        "action_low": [-1.0] * 7,
        "action_high": [1.0] * 7,
    }
    np.testing.assert_allclose(states, native["states"], rtol=1e-6, atol=1e-6)
    assert result.success == native["success"] and result.score == native["score"]
    assert result.details["last_reward"] == native["last_reward"]
    (tmp_path / "native-comparison.json").write_text(json.dumps(native, indent=2))


def test_inconsistent_task_fields_fail_without_observations_or_scores(tmp_path):
    lock = _lock()
    task = lock.task_message("0", 0)
    bad = [
        {"task_id": "00"},
        {"task_id": "999"},
        {"instruction": "a different instruction"},
        {"native": {"suite": "does_not_exist"}},
        {"native": {"suite": "libero_spatial", "ignored_setting": True}},
        {"capabilities": {"action_schema": {"type": "object"}}},
    ]
    with DockerExecutor().start_adapter(lock, tmp_path) as container:
        client = container.client
        for index, updates in enumerate(bad):
            episode = f"bad-task-{index}"
            changed = Task.from_dict({**task.as_dict(), **updates})
            with pytest.raises(RpcError):
                _start(client, changed, episode)
            assert client.status(episode)["state"] == "failed"
            assert client.status(episode)["step_index"] == 0
            code, reply = client.result(episode)
            assert code == 202 and reply["error"]
            assert all(a.kind == "log" for a in client.artifacts(episode_id=episode))
            client.delete(episode)
    _removed(container)


@pytest.mark.parametrize(
    "data",
    [
        {"vector": [0.0] * 6},
        {"vector": [False] * 7},
        {"vector": ["0"] * 7},
        {"vector": [0.0] * 7, "score": 1},
    ],
)
def test_illegal_actions_fail_before_native_step_and_never_produce_score(
    tmp_path, data
):
    lock = _lock()
    with DockerExecutor().start_adapter(lock, tmp_path) as container:
        client = container.client
        _start(client, lock.task_message("0", 0), "bad-action")
        with pytest.raises(RpcError):
            _step(client, "bad-action", 0, data)
        status = client.status("bad-action")
        assert status["state"] == "failed" and status["step_index"] == 0
        assert client.result("bad-action")[0] == 202
        with pytest.raises(RpcError):
            client.transport.request(
                "POST", "/v1/episodes/bad-action/finish", body={"request_id": "finish"}
            )
        artifacts = client.artifacts(episode_id="bad-action")
        assert not any(a.kind == "native_result" for a in artifacts)
        assert len([a for a in artifacts if a.kind == "image"]) == 2
        client.delete("bad-action")
    _removed(container)
