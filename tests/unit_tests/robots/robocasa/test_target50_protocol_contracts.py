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

"""Frozen RoboCasa365 task matrix, rollout limits and process setup."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from robots.robocasa import runtime

REPO_ROOT = Path(__file__).resolve().parents[4]

MANIFEST_PATH = REPO_ROOT / "robots" / "robocasa" / "eval" / "target50.json"

EXPECTED_SPLIT_SHAPES = {
    "atomic": {
        "task_count": 18,
        "seeds": list(range(1, 11)),
        "timeout_seconds": 1800,
        "cell_count": 180,
    },
    "composite_seen": {
        "task_count": 16,
        "seeds": list(range(1, 6)),
        "timeout_seconds": 3600,
        "cell_count": 80,
    },
    "composite_unseen": {
        "task_count": 16,
        "seeds": list(range(1, 6)),
        "timeout_seconds": 3600,
        "cell_count": 80,
    },
}


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_target50_matrix_is_exact_and_has_no_duplicate_cells():
    manifest = _manifest()
    splits = manifest["splits"]

    assert set(splits) == set(EXPECTED_SPLIT_SHAPES)

    tasks: list[str] = []
    cells: list[tuple[str, str, int]] = []
    for split_name, expected_shape in EXPECTED_SPLIT_SHAPES.items():
        split = splits[split_name]
        assert split["task_count"] == expected_shape["task_count"]
        assert split["seeds"] == expected_shape["seeds"]
        assert split["timeout_seconds"] == expected_shape["timeout_seconds"]
        assert split["cell_count"] == expected_shape["cell_count"]
        assert len(split["tasks"]) == split["task_count"]
        assert len(split["tasks"]) == len(set(split["tasks"]))
        assert split["cell_count"] == len(split["tasks"]) * len(split["seeds"])
        tasks.extend(split["tasks"])
        cells.extend(
            (split_name, task, seed)
            for task in split["tasks"]
            for seed in split["seeds"]
        )

    assert len(tasks) == len(set(tasks)) == manifest["total_tasks"] == 50
    assert len(cells) == len(set(cells)) == manifest["total_cells"] == 340
    assert sum(split["cell_count"] for split in splits.values()) == 340


def test_target50_runtime_protocol_is_frozen():
    assert _manifest()["runtime_protocol"] == {
        "scene_seed_source": "cli_seed_identity",
        "use_reset_seed_override": False,
        "result_schema_version": "1.0",
        "result_filename": "result.json",
        "rldx_max_chunks": 40,
        "rldx_settle_patience": 999,
        "rldx_action_steps_per_chunk": 8,
    }


def test_spawned_environment_uses_cli_seed_and_clears_legacy_override(
    tmp_path, monkeypatch
):
    captured = {}

    class FakeDaemon:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def start(self):
            captured["started"] = True

    monkeypatch.setattr(runtime, "ProcessDaemon", FakeDaemon)
    monkeypatch.setattr(runtime, "pick_free_port", lambda: 43210)
    monkeypatch.setattr(runtime, "HttpRpcClient", lambda url: url)
    args = SimpleNamespace(
        env_endpoint=None,
        task_name="OpenDrawer",
        split="target",
        seed=7,
        cuda_device=0,
    )

    daemon, client = runtime._spawn_env_server(args, tmp_path)

    seed_index = captured["cmd"].index("--seed")
    assert captured["cmd"][seed_index + 1] == "7"
    assert captured["env_overrides"]["RLDX_RESET_SEED"] == ""
    assert captured["started"] is True
    assert daemon is not None
    assert client == "http://127.0.0.1:43210"


@pytest.mark.parametrize(
    ("launcher_gl", "server_gl"), [(None, "egl"), ("osmesa", "osmesa")]
)
def test_spawned_environment_inherits_the_launcher_renderer(
    tmp_path, monkeypatch, launcher_gl, server_gl
):
    captured = {}

    class FakeDaemon:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def start(self):
            pass

    if launcher_gl is None:
        monkeypatch.delenv("MUJOCO_GL", raising=False)
    else:
        monkeypatch.setenv("MUJOCO_GL", launcher_gl)
    monkeypatch.setattr(runtime, "ProcessDaemon", FakeDaemon)
    monkeypatch.setattr(runtime, "pick_free_port", lambda: 43210)
    monkeypatch.setattr(runtime, "HttpRpcClient", lambda url: url)
    args = SimpleNamespace(
        env_endpoint=None,
        task_name="OpenDrawer",
        split="target",
        seed=1,
        cuda_device=0,
    )

    runtime._spawn_env_server(args, tmp_path)

    assert captured["env_overrides"]["MUJOCO_GL"] == server_gl
