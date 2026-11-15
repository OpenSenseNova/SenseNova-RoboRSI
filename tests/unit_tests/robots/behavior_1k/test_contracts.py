# Copyright 2026 The SenseNova-RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Offline native lifecycle, public sensors, scoring and manifest checks."""

import json
from pathlib import Path

import numpy as np
import pytest

from roborsi.evaluation.messages import Action, EpisodeStop
from robots.behavior_1k.catalog import CHANNELS, task_from_metadata
from robots.behavior_1k.episode import BehaviorEpisode


class Native:
    def __init__(self):
        self.steps = 0
        self.closes = 0
        self.fail = False
        self.score_error = False
        self.actions = []

    def start(self, **kwargs):
        self.arguments = kwargs
        return {
            "task_name": "turning_on_radio",
            "mode": "public_test",
            "instance_id": 301,
            "max_steps": 2,
            "task_instruction": "Turn on the radio.",
            "action_dim": 21,
            "action_profile": "r1pro_eef21_v1",
        }

    def observe(self):
        return {
            **{name: np.zeros((2, 3, 3), dtype=np.uint8) for name in CHANNELS},
            "step_index": self.steps,
            "proprio": {"trunk_qpos": [0] * 4},
            "action_bounds": {"low": [-1] * 21, "high": [1] * 21},
            "privileged_object_positions": [10, 20, 30],
        }

    def step(self, action):
        if self.fail:
            return {"ok": False, "error": "native failure"}
        self.steps += 1
        self.actions.append(action)
        return {
            "ok": True,
            "step_index": self.steps,
            "terminated": False,
            "truncated": self.steps >= 2,
        }

    def evaluate(self):
        return {
            "success": False,
            "q_score": 0.375,
            "steps": self.steps,
            "metrics_errors": {"TaskMetric": "bad"} if self.score_error else {},
            "invalid": False,
            "infrastructure_failure": False,
        }

    def close(self):
        self.closes += 1


@pytest.fixture
def episode(tmp_path):
    native = Native()
    episode = BehaviorEpisode(tmp_path, native, tmp_path, tmp_path)
    task = task_from_metadata(native.start())
    episode.start(task, "test")
    yield episode
    episode.close()


def action(values=None, step=0, episode_id="test"):
    return Action(
        benchmark="behavior_1k",
        episode_id=episode_id,
        step_index=step,
        data={"action": [0.25] * 21 if values is None else values},
    )


def test_native_actions_public_images_and_score(episode):
    assert "privileged_object_positions" not in episode.last_observation.data
    assert len(episode.last_observation.media) == 6
    assert episode.runtime.arguments["max_steps"] is None
    assert not episode.step(action()).truncated
    assert episode.step(action(step=1)).truncated
    result = episode.finish()
    assert result.score == 0.375 and result.success is False
    assert episode.runtime.actions == [[0.25] * 21] * 2
    assert json.loads((episode.directory / result.source).read_text())["steps"] == 2
    assert episode.finish() == result
    with pytest.raises(ValueError):
        episode.step(action(step=2))
    episode.close()
    episode.close()
    assert episode.runtime.closes == 1


@pytest.mark.parametrize(
    "values", [[0] * 23, [True] * 21, [float("nan")] * 21, [2] * 21]
)
def test_reject_invalid_actions_without_native_step(episode, values):
    with pytest.raises(ValueError):
        episode.step(action(values))
    assert episode.runtime.steps == 0


def test_wrong_identity_and_stop_does_not_step(episode):
    with pytest.raises(ValueError):
        episode.step(action(episode_id="other"))
    stopped = episode.stop(
        EpisodeStop(
            benchmark="behavior_1k",
            episode_id="test",
            step_index=0,
            reason="agent_finished",
        )
    )
    assert stopped.truncated and stopped.observation.step_index == 0
    assert episode.finish().score == 0.375
    assert episode.runtime.actions == []


@pytest.mark.parametrize("steps", [0, 1])
def test_stop_uses_native_task_metric_without_adding_a_step(
    episode, monkeypatch, steps
):
    import sys
    from types import SimpleNamespace

    from robots.behavior_1k.native import Behavior1KEpisode

    class TaskMetric:
        # The pinned MetricBase.aggregate skips empty reset state:
        # https://github.com/StanfordVL/BEHAVIOR-1K/blob/b1979916ec1549b10a4e65e630bc6504a9af1b00/OmniGibson/omnigibson/metrics/metric_base.py#L85-L101
        def __init__(self):
            self.timesteps = steps
            self.state = {"scene": {"timesteps": [steps]} if steps else {}}
            self.aggregate_calls = 0
            self.compute_calls = 0

        def aggregate(self, env):
            self.aggregate_calls += 1
            state = self.state[env.scene]
            return self._compute_episode_metrics(env, state) if state else {}

        def _compute_episode_metrics(self, env, episode_info):
            # TaskMetric's final calculation accepts empty episode_info and
            # falls back to its own reset timesteps; it does not step physics.
            # https://github.com/StanfordVL/BEHAVIOR-1K/blob/b1979916ec1549b10a4e65e630bc6504a9af1b00/OmniGibson/omnigibson/metrics/task_metric.py#L28-L51
            self.compute_calls += 1
            timesteps = (
                episode_info["timesteps"][-1]
                if episode_info.get("timesteps")
                else self.timesteps
            )
            return {
                "q_score": {"final": 1.0 if env.task.success else 0.0},
                "time": {"simulator_steps": timesteps},
            }

    monkeypatch.setitem(
        sys.modules, "omnigibson.metrics", SimpleNamespace(TaskMetric=TaskMetric)
    )
    metric = TaskMetric()
    runtime = Behavior1KEpisode()
    runtime.evaluator = SimpleNamespace(
        metrics=[metric],
        env=SimpleNamespace(scene="scene", task=SimpleNamespace(success=True)),
        n_trials=0,
        n_success_trials=0,
    )
    if steps:
        episode.step(action())
    runtime.steps = steps
    episode.runtime.evaluate = runtime.evaluate
    episode.stop(
        EpisodeStop(
            benchmark="behavior_1k",
            episode_id="test",
            step_index=steps,
            reason="agent_finished",
        )
    )
    result = episode.finish()
    assert result.success and result.score == 1.0
    assert result.details["steps"] == steps
    assert result.details["metrics"]["time"]["simulator_steps"] == steps
    assert len(episode.runtime.actions) == steps
    assert metric.timesteps == steps
    assert metric.compute_calls == 1
    if steps:
        assert metric.aggregate_calls == 1


def test_native_failure_cannot_be_scored_or_retried(episode):
    episode.runtime.fail = True
    with pytest.raises(RuntimeError):
        episode.step(action())
    with pytest.raises(ValueError):
        episode.step(action())
    with pytest.raises(ValueError):
        episode.finish()


def test_native_metric_failure_is_not_zero_score(episode):
    episode.stop(
        EpisodeStop(
            benchmark="behavior_1k",
            episode_id="test",
            step_index=0,
            reason="agent_finished",
        )
    )
    episode.runtime.score_error = True
    with pytest.raises(RuntimeError, match="scoring"):
        episode.finish()


def test_instruction_mismatch_rejected(tmp_path):
    native = Native()
    task = task_from_metadata({**native.start(), "task_instruction": "Another task"})
    episode = BehaviorEpisode(tmp_path, native, tmp_path, tmp_path)
    with pytest.raises(ValueError, match="differs"):
        episode.start(task, "test")
    episode.close()


def test_worker_process_releases_after_native_steps(tmp_path):
    from robots.behavior_1k.roborsi_adapter import BehaviorRoboRSIAdapter

    adapter = BehaviorRoboRSIAdapter(
        tmp_path,
        factory="tests.unit_tests.robots.behavior_1k.test_contracts:Native",
        operation_timeout_s=10,
    )
    try:
        task = task_from_metadata(Native().start())
        adapter.start(task, "test")
        adapter.step(action())
        adapter.step(action(step=1))
        assert adapter.finish().score == 0.375
    finally:
        adapter.close()
    assert adapter.daemon.poll() is not None


def test_public_http_observations_score_match_direct_native(tmp_path):
    import threading

    from roborsi.evaluation.benchmark_client import BenchmarkClient
    from roborsi.evaluation.benchmark_http import (
        BenchmarkApplication,
        BenchmarkHttpServer,
    )
    from roborsi.evaluation.media import CaseMediaChannel
    from robots.behavior_1k.benchmark_service import BehaviorBenchmark

    task = task_from_metadata(Native().start())
    native = Native()

    class Provider(BehaviorBenchmark):
        def __init__(self):
            pass

        def tasks(self):
            return [task]

        def assets(self):
            return []

        def runtime_info(self):
            return None

        def create_adapter(self, directory, artifacts):
            return BehaviorEpisode(directory, native, tmp_path, tmp_path)

    app = BenchmarkApplication(Provider(), tmp_path / "service")
    server = BenchmarkHttpServer(("127.0.0.1", 0), app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = BenchmarkClient(f"http://127.0.0.1:{server.server_port}")
    try:
        assert client.tasks() == [task]
        start = client.start(task, "test", timeout_s=5)
        media = CaseMediaChannel(tmp_path / "runner/media")
        client.collect_observation(start.observation, media, timeout_s=5)
        for i in range(2):
            result = client.step(action(step=i), timeout_s=5)
            client.collect_observation(result.observation, media, timeout_s=5)
        score = client.finish(timeout_s=5)
        client.collect_native(score, tmp_path / "runner", timeout_s=5)
        direct = Native()
        direct.start()
        direct.step([0.25] * 21)
        direct.step([0.25] * 21)
        assert score.score == direct.evaluate()["q_score"]
        assert score.details["steps"] == direct.steps
        assert (
            json.loads((tmp_path / "runner/native/score.json").read_text())["steps"]
            == 2
        )
        client.close()
        assert native.closes == 1
    finally:
        app.close()
        server.shutdown()
        server.server_close()
        thread.join(3)


def test_manifest_is_lockable_and_rejects_external_assets(tmp_path):
    from benchmarks.behavior_1k.prepare import build_manifest
    from roborsi.benchmarks.registry import BenchmarkRegistry, directory_sha256
    from roborsi.evaluation.messages import BenchmarkRuntimeInfo
    from robots.behavior_1k.catalog import EVALUATION_RULES, NATIVE_COMMIT

    manifest_dir = tmp_path / "registry/behavior_1k"
    assets = manifest_dir / "assets"
    catalog = manifest_dir / "catalog"
    assets.mkdir(parents=True)
    catalog.mkdir()
    (assets / "asset.bin").write_bytes(b"native asset fixture")
    (catalog / "catalog.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "native_commit": NATIVE_COMMIT,
                "assets_sha256": directory_sha256(assets),
                "tasks": [task_from_metadata(Native().start()).as_dict()],
            }
        )
    )
    runtime = BenchmarkRuntimeInfo(
        benchmark="behavior_1k",
        benchmark_version="3.9.2",
        benchmark_commit=NATIVE_COMMIT,
        adapter_version="test-fixture",
        evaluation=EVALUATION_RULES,
        platform={"test": "offline"},
    )
    doc = build_manifest(
        "example.invalid/behavior@sha256:" + "a" * 64,
        assets,
        catalog,
        runtime,
        manifest_dir,
    )
    (manifest_dir / "manifest.json").write_text(json.dumps(doc))
    lock = BenchmarkRegistry([tmp_path / "registry"]).lock("behavior_1k")
    assert lock.list_tasks()[0].task_id == "turning_on_radio"
    assert all(not Path(item["path"]).is_absolute() for item in doc["assets"])
    with pytest.raises(ValueError):
        build_manifest(
            "example.invalid/behavior@sha256:" + "a" * 64,
            assets,
            catalog,
            runtime,
            tmp_path / "elsewhere",
        )


@pytest.mark.parametrize("kind", ["leaf", "parent", "manifest_parent", "traversal"])
def test_manifest_rejects_unsafe_asset_paths(tmp_path, kind):
    from benchmarks.behavior_1k.prepare import _asset_subpath

    manifest_dir = tmp_path / "registry/behavior_1k"
    assets = manifest_dir / "snapshots/assets"
    assets.mkdir(parents=True)
    if kind == "leaf":
        candidate = manifest_dir / "linked"
        candidate.symlink_to(assets, target_is_directory=True)
    elif kind == "parent":
        (manifest_dir / "linked").symlink_to(assets.parent, target_is_directory=True)
        candidate = manifest_dir / "linked/assets"
    elif kind == "manifest_parent":
        (tmp_path / "linked").symlink_to(manifest_dir.parent, target_is_directory=True)
        manifest_dir = tmp_path / "linked/behavior_1k"
        candidate = manifest_dir / "snapshots/assets"
    else:
        candidate = manifest_dir / "snapshots/../snapshots/assets"
    with pytest.raises(ValueError, match="symlink|traversal"):
        _asset_subpath(candidate, manifest_dir)


def test_official_21d_config_preserves_input_and_replaces_joint_controllers():
    import copy

    from robots.behavior_1k.official import robot_config_for_profile

    source = {
        "action_normalize": True,
        "controller_config": {
            "base": {"name": "JointController", "mode": "velocity"},
            "arm_left": {"name": "JointController", "motor_type": "position"},
            "arm_right": {"name": "JointController", "motor_type": "position"},
        },
    }
    original = copy.deepcopy(source)
    result = robot_config_for_profile(source)
    assert source == original
    assert result["action_normalize"] is False
    assert result["controller_config"]["base"] == source["controller_config"]["base"]
    for arm in ("left", "right"):
        assert result["controller_config"][f"arm_{arm}"] == {
            "name": "InverseKinematicsController",
            "mode": "absolute_pose",
            "command_input_limits": None,
            "command_output_limits": None,
            "use_impedances": False,
            "pos_kp": 150,
            "smoothing_filter_size": 2,
        }
    result["controller_config"]["base"]["mode"] = "changed"
    assert source == original


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"action_profile": "r1pro_joint23_v1"}, "action_profile"),
        ({"include_depth": False}, "RGBDFullResWrapper"),
        ({"include_depth": 1}, "include_depth must be boolean"),
    ],
)
def test_native_rejects_nonofficial_controls_before_simulator_start(
    tmp_path, options, error
):
    from robots.behavior_1k.native import Behavior1KEpisode

    native = Behavior1KEpisode()
    with pytest.raises(ValueError, match=error):
        native.start(
            repo_root=str(tmp_path),
            data_root=str(tmp_path),
            task_name="turning_on_radio",
            mode="public_test",
            instance_id=301,
            max_steps=None,
            **options,
        )


@pytest.mark.parametrize("failure", ["daemon", "rpc"])
def test_failed_process_cleanup_can_retry_without_accepting_new_work(failure):
    from robots.behavior_1k.roborsi_adapter import BehaviorRoboRSIAdapter

    class Daemon:
        stops = 0

        def poll(self):
            return 0

        def stop(self, *, timeout):
            self.stops += 1
            if failure == "daemon" and self.stops == 1:
                raise RuntimeError("cleanup failed")

    class Rpc:
        closes = 0

        def call(self, *args, **kwargs):
            raise AssertionError("closed adapter sent new RPC work")

        def close(self):
            self.closes += 1
            if failure == "rpc" and self.closes == 1:
                raise RuntimeError("cleanup failed")

    adapter = BehaviorRoboRSIAdapter.__new__(BehaviorRoboRSIAdapter)
    adapter._closed = False
    adapter._cleanup_complete = False
    adapter.daemon = Daemon()
    adapter._rpc = Rpc()
    with pytest.raises(RuntimeError, match="cleanup failed"):
        adapter.close()
    with pytest.raises(RuntimeError, match="process is closed"):
        adapter.inspect({"task_id": "turning_on_radio"})
    adapter.close()
    adapter.close()
    assert adapter.daemon.stops == 2
    assert adapter._rpc.closes == 2
    with pytest.raises(RuntimeError, match="process is closed"):
        adapter.inspect({"task_id": "turning_on_radio"})
