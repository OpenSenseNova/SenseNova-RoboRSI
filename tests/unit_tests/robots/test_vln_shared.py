# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Behavioral regression tests for shared exploration and conditional reruns."""

from __future__ import annotations

import base64
import fcntl
import sys
import threading
import time

import pytest

from roborsi.agents.config import AgentConfig
from roborsi.agents.package import export_agent
from roborsi.evaluation.benchmark_client import BenchmarkClient
from roborsi.evaluation.media import CaseMediaChannel
from roborsi.evaluation.messages import Action
from robots.vln_shared.assets import manifest
from robots.vln_shared.benchmark import (
    NavigationAdapter,
    NavigationBenchmark,
    make_task,
    serve,
)
from robots.vln_shared.campaign import Campaign, validate_outcome
from robots.vln_shared.cli import make_preset
from robots.vln_shared.execution import BenchExecutor, run_process
from robots.vln_shared.exploration import TrainingTools
from robots.vln_shared.progress import Progress
from robots.vln_shared.storage import (
    freeze_snapshot,
    read_json,
    verify_snapshot,
    write_json,
)


@pytest.fixture
def snapshot(tmp_path):
    workspace = tmp_path / "training"
    workspace.mkdir()
    (workspace / "empty-runlogs").mkdir()
    (workspace / "__pycache__").mkdir()
    (workspace / "__pycache__/old.pyc").write_bytes(b"historical generated cache")
    (workspace / "EXPLORATION.md").write_text(
        "Check each turn against visible geometry."
    )
    root = tmp_path / "snapshot"
    freeze_snapshot(workspace, root)
    return root


def cases(*names):
    return [
        {
            "id": name,
            "dataset": "r2r",
            "episode_id": name,
            "episode_index": i,
            "scene": "test",
            "instruction": "Navigate",
        }
        for i, name in enumerate(names)
    ]


def outcome(success, steps=3, **metrics):
    return {
        "status": "success" if success else "task_failure",
        "success": success,
        "score": float(success),
        "native": {
            "success": success,
            "steps": steps,
            "terminated": True,
            "truncated": False,
            "metrics": metrics,
        },
    }


def campaign(tmp_path, snapshot, items, backend, **kwargs):
    return Campaign(
        tmp_path / "run", {"seed": 42}, items, {"r2r": snapshot}, backend, **kwargs
    )


def test_conditional_stages_and_whole_run_selection(tmp_path, snapshot):
    calls = []
    active, peak = 0, 0
    lock = threading.Lock()
    finished = set()

    def execute(case, budget, frozen, directory):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            calls.append((case["id"], budget, case["episode_index"], directory))
            if budget == 5000:
                assert finished == {"early", "late", "failed", "broken"}
        assert frozen == snapshot
        assert directory.is_dir()
        time.sleep(0.015)
        with lock:
            active -= 1
            if budget == 500:
                finished.add(case["id"])
        if case["id"] == "broken":
            raise TimeoutError("simulator unavailable")
        success = case["id"] == "early" or (case["id"] == "late" and budget == 5000)
        return outcome(success, budget, distance=7 if budget == 5000 else 1)

    runner = campaign(
        tmp_path,
        snapshot,
        cases("early", "late", "failed", "broken"),
        execute,
        concurrency=2,
        retries=0,
    )
    report = runner.run()
    assert peak == 2
    assert {c[0] for c in calls if c[1] == 5000} == {"late", "failed"}
    rows = {r["id"]: r for r in report["cases"]}
    assert rows["late"]["selected"]["native"]["metrics"]["distance"] == 7
    assert rows["late"]["policy_steps"] == 5500
    assert rows["failed"]["selected"]["success"] is False
    assert rows["broken"]["selected"] is None
    assert report["datasets"]["r2r"] == {
        "requested": 4,
        "scored": 3,
        "incomplete": 1,
        "successful": 2,
        "success_rate_scored": 2 / 3,
        "success_rate": None,
        "stage_500": {"scored": 3, "successful": 1, "success_rate_scored": 1 / 3},
        "stage_5000": {"scored": 2, "successful": 1, "success_rate_scored": 1 / 2},
    }
    before = len(calls)
    runner.run(resume=True)
    assert len(calls) == before + 1  # only the unresolved infrastructure case
    assert len({c[3] for c in calls}) == len(calls)


def test_all_success_skip_stage_two_and_resume_without_duplicates(tmp_path, snapshot):
    calls = []

    def execute(case, budget, frozen, directory):
        calls.append(budget)
        return outcome(True)

    runner = campaign(tmp_path, snapshot, cases("a", "b"), execute)
    assert runner.run()["complete"]
    assert calls == [500, 500]
    runner.run(resume=True)
    assert calls == [500, 500]
    assert read_json(tmp_path / "run/stage-5000.json")["cases"] == []
    with pytest.raises(ValueError, match="resume"):
        runner.run()


def test_resume_after_barrier_interruption(tmp_path, snapshot):
    cancel = threading.Event()
    calls = []

    def execute(case, budget, frozen, directory):
        calls.append((case["id"], budget))
        if budget == 500:
            cancel.set()
        return outcome(budget == 5000)

    runner = campaign(tmp_path, snapshot, cases("a"), execute, cancel=cancel)
    assert not runner.run()["complete"]
    cancel.clear()
    assert runner.run(resume=True)["complete"]
    assert calls == [("a", 500), ("a", 5000)]


def test_technical_retries_do_not_trigger_long_budget(tmp_path, snapshot):
    calls = []

    def execute(case, budget, frozen, directory):
        calls.append(budget)
        if len(calls) == 1:
            return {"status": "agent_error", "success": None, "score": None}
        return outcome(True)

    runner = campaign(tmp_path, snapshot, cases("a"), execute, retries=1)
    assert runner.run()["complete"]
    assert calls == [500, 500]
    assert len(list((tmp_path / "run").glob("cases/*/500/attempt-*/outcome.json"))) == 2


def test_missing_native_score_never_triggers_long_budget(tmp_path, snapshot):
    calls = []

    def execute(case, budget, frozen, directory):
        calls.append(budget)
        return {"status": "task_failure", "score": 0, "success": False}

    assert not campaign(tmp_path, snapshot, cases("a"), execute, retries=0).run()[
        "complete"
    ]
    assert calls == [500]


def test_resume_rejects_changed_inputs_and_results(tmp_path, snapshot):
    items = cases("a")
    runner = campaign(tmp_path, snapshot, items, lambda *args: outcome(True))
    runner.run()
    with pytest.raises(ValueError, match="inputs"):
        Campaign(
            tmp_path / "run",
            {"seed": 43},
            items,
            {"r2r": snapshot},
            lambda *a: outcome(True),
        ).run(resume=True)
    path = next((tmp_path / "run").glob("cases/*/500/attempt-*/outcome.json"))
    document = read_json(path)
    document["result"]["score"] = 0.1
    write_json(path, document)
    with pytest.raises(ValueError, match="changed"):
        runner.run(resume=True)


def test_campaign_resume_rejects_changed_stage_policy(tmp_path, snapshot):
    items = cases("a")
    campaign(tmp_path, snapshot, items, lambda *a: outcome(False), only_500=True).run()
    with pytest.raises(ValueError, match="changed"):
        campaign(tmp_path, snapshot, items, lambda *a: outcome(True)).run(resume=True)


def test_snapshot_verification_and_symlink_rejection(tmp_path, snapshot):
    (snapshot / "workspace/EXPLORATION.md").write_text("Changed")
    with pytest.raises(ValueError, match="changed"):
        verify_snapshot(snapshot)
    (tmp_path / "training/outside").symlink_to(tmp_path / "snapshot")
    with pytest.raises(ValueError, match="resource"):
        freeze_snapshot(tmp_path / "training", tmp_path / "unsafe")


def test_fresh_workspace_and_agent_export(tmp_path, snapshot, monkeypatch):
    preset = make_preset(
        tmp_path / "r2r",
        {"executable": "codex", "expected_version": "codex-cli 0.161.0"},
        snapshot,
    )
    captured = []

    def embodied(self, directory, **kwargs):
        captured.append(kwargs["working_dir"])
        return object()

    monkeypatch.setattr(AgentConfig, "embodied_agent", embodied)
    config = AgentConfig.from_file(preset)
    for name in ("500", "5000"):
        directory = tmp_path / name
        directory.mkdir()
        config.create_agent(directory)
        if name == "500":
            (directory / "workspace/EXPLORATION.md").write_text(
                "Test memory must not propagate"
            )
    assert captured[0] != captured[1]
    assert "Test memory" not in (captured[1] / "EXPLORATION.md").read_text()
    exported = export_agent(preset, tmp_path / "wheel", name="vln_test")
    manifest = read_json(exported / "src/roborsi_agent_bundle_vln_test/manifest.json")
    assert "exploration/workspace/EXPLORATION.md" in manifest["files"]
    assert verify_snapshot(snapshot)


PNG = base64.b64encode(b"public image bytes").decode()


class NativeStub:
    def __init__(self, python, directory, *, timeout):
        self.step = 0
        self.closed = False
        self.actions = []

    def call(self, command, **args):
        if command == "start":
            self.spec = args["spec"]
        elif command == "step":
            self.actions.append(args["action"])
            # One forward request can perform 3 native actions during recovery.
            self.step += 3 if args["action"] == "move_forward" else 1
        elif command == "finish":
            return outcome(False, self.step)["native"]
        return {
            "instruction": "navigate",
            "step_index": self.step,
            "terminated": self.step >= 4,
            "truncated": False,
            "images": {"front": PNG},
            "feedback": {"subtask_boundary": args.get("action") == "stop"},
        }

    def close(self):
        self.closed = True


def native_spec():
    return {
        "dataset": "r2r",
        "python": "unused",
        "seed": 42,
        "split": "test",
        "inputs": {},
        "native_commit": "a" * 40,
    }


def test_adapter_native_budget_batch_boundary_and_lifecycle(tmp_path):
    spec, episode = native_spec(), cases("a")[0]
    task = make_task(episode, spec, 5)
    adapter = NavigationAdapter(tmp_path, [task], spec, NativeStub)
    start = adapter.start(task, "episode")
    assert start.observation.step_index == 0
    assert adapter.worker.spec["episode_index"] == episode["episode_index"]
    assert adapter.worker.spec["seed"] == 42
    with pytest.raises(ValueError, match="terminal"):
        adapter.finish()
    result = adapter.step(
        Action(
            benchmark="vlnce",
            episode_id="episode",
            step_index=0,
            data={"actions": ["stop", "move_forward"]},
        )
    )
    assert adapter.worker.actions == ["stop"]
    assert [a["action"] for a in result.observation.data["executed_batch"]] == ["stop"]
    assert result.observation.step_index == 1
    assert result.observation.data["physical_steps"] == 1
    result = adapter.step(
        Action(
            benchmark="vlnce",
            episode_id="episode",
            step_index=1,
            data={"actions": ["move_forward", "turn_left"]},
        )
    )
    assert result.terminated
    assert adapter.worker.actions == ["stop", "move_forward"]
    assert result.observation.step_index == 2
    assert result.observation.data["physical_steps"] == 4
    assert result.observation.data["executed_batch"][0]["physical_steps"] == 4
    assert adapter.finish().success is False
    adapter.close()
    adapter.close()
    assert adapter.worker.closed


def test_http_public_media_and_native_artifacts(tmp_path):
    class Provider(NavigationBenchmark):
        def create_adapter(self, directory, artifacts):
            return NavigationAdapter(directory, self.tasks(), self.spec, NativeStub)

    provider = Provider(native_spec(), cases("a"), 5)
    channel = CaseMediaChannel(tmp_path / "client/media")
    with serve(provider, tmp_path / "service") as endpoint:
        client = BenchmarkClient(endpoint)
        try:
            task = client.tasks()[0]
            start = client.start(task, "test", timeout_s=5)
            client.collect_observation(start.observation, channel, timeout_s=5)
            assert (
                channel.read(start.observation.media[0], "test", 0)
                == b"public image bytes"
            )
            result = client.step(
                Action(
                    benchmark="vlnce",
                    episode_id="test",
                    step_index=0,
                    data={"actions": ["turn_left", "move_forward"]},
                ),
                timeout_s=5,
            )
            assert result.terminated
            native = client.finish(timeout_s=5)
            client.collect_native(native, tmp_path / "client", timeout_s=5)
            assert read_json(tmp_path / "client/native/result.json")["steps"] == 4
        finally:
            client.close()


def test_bench_executor_live_progress_uses_physical_steps(
    tmp_path, snapshot, monkeypatch, capsys
):
    from robots.vln_shared import benchmark

    def adapter(directory, tasks, spec, **kwargs):
        return NavigationAdapter(directory, tasks, spec, NativeStub, **kwargs)

    monkeypatch.setattr(benchmark, "NavigationAdapter", adapter)
    source = tmp_path / "policy"
    source.mkdir()
    (source / "policy.py").write_text("""from roborsi.evaluation.messages import Action

def make_agent(case_dir, *, config):
    class Policy:
        def reset(self, task):
            self.task = task
        def act(self, observation):
            return Action(benchmark=self.task.benchmark,
                          episode_id=observation.episode_id,
                          step_index=observation.step_index,
                          data={"actions": ["turn_left", "move_forward"]})
        def close(self):
            pass
    return Policy()
""")
    preset = source / "agent.json"
    write_json(
        preset,
        {"schema_version": 1, "factory": "policy:make_agent", "code": ["policy.py"]},
    )
    with Progress(interval=0.05) as progress:
        executor = BenchExecutor(
            {"r2r": {"test": {"spec": native_spec()}}},
            {"r2r": preset},
            threading.Event(),
            progress=progress,
        )
        runner = campaign(
            tmp_path,
            snapshot,
            cases("a", "b"),
            executor,
            concurrency=2,
            retries=0,
            only_500=True,
            progress=progress,
        )
        report = runner.run()
        assert report["complete"]
        assert all(row["selected"]["native"]["steps"] == 4 for row in report["cases"])
        assert not progress.active
    output = capsys.readouterr()
    assert not output.out
    for name in ("a", "b"):
        lines = [line for line in output.err.splitlines() if f"episode={name} " in line]
        assert any("event=start steps=?/500" in line for line in lines)
        assert any("event=step steps=0/500" in line for line in lines)
        assert any("event=step steps=4/500" in line for line in lines)
        assert any(
            "event=attempt_end steps=4/500" in line and "status=task_failure" in line
            for line in lines
        )
        assert not any("steps=1/500" in line for line in lines)
    assert not progress.thread.is_alive()


def test_progress_resets_on_retry_and_5000_stage(tmp_path, snapshot, capsys):
    with Progress() as progress:
        calls = []

        def execute(case, budget, frozen, directory):
            calls.append(budget)
            progress.advance(case, budget, 0)
            progress.advance(case, budget, 9)
            if len(calls) == 1:
                raise TimeoutError("offline transient failure")
            return outcome(budget == 5000, steps=9)

        report = campaign(
            tmp_path, snapshot, cases("retry"), execute, progress=progress
        ).run()
        assert report["complete"]
    lines = capsys.readouterr().err.splitlines()
    assert calls == [500, 500, 5000]
    assert any(
        "event=retry steps=?/500" in line and "retry=1/1" in line for line in lines
    )
    assert any(
        "event=start steps=?/5000" in line and "retry=0/1" in line for line in lines
    )
    assert any("event=step steps=0/5000" in line for line in lines)
    assert any(
        "event=attempt_end steps=9/5000" in line and "status=success" in line
        for line in lines
    )


@pytest.mark.parametrize("value", [float("nan"), None, "0"])
def test_invalid_native_scores(value):
    result = outcome(False)
    result["score"] = value
    with pytest.raises(ValueError, match="native"):
        validate_outcome(result)


def test_recover_committed_attempt_without_running_it_again(tmp_path, snapshot):
    class Backend:
        calls = 0

        def __call__(self, case, budget, frozen, directory):
            self.calls += 1
            return outcome(True)

        def recover(self, case, budget, directory):
            return {**outcome(True), "completed_at": time.time()}

    backend = Backend()
    runner = campaign(tmp_path, snapshot, cases("a"), backend)
    runner.run()
    path = next((tmp_path / "run").glob("cases/*/500/attempt-*/outcome.json"))
    path.unlink()  # simulate losing only the scheduler commit after bench committed
    assert runner.run(resume=True)["complete"]
    assert backend.calls == 1
    assert read_json(path)["result"]["recovered"]


def test_exclusive_campaign_lock(tmp_path, snapshot):
    runner = campaign(tmp_path, snapshot, cases("a"), lambda *a: outcome(True))
    runner.root.mkdir()
    with (runner.root / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            runner.run()


def test_subprocess_cancellation_reaps_owned_process(tmp_path):
    cancel = threading.Event()
    timer = threading.Timer(0.1, cancel.set)
    timer.start()
    try:
        with pytest.raises(InterruptedError):
            run_process(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                tmp_path,
                cancel,
                10,
            )
        from robots.vln_shared.execution import _birth

        assert _birth(read_json(tmp_path / "process.json")["pid"]) is None
    finally:
        timer.cancel()


def test_exploration_tools_publish_ready_only_after_practice(tmp_path, monkeypatch):
    original = NavigationBenchmark.create_adapter

    def fake(self, directory, artifacts):
        return NavigationAdapter(directory, self.tasks(), self.spec, NativeStub)

    monkeypatch.setattr(NavigationBenchmark, "create_adapter", fake)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tools = TrainingTools(
        {"spec": native_spec(), "episodes": cases("train")}, tmp_path, workspace
    )
    try:
        assert tools.browse({"offset": 0}).result["tasks"][0]["id"] == "train"
        with pytest.raises(ValueError):
            tools.complete({})
        first = tools.reset({"id": "train"})
        assert first.result["physical_steps"] == 0
        tools.act({"actions": ["turn_left", "move_forward"]})
        second = tools.reset({"id": "train"})
        assert second.result["physical_steps"] == 0
        tools.remember({"path": "EXPLORATION.md", "content": "Observe before moving."})
        assert tools.complete({}).is_finish
        with pytest.raises(ValueError):
            tools.remember({"path": "../escape", "content": "forbidden"})
    finally:
        tools.close()
        assert original is not fake


def test_portable_manifest_checks_assets_and_paths(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    write_json(assets / "catalog.json", {"spec": native_spec(), "episodes": cases("a")})
    document = manifest(
        assets, tmp_path / "benchmark.json", "example/vlnce@sha256:" + "b" * 64
    )
    assert document["assets"][0]["path"] == "assets"
    assert document["tasks"][0]["seeds"] == [42]
    assert document["capabilities"]["observation_channels"] == ["front"]
    with pytest.raises(ValueError):
        manifest(assets, assets / "manifest.json", "bad-image")
    (assets / "link").symlink_to(tmp_path / "benchmark.json")
    with pytest.raises(ValueError, match="symlink"):
        manifest(assets, tmp_path / "other.json", "example/vlnce@sha256:" + "b" * 64)


def test_failed_training_reset_cannot_publish_ready(tmp_path, monkeypatch):
    from roborsi.utils.rpc import RpcError

    class BrokenNative(NativeStub):
        def call(self, command, **arguments):
            raise RuntimeError("native startup failed")

    def fake(self, directory, artifacts):
        return NavigationAdapter(directory, self.tasks(), self.spec, BrokenNative)

    monkeypatch.setattr(NavigationBenchmark, "create_adapter", fake)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tools = TrainingTools(
        {"spec": native_spec(), "episodes": cases("train")}, tmp_path, workspace
    )
    try:
        tools.remember({"path": "EXPLORATION.md", "content": "not yet practiced"})
        with pytest.raises(RpcError, match="Benchmark execution"):
            tools.reset({"id": "train"})
        assert tools.resets == 0
        with pytest.raises(ValueError, match="practice"):
            tools.complete({})
    finally:
        tools.close()
