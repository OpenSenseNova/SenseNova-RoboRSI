# Copyright 2026 The SenseNova-RoboRSI Authors.
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

"""RoboCasa365 catalog, action schema, scoring and cleanup without a simulator."""

import json

import pytest
from jsonschema import Draft202012Validator, ValidationError

from roborsi.evaluation.messages import Action, EpisodeStop
from roborsi.tools.toolkit import ToolResult
from robots.robocasa import tools as robocasa_tools
from robots.robocasa.benchmark_service import (
    ACTION_SCHEMA,
    PRIMITIVES,
    RLDX_ENVIRONMENT,
    TARGET50,
    RoboCasa365Benchmark,
    make_task,
    runtime_info,
)
from robots.robocasa.roborsi_adapter import RoboCasa365Adapter

PNG = b"\x89PNG\r\n\x1a\nfake"


@pytest.fixture(autouse=True)
def _restore_rldx_environment():
    """The adapter pins process-wide RLDX limits; undo them after each test."""
    import os

    saved = {name: os.environ.get(name) for name in RLDX_ENVIRONMENT}
    yield
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


class FakeToolkit:
    """Record calls; the latest state decides success, as RoboCasaToolkit does."""

    def __init__(self):
        self.calls = []
        self.success = False
        self.closed = False

    def execute_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "finish":
            return ToolResult(name, robocasa_tools.finish(**arguments))
        return ToolResult(
            name,
            {
                "step": len(self.calls),
                "state": {"success": self.success, "task_language": "open it"},
                "_image_cam_bytes": PNG,
                "_image_nav_bytes": PNG,
                "_image_wrist_bytes": PNG,
            },
        )

    def solved(self):
        return self.success

    def close(self):
        self.closed = True


class FakeRpc:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


class FakeClient:
    def __init__(self, rpc):
        self._client = rpc


class Runtime:
    def __init__(self, fail=False):
        self.fail = fail
        self.args = None
        self.toolkit = FakeToolkit()
        self.stopped = []

    def start(self, args, output_dir, events):
        self.args = args
        self.rpcs = [FakeRpc(), FakeRpc()]
        return ["env-daemon"], {
            "env_client": FakeClient(self.rpcs[0]),
            "vla_client": FakeClient(self.rpcs[1]),
        }

    def make(self, runtime_kwargs, events, state_dir):
        if self.fail:
            raise RuntimeError("kitchen failed to load")
        return self.toolkit

    def stop(self, daemons, events):
        self.stopped.extend(daemons)


def create(tmp_path, task="OpenDrawer", seed=1, runtime=None, vla="http://vla:1"):
    runtime = runtime or Runtime()
    adapter = RoboCasa365Adapter(
        tmp_path / "case",
        vla_endpoint=vla,
        cuda_device=3,
        start_runtime=runtime.start,
        make_toolkit=runtime.make,
        stop_runtime=runtime.stop,
    )
    return adapter, runtime, adapter.start(make_task(task, seed), "episode")


def action(index, **data):
    return Action(
        benchmark="robocasa365", episode_id="episode", step_index=index, data=data
    )


def test_catalog_is_the_target50_protocol():
    tasks = RoboCasa365Benchmark("http://vla:1").tasks()
    assert len(tasks) == TARGET50["total_cells"] == 340
    assert len({task.task_id for task in tasks}) == 50
    seed1 = RoboCasa365Benchmark("http://vla:1", cells="seed1").tasks()
    assert len(seed1) == 50 and {task.seed for task in seed1} == {1}
    task = make_task("ArrangeTea", 5)
    assert task.native["split"] == "composite_unseen"
    assert task.native["timeout_s"] == 3600
    assert make_task("OpenDrawer", 10).native["timeout_s"] == 1800
    with pytest.raises(ValueError):
        make_task("ArrangeTea", 6)
    with pytest.raises(ValueError):
        make_task("NotATask", 1)
    info = runtime_info()
    assert info.benchmark == "robocasa365"
    assert info.benchmark_commit == TARGET50["dependencies"]["robocasa"]["commit"]
    assert info.evaluation["aggregation"] == "task_weighted_success_rate"


def test_action_schema_names_each_evaluation_tool_once():
    Draft202012Validator.check_schema(ACTION_SCHEMA)
    names = [spec["name"] for spec in robocasa_tools.TOOLS_SPEC]
    assert list(PRIMITIVES) == [name for name in names if name != "reset"]
    validator = Draft202012Validator(ACTION_SCHEMA)
    validator.validate({"primitive": "navigate_to", "xy": [1.0, 0.5], "tol": 0.6})
    validator.validate({"primitive": "view_env_state"})
    validator.validate({"primitive": "finish", "status": "success", "summary": "x"})
    for bad in ({"primitive": "reset"}, {"primitive": "teleport"}, {"xyz": [0, 0, 0]}):
        with pytest.raises(ValidationError):
            validator.validate(bad)
    # Argument mistakes reach the tool, which reports them and the episode goes on.
    validator.validate({"primitive": "navigate_to"})
    validator.validate({"primitive": "move_delta", "dxyz": [0, 0, 0.1], "tol": 0.01})


def test_one_action_runs_one_tool_and_returns_its_state(tmp_path):
    adapter, runtime, start = create(tmp_path)
    assert runtime.args.task_name == "OpenDrawer"
    assert runtime.args.split == "target" and runtime.args.seed == 1
    assert runtime.args.vla_endpoint == "http://vla:1"
    assert runtime.args.cuda_device == 3 and runtime.args.env_endpoint is None
    assert runtime.toolkit.calls == [("view_env_state", {})]
    assert start.observation.step_index == 0
    assert start.observation.data["camera_names"] == ["agentview", "navview", "wrist"]
    assert len(start.observation.media) == 3
    assert not any(key.endswith("_bytes") for key in start.observation.data)
    step = adapter.step(action(0, primitive="rldx_arm", prompt="open the drawer"))
    assert runtime.toolkit.calls[-1] == ("rldx_arm", {"prompt": "open the drawer"})
    assert step.observation.step_index == 1
    assert not (step.terminated or step.truncated)
    with pytest.raises(ValueError):
        adapter.step(action(0, primitive="view_env_state"))


@pytest.mark.parametrize("success", [False, True])
def test_finish_ends_the_cell_and_state_success_scores_it(tmp_path, success):
    adapter, runtime, _ = create(tmp_path)
    with pytest.raises(ValueError):
        adapter.finish()
    runtime.toolkit.success = success
    step = adapter.step(action(0, primitive="finish", status="success", summary="ok"))
    assert step.terminated and step.termination_reason == "agent_finish"
    assert step.observation.data["finish"]["status"] == "success"
    native = adapter.finish()
    assert native.success is success and native.score == float(success)
    assert native.details["success_source"] == "state.success"
    assert native.details["actions"] == 1
    assert adapter.finish() == native
    score = json.loads((tmp_path / "case/native/score.json").read_text())
    assert score["termination_reason"] == "agent_finish"
    lines = (tmp_path / "case/native/trajectory.jsonl").read_text().splitlines()
    assert [json.loads(line)["phase"] for line in lines] == ["reset", "action"]
    adapter.close()
    assert runtime.toolkit.closed and runtime.stopped == ["env-daemon"]


def test_stop_scores_the_state_the_planner_left(tmp_path):
    adapter, runtime, _ = create(tmp_path)
    adapter.step(action(0, primitive="view_env_state"))
    runtime.toolkit.success = True
    stop = adapter.stop(
        EpisodeStop(
            benchmark="robocasa365",
            episode_id="episode",
            step_index=1,
            reason="planner_timeout",
        )
    )
    assert stop.truncated and stop.observation.step_index == 1
    assert adapter.finish().success is True
    adapter.close()
    adapter.close()
    assert runtime.stopped == ["env-daemon"]


def test_step_budget_truncates_a_runaway_cell(tmp_path):
    adapter, _, _ = create(tmp_path)
    adapter.task = adapter.task.model_copy(update={"max_steps": 2})
    assert not adapter.step(action(0, primitive="view_env_state")).truncated
    last = adapter.step(action(1, primitive="view_env_state"))
    assert last.truncated and last.termination_reason == "episode_limit"
    assert adapter.finish().details["termination_reason"] == "episode_limit"


def test_start_rejects_foreign_tasks_and_cleans_up_failures(tmp_path):
    adapter = RoboCasa365Adapter(tmp_path / "a", vla_endpoint="http://vla:1")
    foreign = make_task("OpenDrawer", 1).model_copy(update={"max_steps": 5})
    with pytest.raises(ValueError):
        adapter.start(foreign, "episode")
    with pytest.raises(ValueError, match="RLDX-1"):
        create(tmp_path / "b", vla=None)
    runtime = Runtime(fail=True)
    with pytest.raises(RuntimeError):
        create(tmp_path / "c", runtime=runtime)
    assert runtime.stopped == ["env-daemon"]


def test_shared_http_service_runs_a_cell_and_returns_its_records(tmp_path):
    import threading

    from roborsi.evaluation.benchmark_client import BenchmarkClient
    from roborsi.evaluation.benchmark_http import (
        BenchmarkApplication,
        BenchmarkHttpServer,
    )
    from roborsi.evaluation.media import CaseMediaChannel

    runtime = Runtime()

    made = []

    class Provider(RoboCasa365Benchmark):
        def create_adapter(self, directory, artifacts):
            made.append(directory)
            return RoboCasa365Adapter(
                directory,
                vla_endpoint=self.vla_endpoint,
                artifacts=artifacts,
                start_runtime=runtime.start,
                make_toolkit=runtime.make,
                stop_runtime=runtime.stop,
            )

    app = BenchmarkApplication(Provider("http://vla:1"), tmp_path / "private")
    server = BenchmarkHttpServer(("127.0.0.1", 0), app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = BenchmarkClient(f"http://127.0.0.1:{server.server_port}")
    media = CaseMediaChannel(tmp_path / "agent/media")
    task = make_task("OpenDrawer", 1)
    try:
        assert task in client.tasks()
        assert client.runtime_info(timeout_s=3) == runtime_info()
        initial = client.start(task, "case", timeout_s=10)
        client.collect_observation(initial.observation, media, timeout_s=3)
        runtime.toolkit.success = True
        end = client.step(
            Action(
                benchmark="robocasa365",
                episode_id="case",
                step_index=0,
                data={"primitive": "finish", "status": "success", "summary": "ok"},
            ),
            timeout_s=5,
        )
        client.collect_observation(end.observation, media, timeout_s=3)
        assert end.terminated
        log = made[0] / "native/env_server.log"
        log.write_text("kitchen ready\n")
        native = client.finish(timeout_s=5)
        assert native.success is True
        # env_server keeps logging until close; the published copy must not move.
        with log.open("a") as stream:
            stream.write("still running\n")
        client.collect_native(native, tmp_path / "agent", timeout_s=3)
        client.collect_artifacts(tmp_path / "agent", timeout_s=3)
        client.close()
        client.delete("case")
        media.verify_observation(initial.observation)
        media.verify_observation(end.observation)
        assert (tmp_path / "agent/native/score.json").exists()
        assert (tmp_path / "agent/native/records/trajectory.jsonl").exists()
        records = tmp_path / "agent/native/records/env_server.log"
        assert records.read_text() == "kitchen ready\n"
        assert runtime.stopped == ["env-daemon"]
    finally:
        app.close()
        server.shutdown()
        server.server_close()
        thread.join(3)


def test_start_pins_the_target50_rldx_limits(tmp_path, monkeypatch):
    monkeypatch.setenv("RLDX_MAX_CHUNKS", "70")
    adapter, _, _ = create(tmp_path)
    import os

    assert os.environ["RLDX_MAX_CHUNKS"] == "40"
    assert os.environ["RLDX_ACTION_STEPS_PER_CHUNK"] == "8"
    adapter.step(action(0, primitive="finish", status="stuck", summary="x"))
    assert adapter.finish().details["rldx"]["RLDX_SETTLE_PATIENCE"] == "999"


def test_close_ends_this_episodes_rpc_sessions_but_not_the_shared_service(tmp_path):
    adapter, runtime, _ = create(tmp_path)
    adapter.step(action(0, primitive="finish", status="stuck", summary="x"))
    adapter.finish()
    adapter.close()
    adapter.close()
    assert [rpc.closed for rpc in runtime.rpcs] == [1, 1]
    assert runtime.stopped == ["env-daemon"]
    failing = Runtime(fail=True)
    with pytest.raises(RuntimeError):
        create(tmp_path / "failed", runtime=failing)
    assert [rpc.closed for rpc in failing.rpcs] == [1, 1]


def test_service_starts_one_owned_rldx_server_when_no_endpoint_is_given(
    tmp_path, monkeypatch
):
    from robots.robocasa import benchmark_service

    spawned = []

    def spawn(checkpoint, cuda):
        spawned.append((checkpoint, cuda))
        return object(), "http://127.0.0.1:4321"

    monkeypatch.delenv("ROBORSI_ROBOCASA_VLA_ENDPOINT", raising=False)
    monkeypatch.setenv("RLDX_CHECKPOINT", "/opt/rldx/RLDX-1-FT-RC365")
    provider = benchmark_service.make_robocasa365_benchmark()
    provider._spawn_vla = spawn
    first = provider.create_adapter(tmp_path / "a", None)
    second = provider.create_adapter(tmp_path / "b", None)
    assert spawned == [("/opt/rldx/RLDX-1-FT-RC365", None)]
    assert first.vla_endpoint == second.vla_endpoint == "http://127.0.0.1:4321"
    attached = RoboCasa365Benchmark(
        "http://vla:1", vla_checkpoint="/ckpt", spawn_vla=spawn
    )
    assert attached.create_adapter(tmp_path / "c", None).vla_endpoint == "http://vla:1"
    assert len(spawned) == 1


def test_bad_tool_arguments_come_back_as_feedback(tmp_path):
    adapter, runtime, _ = create(tmp_path)

    def execute(name, arguments):
        runtime.toolkit.calls.append((name, arguments))
        return ToolResult(
            name, {"error": f"bad arguments for {name}", "got": arguments}
        )

    runtime.toolkit.execute_tool = execute
    step = adapter.step(action(0, primitive="move_delta", dxyz=[0, 0, 0.1], tol=0.01))
    assert not (step.terminated or step.truncated)
    assert step.observation.data["error"] == "bad arguments for move_delta"
