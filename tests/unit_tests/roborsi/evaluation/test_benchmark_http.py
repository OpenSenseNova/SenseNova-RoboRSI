# Copyright 2026 The RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HTTP-only Benchmark execution and artifact transfer across private directories."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from contextlib import contextmanager

import pytest

from roborsi.evaluation.benchmark_client import BenchmarkClient
from roborsi.evaluation.benchmark_http import (
    BenchmarkApplication,
    BenchmarkHttpServer,
)
from roborsi.evaluation.media import CaseMediaChannel
from roborsi.evaluation.messages import (
    Action,
    BenchmarkRuntimeInfo,
    NativeResult,
    Observation,
    StartResult,
    StepResult,
    Task,
    to_envelope,
)
from roborsi.utils.rpc import RpcError


def task():
    return Task(
        benchmark="demo", task_id="place", seed=7, instruction="Place", max_steps=2
    )


class Provider:
    name = "demo"

    def __init__(self):
        self.steps = 0
        self.closed = 0
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.fail = False

    def tasks(self):
        return [task()]

    def assets(self):
        return []

    def runtime_info(self):
        return BenchmarkRuntimeInfo(
            benchmark="demo",
            benchmark_version="demo-1",
            benchmark_commit="a" * 40,
            adapter_version="adapter-1",
            evaluation={
                "aggregation": "case_mean",
                "error_denominator": "expected",
                "score_metric": "native_score",
                "success_metric": "native_success",
            },
            platform={"machine": "test"},
        )

    def create_adapter(self, directory, artifacts):
        return Adapter(self, directory, artifacts)


class Adapter:
    def __init__(self, provider, directory, artifacts):
        self.provider, self.directory, self.artifacts = provider, directory, artifacts
        self.media = CaseMediaChannel(directory / "media")
        self.index = 0

    def observation(self):
        return Observation(
            benchmark="demo",
            episode_id=self.episode_id,
            step_index=self.index,
            media=(
                self.media.write(
                    self.episode_id, self.index, b"camera image", "image/png"
                ),
            ),
        )

    def start(self, task, episode_id):
        self.task, self.episode_id = task, episode_id
        return StartResult(task=task, observation=self.observation())

    def step(self, action):
        self.provider.entered.set()
        assert self.provider.release.wait(5)
        if self.provider.fail:
            raise RuntimeError("do not expose /private/path or backend traceback")
        self.provider.steps += 1
        self.index += 1
        return StepResult(
            observation=self.observation(),
            terminated=True,
            truncated=False,
            termination_reason="native_success",
        )

    def finish(self):
        source = self.directory / "native" / "score.json"
        source.parent.mkdir(exist_ok=True)
        source.write_bytes(b'{"score":0.75}')
        return NativeResult(
            task=self.task,
            episode_id=self.episode_id,
            success=True,
            score=0.75,
            source="native/score.json",
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        )

    def close(self):
        self.provider.closed += 1


@contextmanager
def server(tmp_path, *, token=None):
    provider = Provider()
    application = BenchmarkApplication(provider, tmp_path / "server-private")
    http = BenchmarkHttpServer(("127.0.0.1", 0), application, token=token)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    client = BenchmarkClient(f"http://127.0.0.1:{http.server_port}", token=token)
    try:
        yield provider, application, client
    finally:
        provider.release.set()
        application.close()
        http.shutdown()
        http.server_close()
        thread.join(3)


def test_interactive_outputs_use_http_and_survive_server_deletion(tmp_path):
    with server(tmp_path) as (provider, app, client):
        assert client.tasks() == [task()]
        assert client.runtime_info(timeout_s=3).benchmark_commit == "a" * 40
        start = client.start(task(), "case-one", timeout_s=3)
        channel = CaseMediaChannel(tmp_path / "runner" / "media")
        assert not list(channel.media_dir.rglob("*.bin"))
        client.collect_observation(start.observation, channel, timeout_s=3)
        step = client.step(
            Action(benchmark="demo", episode_id="case-one", step_index=0, data={}),
            timeout_s=3,
        )
        client.collect_observation(step.observation, channel, timeout_s=3)
        native = client.finish(timeout_s=3)
        client.collect_native(native, channel.media_dir.parent, timeout_s=3)
        client.collect_artifacts(channel.media_dir.parent, timeout_s=3)
        infos = client.artifacts()
        assert {a.kind for a in infos} == {"image", "native_result", "log"}
        client.close()
        client.delete("case-one")
        assert not list(app.root.iterdir())
        channel.verify_observation(step.observation)
        assert (tmp_path / "runner/native/score.json").read_bytes() == b'{"score":0.75}'
        assert len(list((tmp_path / "runner/trace/events").glob("*.json"))) == 3
        assert provider.steps == 1 and provider.closed == 1
        with pytest.raises(RpcError, match="episode not found"):
            client.status("case-one")
        assert client.start(task(), "case-two", timeout_s=3).task == task()
        client.close()


def test_complete_test_and_retries_do_not_execute_actions_twice(tmp_path):
    with server(tmp_path) as (provider, _, client):
        submitted = client.submit_test(
            "full", task(), [{}], request_id="test-1", repeat_last=True
        )
        assert submitted["state"] in {"starting", "running", "completed"}
        end = time.monotonic() + 3
        while client.status("full")["state"] not in {"completed", "failed"}:
            assert time.monotonic() < end
            time.sleep(0.01)
        code, result = client.result("full")
        assert code == 200 and result["payload"]["score"] == 0.75
        assert (
            client.submit_test(
                "full", task(), [{}], request_id="test-1", repeat_last=True
            )["state"]
            == "completed"
        )
        assert provider.steps == 1
        with pytest.raises(RpcError, match="another complete test"):
            client.submit_test("full", task(), [{"different": 1}], request_id="test-1")
        for info in client.artifacts(episode_id="full"):
            client.download(info, tmp_path / "download")
        client.delete("full")


def test_replay_conflict_order_and_cross_episode_requests(tmp_path):
    with server(tmp_path) as (provider, _, client):
        transport = client.transport
        wire = to_envelope(
            task(), benchmark="demo", episode_id="one", request_id="start-1"
        ).as_dict()
        _, start = transport.request("POST", "/v1/episodes", body=wire)
        assert transport.request("POST", "/v1/episodes", body=wire)[1] == start
        action = Action(benchmark="demo", episode_id="one", step_index=1, data={})
        body = to_envelope(
            action, benchmark="demo", episode_id="one", request_id="step-1"
        ).as_dict()
        with pytest.raises(RpcError, match="out of order"):
            transport.request("POST", "/v1/episodes/one/actions", body=body)
        action = action.model_copy(update={"step_index": 0})
        body = to_envelope(
            action, benchmark="demo", episode_id="one", request_id="step-1"
        ).as_dict()
        _, response = transport.request("POST", "/v1/episodes/one/actions", body=body)
        assert (
            transport.request("POST", "/v1/episodes/one/actions", body=body)[1]
            == response
        )
        assert provider.steps == 1
        body["payload"]["data"] = {"other": 1}
        with pytest.raises(RpcError):
            transport.request("POST", "/v1/episodes/one/actions", body=body)
        other = {**wire, "episode_id": "two"}
        with pytest.raises(RpcError, match="still running"):
            transport.request("POST", "/v1/episodes", body=other)
        with pytest.raises(RpcError, match="stop the episode"):
            client.delete("one")


def test_cancel_and_status_remain_available_during_native_call(tmp_path):
    with server(tmp_path) as (provider, _, client):
        provider.release.clear()
        client.submit_test("full", task(), [{}], request_id="test-1")
        assert provider.entered.wait(2)
        started = time.monotonic()
        assert client.status("full")["state"] == "running"
        assert client.cancel("full")["state"] == "cancelling"
        assert time.monotonic() - started < 1
        assert client.result("full")[0] == 202
        provider.release.set()
        end = time.monotonic() + 3
        while client.status("full")["state"] != "cancelled":
            assert time.monotonic() < end
            time.sleep(0.01)
        assert client.result("full")[1]["state"] == "cancelled"
        assert provider.closed == 1
        client.delete("full")


def test_auth_artifact_integrity_and_cross_episode_access(tmp_path):
    with server(tmp_path, token="test-token") as (_, app, client):
        stranger = BenchmarkClient(client.transport.endpoint)
        with pytest.raises(RpcError, match="authentication required"):
            stranger.tasks()
        start = client.start(task(), "one", timeout_s=3)
        info = client.artifacts(kind="image")[0]
        private = next(app.root.iterdir()) / info.ref
        private.write_bytes(b"corrupt")
        with pytest.raises(RpcError, match="validation failed"):
            client.download(info, tmp_path / "download")
        assert not (tmp_path / "download" / info.ref).exists()
        private.write_bytes(b"camera image")
        client.close()
        client.start(task(), "two", timeout_s=3)
        with pytest.raises(RpcError, match="not found"):
            client.transport.request(
                "GET", f"/v1/episodes/two/artifacts/{info.artifact_id}"
            )
        channel = CaseMediaChannel(tmp_path / "runner/media")
        with pytest.raises(ValueError, match="reference differs"):
            client.collect_observation(start.observation, channel, timeout_s=3)
        client.close()


def test_native_failure_leaves_queryable_status_without_private_error_details(tmp_path):
    with server(tmp_path) as (provider, _, client):
        provider.fail = True
        client.start(task(), "one", timeout_s=3)
        with pytest.raises(RpcError, match="Benchmark execution"):
            client.step(
                Action(benchmark="demo", episode_id="one", step_index=0, data={}),
                timeout_s=3,
            )
        state = client.status("one")
        assert state["state"] == "failed" and state["error"]["code"] == "adapter_error"
        assert "private/path" not in json.dumps(state)
        assert client.result("one")[1]["state"] == "failed"
        assert any(a.kind == "image" for a in client.artifacts())
        client.close()


@pytest.mark.parametrize("phase", ["startup", "cleanup"])
def test_remote_metadata_checks_stop_waiting_when_cancelled(tmp_path, phase):
    from roborsi.evaluation.deadline import CaseCancelled, CaseDeadline
    from roborsi.evaluation.http_benchmark import (
        HttpBenchmarkExecutor,
        HttpBenchmarkLock,
    )

    with server(tmp_path) as (provider, _, client):
        lock = HttpBenchmarkLock(client.transport.endpoint)
        cancelled, release = threading.Event(), threading.Event()
        deadline = CaseDeadline(30, cancel_event=cancelled)
        executor = HttpBenchmarkExecutor()
        session = (
            executor.start_adapter(lock, tmp_path, deadline=deadline)
            if phase == "cleanup"
            else None
        )
        original = provider.runtime_info

        def slow_metadata():
            cancelled.set()
            release.wait(5)
            return original()

        provider.runtime_info = slow_metadata
        started = time.monotonic()
        try:
            with pytest.raises(CaseCancelled):
                if session is None:
                    executor.start_adapter(lock, tmp_path, deadline=deadline)
                else:
                    with session:
                        pass
            assert time.monotonic() - started < 1
        finally:
            release.set()


def test_remote_registry_and_batch_run_without_local_assets_or_docker(tmp_path):
    from roborsi.evaluation.agent_process import PythonAgentLauncher
    from roborsi.evaluation.batch import BatchEvaluator, expand_cases
    from roborsi.evaluation.http_benchmark import (
        HttpBenchmarkExecutor,
        HttpBenchmarkRegistry,
    )
    from roborsi.evaluation.orchestration import CaseOrchestrator
    from roborsi.evaluation.store import ResultStore

    with server(tmp_path, token="test-token") as (_, app, client):
        registry = HttpBenchmarkRegistry(
            {"demo": client.transport.endpoint}, token="test-token"
        )
        store = ResultStore(tmp_path / "runner-results")
        launcher = PythonAgentLauncher(
            "tests.unit_tests.roborsi.evaluation.service_child:make_demo_agent"
        )
        runner = CaseOrchestrator(registry, store, launcher, HttpBenchmarkExecutor())
        cases = expand_cases(registry, ["demo"])
        report = BatchEvaluator(registry, store, runner).run(
            "remote", cases, "demo-agent"
        )
        result = report.cases[0].result
        assert result.status == "success" and result.score == 0.75, result.as_dict()
        assert result.runtime.observed.docker_image_digest is None
        assert (
            result.runtime.benchmark.configuration["verification"] == "service_report"
        )
        assert store.load("remote", cases[0].case_id) == result
        assert store.trace_state("remote", cases[0].case_id).complete
        assert {a.kind for a in result.benchmark_artifacts} == {
            "image",
            "native_result",
            "log",
        }
        log = next(a for a in result.benchmark_artifacts if a.kind == "log")
        (store.root / "remote" / cases[0].case_id / log.ref).write_text("changed")
        with pytest.raises(ValueError, match="differs from HTTP metadata"):
            store.load("remote", cases[0].case_id)
        assert not list(app.root.iterdir())  # Downloaded first, then deleted over HTTP.
        assert list((tmp_path / "runner-results").rglob("score.json"))


def test_corrupted_http_bytes_never_become_archived_files(tmp_path, monkeypatch):
    with server(tmp_path) as (_, app, client):
        client.start(task(), "one", timeout_s=3)
        info = client.artifacts(kind="image")[0]
        handle = app.handle

        def corrupt(method, path, body):
            status, document, headers = handle(method, path, body)
            if isinstance(document, bytes):
                document = b"x" * len(document)
            return status, document, headers

        monkeypatch.setattr(app, "handle", corrupt)
        with pytest.raises(ValueError, match="SHA-256 differs"):
            client.download(info, tmp_path / "download")
        assert not (tmp_path / "download").exists()
        client.close()


def test_failed_full_test_and_invalid_requests_are_queryable(tmp_path):
    with server(tmp_path) as (provider, _, client):
        provider.fail = True
        client.submit_test("broken", task(), [{}], request_id="full-1")
        end = time.monotonic() + 3
        while client.status("broken")["state"] != "failed":
            assert time.monotonic() < end
            time.sleep(0.01)
        assert client.result("broken")[1]["error"]["code"] == "adapter_error"
        with pytest.raises(RpcError):
            client.submit_test("invalid", task(), [], request_id="full-2")
        with pytest.raises(RpcError):
            client.transport.request("POST", "/v1/episodes", body={"unexpected": True})
        client.delete("broken")


def test_inflight_duplicate_waits_for_the_original_native_result(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    with server(tmp_path) as (provider, _, client):
        client.start(task(), "one", timeout_s=3)
        provider.release.clear()
        wire = to_envelope(
            Action(benchmark="demo", episode_id="one", step_index=0, data={}),
            benchmark="demo",
            episode_id="one",
            request_id="same-step",
        ).as_dict()

        def request():
            return client.transport.request(
                "POST", "/v1/episodes/one/actions", body=wire
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(request)
            assert provider.entered.wait(2)
            replay = pool.submit(request)
            time.sleep(0.1)
            assert not replay.done()
            provider.release.set()
            assert first.result(timeout=2) == replay.result(timeout=2)
        assert provider.steps == 1
        client.close()


@pytest.mark.parametrize("fail_close", [False, True])
def test_cancel_does_not_wait_for_slow_close_and_preserves_cleanup_errors(
    tmp_path, monkeypatch, fail_close
):
    entered, release = threading.Event(), threading.Event()
    original = Adapter.close

    def slow_close(self):
        entered.set()
        assert release.wait(3)
        if fail_close:
            raise RuntimeError("native close failed")
        original(self)

    monkeypatch.setattr(Adapter, "close", slow_close)
    with server(tmp_path) as (_, _, client):
        client.start(task(), "one", timeout_s=3)
        try:
            started = time.monotonic()
            assert client.cancel("one")["state"] == "cancelling"
            assert time.monotonic() - started < 1
            assert entered.wait(1)
            assert client.status("one")["state"] == "cancelling"
        finally:
            release.set()
        end = time.monotonic() + 3
        while client.status("one")["state"] == "cancelling":
            assert time.monotonic() < end
            time.sleep(0.01)
        assert client.status("one")["state"] == (
            "failed" if fail_close else "cancelled"
        )
        if fail_close:
            with pytest.raises(RpcError, match="cleanup failed"):
                client.close()
        else:
            client.close()


def test_provider_can_publish_other_intermediate_files(tmp_path, monkeypatch):
    original = Adapter.step

    def step_with_file(self, action):
        result = original(self, action)
        output = self.directory / "trace" / "diagnostic.txt"
        output.write_text("native diagnostic")
        self.artifacts.publish(
            "trace/diagnostic.txt", "file", "text/plain", step_index=1
        )
        return result

    monkeypatch.setattr(Adapter, "step", step_with_file)
    with server(tmp_path) as (_, _, client):
        client.start(task(), "one", timeout_s=3)
        client.step(
            Action(benchmark="demo", episode_id="one", step_index=0, data={}),
            timeout_s=3,
        )
        files = client.artifacts(kind="file")
        assert len(files) == 1 and files[0].step_index == 1
        client.finish(timeout_s=3)
        client.collect_artifacts(tmp_path / "download", timeout_s=3)
        assert (
            tmp_path / "download/trace/diagnostic.txt"
        ).read_text() == "native diagnostic"
        assert files[0] in client.artifact_records
        client.close()


@pytest.mark.parametrize("cancel_first", [False, True], ids=["close", "cancel"])
def test_http_close_retries_failed_adapter_cleanup(tmp_path, monkeypatch, cancel_first):
    attempts = []
    original = Adapter.close

    def fail_once(self):
        attempts.append(self)
        if len(attempts) == 1:
            raise TimeoutError("native cleanup incomplete")
        original(self)

    monkeypatch.setattr(Adapter, "close", fail_once)
    with server(tmp_path) as (provider, _, client):
        client.start(task(), "retry-close", timeout_s=3)
        if cancel_first:
            client.cancel("retry-close")
            deadline = time.monotonic() + 3
            while client.status("retry-close")["state"] == "cancelling":
                assert time.monotonic() < deadline
                time.sleep(0.01)
            assert client.status("retry-close")["state"] == "failed"
        else:
            with pytest.raises(RpcError):
                client.transport.request(
                    "POST", "/v1/episodes/retry-close/close", body={}
                )
        assert len(attempts) == 1 and provider.closed == 0
        code, reply = client.transport.request(
            "POST", "/v1/episodes/retry-close/close", body={}
        )
        assert code == 200 and len(attempts) == 2 and provider.closed == 1
        assert reply["state"] == ("failed" if cancel_first else "closed")
        if cancel_first:
            assert reply["error"]["code"] == "infrastructure_error"
        client.transport.request("POST", "/v1/episodes/retry-close/close", body={})
        assert len(attempts) == 2
