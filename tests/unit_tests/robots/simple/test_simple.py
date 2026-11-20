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

"""CPU-only SIMPLE contract, catalog, physical budget and lifecycle checks."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

import pytest

from benchmarks.simple.prepare import export_catalog
from roborsi.agents.config import AgentConfig
from roborsi.evaluation.adapter_service import AdapterFacade
from roborsi.evaluation.benchmark_client import BenchmarkClient
from roborsi.evaluation.benchmark_http import BenchmarkApplication, BenchmarkHttpServer
from roborsi.evaluation.http_artifacts import ArtifactPublisher
from roborsi.evaluation.media import CaseMediaChannel
from roborsi.evaluation.messages import Action, Task, to_envelope
from roborsi.evaluation.service_protocol import RoboRSIRequestError
from robots.simple.benchmark_service import SimpleBenchmark
from robots.simple.catalog import Catalog
from robots.simple.contracts import (
    ACTION_SCHEMA,
    EVALUATION_RULES,
    SIMPLE_REVISION,
    SOURCES,
    reset_instruction,
    validate_actions,
    validate_task,
)
from robots.simple.episode import SimpleEpisode
from robots.simple.roborsi_adapter import SimpleAdapter
from robots.simple.runtime_setup import kit_config, prepare_kit_cleanup
from tests.unit_tests.robots.simple.fakes import FakeRuntime


def test_native_checkout_trust_preserves_revision_and_drift_checks(
    tmp_path, monkeypatch
):
    from robots.simple import native

    # Host/CI trust rules must not pre-authorize this ownership fixture.
    git_config = tmp_path / "gitconfig"
    git_config.write_text("")
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(git_config))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(git_config))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "0")
    monkeypatch.delenv("GIT_CONFIG_PARAMETERS", raising=False)
    root = tmp_path / "checkout"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    source = root / "src/simple/example.py"
    source.parent.mkdir(parents=True)
    source.write_text("# pinned evaluator\n")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Offline test",
            "-c",
            "user.email=offline@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    path, t = catalog(tmp_path)
    monkeypatch.setenv("ROBORSI_SIMPLE_CATALOG", str(path))
    monkeypatch.setenv("ROBORSI_SIMPLE_ROOT", str(root))
    monkeypatch.setenv("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")
    monkeypatch.delenv("OMNI_KIT_ACCEPT_EULA", raising=False)
    monkeypatch.setattr(native, "SIMPLE_REVISION", revision)
    # Simulate a root-owned image checkout without requiring root in CPU CI.
    rejected = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True
    )
    assert rejected.returncode != 0 and b"dubious ownership" in rejected.stderr
    runtime = native.SimpleRuntime(tmp_path / "episode")
    with pytest.raises(ValueError, match="accept the NVIDIA EULA"):
        runtime.start(t)
    source.write_text("# changed evaluator\n")
    with pytest.raises(ValueError, match="tracked source changes"):
        runtime.start(t)
    monkeypatch.setattr(native, "SIMPLE_REVISION", "0" * 40)
    with pytest.raises(ValueError, match="pinned revision"):
        runtime.start(t)


def test_kit_placement_preserves_renderer(tmp_path):
    original = {
        "renderer": "RayTracedLighting",
        "anti_aliasing": 0,
        "extra_args": ["--/existing=true"],
    }
    configured = kit_config(original, tmp_path, 6)
    assert original["extra_args"] == ["--/existing=true"]
    assert configured["renderer"] == original["renderer"]
    assert configured["anti_aliasing"] == original["anti_aliasing"]
    assert configured["active_gpu"] == configured["physics_gpu"] == 6
    args = configured["extra_args"]
    assert args[args.index("--portable-root") + 1] == str(tmp_path / "kit")
    with pytest.raises(ValueError, match="conflicting"):
        kit_config(configured, tmp_path, 6)


def test_start_failure_saved_before_native_close_exits(tmp_path):
    class ExitingRuntime:
        def start(self, task):
            raise RuntimeError("missing native scene asset")

        def close(self):
            raise SystemExit(0)

    episode = SimpleEpisode(tmp_path, ExitingRuntime())
    with pytest.raises(SystemExit):
        episode.start(task(), "case")
    assert (
        "missing native scene asset"
        in (tmp_path / "native/start_failure.txt").read_text()
    )


def task(*, alias="tabletop", multiplier=1, digest="a" * 64):
    spec = SOURCES["tasks"][alias]
    return Task(
        benchmark="simple",
        task_id=f"{alias}/L0/E00",
        seed=20261002,
        instruction="Pick up the object.",
        max_steps=spec["official_steps"] * multiplier,
        native={
            "alias": alias,
            "level": 0,
            "episode": 0,
            "official_steps": spec["official_steps"],
            "controller": spec["controller"],
            "reset_sha256": digest,
            "score_protocol": EVALUATION_RULES["score_protocol"],
        },
        capabilities={
            "action_schema": ACTION_SCHEMA,
            "observation_channels": ["head_stereo_left"],
            "state_dim": 32,
        },
    )


def catalog(tmp_path):
    directory = tmp_path / "catalog"
    (directory / "resets").mkdir(parents=True)
    raw = b'{"private_object_position":[1,2,3]}'
    (directory / "resets/one.json").write_bytes(raw)
    t = task(digest=hashlib.sha256(raw).hexdigest())
    path = directory / "catalog.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "simple_revision": SIMPLE_REVISION,
                "dataset_revision": SOURCES["dataset_revision"],
                "cases": [{"task": t.as_dict(), "reset": "resets/one.json"}],
            }
        )
    )
    return path, t


def action(rows, step=0):
    return Action(
        benchmark="simple",
        episode_id="case",
        step_index=step,
        data={"actions": [[0.0] * 36 for _ in range(rows)]},
    )


@pytest.mark.parametrize("alias", list(SOURCES["tasks"]))
@pytest.mark.parametrize("multiplier", [1, 10])
def test_exact_native_budgets_and_controller(alias, multiplier):
    t = task(alias=alias, multiplier=multiplier)
    assert validate_task(t)["controller"] == t.native["controller"]
    with pytest.raises(ValueError, match="1x or 10x"):
        validate_task(t.model_copy(update={"max_steps": t.max_steps + 1}))


@pytest.mark.parametrize(
    "bad", [float("nan"), float("inf"), 1e300, 10**1000, True, "0"]
)
def test_nonfinite_or_coerced_actions_rejected(bad):
    row = [0.0] * 36
    row[0] = bad
    with pytest.raises(ValueError):
        validate_actions({"actions": [row]}, 800)


def test_catalog_rejects_changed_reset_and_duplicate_identity(tmp_path):
    path, t = catalog(tmp_path)
    c = Catalog(path)
    assert c.tasks() == [t]
    assert "private_object_position" not in str(t.as_dict())
    (path.parent / "resets/one.json").write_text("{}")
    with pytest.raises(ValueError, match="SHA256"):
        c.reset(t)


def test_catalog_rejects_task_forgery_before_reset(tmp_path):
    path, t = catalog(tmp_path)
    c = Catalog(path)
    with pytest.raises(ValueError, match="catalog"):
        c.reset(t.model_copy(update={"instruction": "Another task"}))


def test_native_success_inside_chunk_and_no_privileged_feedback(tmp_path):
    runtime = FakeRuntime()
    publisher = ArtifactPublisher(tmp_path, "case")
    episode = SimpleEpisode(tmp_path, runtime, publisher)
    initial = episode.start(task(), "case")
    assert initial.observation.step_index == 0
    assert "object_pose" not in str(initial.observation.as_dict())
    assert "reward" not in initial.observation.data
    outcome = episode.step(action(8))
    assert outcome.terminated and not outcome.truncated
    assert outcome.observation.step_index == 1
    assert outcome.observation.data["native_env_steps"] == 3
    assert runtime.calls == 3
    assert outcome.reward is None and not outcome.native
    result = episode.finish()
    assert result.success and result.details["first_success_step"] == 3
    assert result.details["success_within_1x"]
    assert {r.ref for r in publisher.list()} >= {
        "native/result.json",
        "native/actions.jsonl",
    }
    episode.close()
    episode.close()
    assert runtime.closed == 1


def test_invalid_chunk_and_stale_sequence_do_not_advance(tmp_path):
    runtime = FakeRuntime()
    episode = SimpleEpisode(tmp_path, runtime)
    episode.start(task(), "case")
    with pytest.raises(ValueError):
        episode.step(action(513))
    with pytest.raises(ValueError):
        episode.step(action(1, step=1))
    assert runtime.calls == 0 and not runtime.submissions
    with pytest.raises(RuntimeError, match="terminated"):
        episode.finish()
    episode.close()


def test_timeout_uses_physical_rows_not_number_of_chunks(tmp_path):
    runtime = FakeRuntime(success=False, end=None)
    episode = SimpleEpisode(tmp_path, runtime)
    episode.start(task(alias="bendpick"), "case")
    out = episode.step(action(399))
    assert not out.terminated and not out.truncated
    with pytest.raises(ValueError, match="budget"):
        episode.step(action(2, step=1))
    assert runtime.calls == 399
    out = episode.step(action(1, step=1))
    assert out.truncated and out.observation.step_index == 2
    assert out.observation.data["native_env_steps"] == 400
    assert not episode.finish().success
    episode.close()


def test_real_process_rpc_media_result_and_cleanup(tmp_path):
    path, t = catalog(tmp_path)
    directory = tmp_path / "episode"
    directory.mkdir()
    adapter = SimpleAdapter(
        directory,
        ArtifactPublisher(directory, "case"),
        factory="tests.unit_tests.robots.simple.fakes:make_runtime",
        timeout_s=20,
    )
    try:
        initial = adapter.start(t, "case")
        assert initial.observation.media
        end = adapter.step(action(10))
        assert end.terminated
        result = adapter.finish()
        assert result.details["native_env_steps"] == 3
        assert (directory / result.source).is_file()
        screenshots = directory / "native/kit/data/documents/Kit/shared/screenshots"
        screenshots.mkdir(parents=True)
        screenshots.chmod(0)
    finally:
        adapter.close()
    assert adapter.daemon.poll() is not None
    assert screenshots.stat().st_mode & 0o700 == 0o700


@pytest.mark.parametrize("failure", ["stop", "kit", "rpc"])
def test_close_retries_failed_cleanup_through_facade(tmp_path, monkeypatch, failure):
    adapter = SimpleAdapter(
        tmp_path,
        ArtifactPublisher(tmp_path, "case"),
        factory="tests.unit_tests.robots.simple.fakes:make_runtime",
        timeout_s=20,
    )
    facade = AdapterFacade(adapter)
    screenshots = tmp_path / "native/kit/data/documents/Kit/shared/screenshots"
    stop = adapter.daemon.stop
    rpc_close = adapter.rpc.close
    calls = {"stop": 0, "kit": 0, "rpc": 0}

    def stop_once(*, timeout):
        calls["stop"] += 1
        if failure == "stop" and calls["stop"] == 1:
            raise OSError("injected stop failure")
        stop(timeout=timeout)

    def cleanup_once(directory):
        calls["kit"] += 1
        if failure == "kit" and calls["kit"] == 1:
            raise PermissionError("injected kit failure")
        prepare_kit_cleanup(directory)

    def close_rpc_once():
        calls["rpc"] += 1
        if failure == "rpc" and calls["rpc"] == 1:
            raise OSError("injected rpc failure")
        rpc_close()

    def unavailable(*args, **kwargs):
        raise TimeoutError("graceful worker shutdown unavailable")

    try:
        wire = json.dumps(
            to_envelope(
                task(), benchmark="simple", episode_id="case", request_id="start"
            ).as_dict()
        )
        facade.start(wire)
        screenshots.mkdir(parents=True)
        screenshots.chmod(0)
        monkeypatch.setattr(adapter.daemon, "stop", stop_once)
        monkeypatch.setattr(
            "robots.simple.roborsi_adapter.prepare_kit_cleanup", cleanup_once
        )
        monkeypatch.setattr(adapter.rpc, "close", close_rpc_once)
        if failure == "stop":
            # Leave a real worker alive so a skipped retry cannot appear to pass.
            monkeypatch.setattr(adapter.rpc, "call", unavailable)
        with pytest.raises(OSError, match=f"injected {failure} failure"):
            facade.close_episode()
        if failure == "stop":
            assert adapter.daemon.poll() is None
        if failure in {"stop", "kit"}:
            assert screenshots.stat().st_mode & 0o777 == 0
        with pytest.raises(RuntimeError, match="closed"):
            adapter.step(action(1))
        retry_start = json.loads(wire)
        retry_start["request_id"] = "new-start"
        with pytest.raises(RoboRSIRequestError, match="one episode"):
            facade.start(json.dumps(retry_start))
        assert facade.close_episode() == {"ok": True}
        assert adapter.daemon.poll() is not None
        assert screenshots.stat().st_mode & 0o700 == 0o700
        assert calls[failure] == 2
        completed = dict(calls)
        assert facade.close_episode() == {"ok": True}
        assert calls == completed
    finally:
        # Test failures must not leave a real child or mode-000 directory behind.
        stop(timeout=5)
        rpc_close()
        prepare_kit_cleanup(tmp_path)


def test_kit_cleanup_does_not_follow_directory_links(tmp_path):
    outside = tmp_path / "outside"
    screenshots = outside / "data/documents/Kit/shared/screenshots"
    screenshots.mkdir(parents=True)
    screenshots.chmod(0)
    episode = tmp_path / "episode"
    (episode / "native").mkdir(parents=True)
    (episode / "native/kit").symlink_to(outside, target_is_directory=True)
    try:
        prepare_kit_cleanup(episode)
        assert screenshots.stat().st_mode & 0o777 == 0
    finally:
        screenshots.chmod(0o700)


def test_http_downloads_survive_private_server_deletion(tmp_path, monkeypatch):
    path, t = catalog(tmp_path)
    # Framework registration is a separate owner PR. Only version discovery is
    # faked here; the SIMPLE adapter/process/RPC/HTTP/media are real.
    monkeypatch.setattr(
        "robots.simple.benchmark_service.benchmark_source_digest",
        lambda root, name: "b" * 64,
    )
    provider = SimpleBenchmark(
        path, factory="tests.unit_tests.robots.simple.fakes:make_runtime"
    )
    app = BenchmarkApplication(provider, tmp_path / "private")
    server = BenchmarkHttpServer(("127.0.0.1", 0), app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = BenchmarkClient(f"http://127.0.0.1:{server.server_port}")
    media = CaseMediaChannel(tmp_path / "external-agent/media")
    try:
        assert client.tasks() == [t]
        assert client.runtime_info(timeout_s=3).benchmark_commit == SIMPLE_REVISION
        initial = client.start(t, "case", timeout_s=10)
        client.collect_observation(initial.observation, media, timeout_s=3)
        end = client.step(action(8), timeout_s=5)
        client.collect_observation(end.observation, media, timeout_s=3)
        assert end.terminated
        native = client.finish(timeout_s=5)
        client.collect_native(native, tmp_path / "external-agent", timeout_s=3)
        client.collect_artifacts(tmp_path / "external-agent", timeout_s=3)
        client.close()
        client.delete("case")
        assert not list(app.root.iterdir())
        media.verify_observation(initial.observation)
        media.verify_observation(end.observation)
        assert (tmp_path / "external-agent/native/result.json").exists()
    finally:
        app.close()
        server.shutdown()
        server.server_close()
        thread.join(3)


def test_native_worker_does_not_inherit_model_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder-not-a-real-credential")
    adapter = SimpleAdapter(
        tmp_path,
        ArtifactPublisher(tmp_path, "case"),
        factory="tests.unit_tests.robots.simple.fakes:make_runtime",
        timeout_s=20,
    )
    try:
        assert "OPENAI_API_KEY" not in adapter.daemon.subprocess_env
    finally:
        adapter.close()


@pytest.mark.parametrize("terminal", [False, True])
def test_http_smoke_stops_after_three_rows_and_keeps_downloads(
    tmp_path, monkeypatch, terminal
):
    from benchmarks.simple.smoke_http import check

    path, t = catalog(tmp_path)
    monkeypatch.setattr(
        "robots.simple.benchmark_service.benchmark_source_digest",
        lambda root, name: "b" * 64,
    )
    factory = "make_runtime" if terminal else "make_nonterminal_runtime"
    provider = SimpleBenchmark(
        path, factory=f"tests.unit_tests.robots.simple.fakes:{factory}"
    )
    app = BenchmarkApplication(provider, tmp_path / "private")
    server = BenchmarkHttpServer(("127.0.0.1", 0), app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output = tmp_path / "downloaded"
    try:
        report = check(
            f"http://127.0.0.1:{server.server_port}", t.task_id, output, "b" * 64
        )
        assert report["passed"] and report["case_deleted"]
        assert report["native_rows"] == 3 and report["model_requests"] == 0
        assert report["task_success"] is None
        assert report["native_finish_exercised"] is terminal
        assert len(list(output.glob("observation_*.json"))) == 4
        assert not list(app.root.iterdir())
    finally:
        app.close()
        server.shutdown()
        server.server_close()
        thread.join(3)


@pytest.mark.parametrize("early_success", [False, True])
def test_terminal_check_verifies_finish_score_trace_and_cleanup(
    tmp_path, monkeypatch, early_success
):
    from benchmarks.simple.check_terminal import check

    path, t = catalog(tmp_path)
    monkeypatch.setattr(
        "robots.simple.benchmark_service.benchmark_source_digest",
        lambda root, name: "b" * 64,
    )
    factory = "make_runtime" if early_success else "make_nonterminal_runtime"
    provider = SimpleBenchmark(
        path, factory=f"tests.unit_tests.robots.simple.fakes:{factory}"
    )
    app = BenchmarkApplication(provider, tmp_path / "private")
    server = BenchmarkHttpServer(("127.0.0.1", 0), app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output = tmp_path / "downloaded"
    try:
        report = check(
            f"http://127.0.0.1:{server.server_port}", t.task_id, output, "b" * 64
        )
        assert report["passed"] and report["case_deleted"]
        assert report["native_rows"] == (3 if early_success else t.max_steps)
        assert report["task_success"] is early_success
        assert report["score"] == float(early_success)
        assert report["truncated"] is not early_success
        assert report["native_finish_exercised"]
        assert report["score_matches_action_trace"]
        assert report["model_requests"] == 0
        assert report["state_after_finish"] == "completed"
        # HTTP close preserves an already completed outcome; this check calls
        # close before deleting the completed case and its downloaded artifacts.
        assert report["state_after_close"] == "completed"
        assert not list(app.root.iterdir())
        result = json.loads((output / "native/result.json").read_text())
        assert result["success"] is early_success
    finally:
        app.close()
        server.shutdown()
        server.server_close()
        thread.join(3)


def test_reset_language_is_separate_from_episode_instruction():
    # Released XMove L0/E00 has a stale bend template but a walk task prompt.
    state = {
        "dr_state_dict": {
            "language": "bend to pick up the {}",
            "target": {"name": "cracker box"},
        }
    }
    prompt = "move forward to pick up the cracker box"
    assert reset_instruction(state) == "bend to pick up the cracker box"
    assert reset_instruction(state) != prompt
    assert state["dr_state_dict"]["language"] == "bend to pick up the {}"
    state["dr_state_dict"]["language"] = "pick up the cracker box"
    assert reset_instruction(state) == "pick up the cracker box"
    for template in (None, "", "{0.__class__}", "{} {}", "{target}"):
        state["dr_state_dict"]["language"] = template
        with pytest.raises(ValueError, match="instruction template"):
            reset_instruction(state)
    with pytest.raises(ValueError, match="lacks language"):
        reset_instruction({})


def test_export_checks_archive_and_preserves_all_resets(tmp_path, monkeypatch):
    spec = SOURCES["tasks"]["tabletop"]
    archive = tmp_path / spec["test_archive"]["path"]
    archive.parent.mkdir()
    with zipfile.ZipFile(archive, "w") as z:
        for level in range(3):
            rows = [
                {
                    "episode_index": i,
                    "tasks": ["Pick up the object."],
                    "environment_config": json.dumps(
                        {"dr_state_dict": {"scene": {"uid": "102344280"}}}
                    ),
                }
                for i in range(10)
            ]
            z.writestr(
                f"{spec['env_id']}/dr-level-{level}/meta/episodes.jsonl",
                "\n".join(json.dumps(r) for r in rows),
            )
    raw = archive.read_bytes()
    monkeypatch.setitem(
        spec,
        "test_archive",
        dict(
            spec["test_archive"], bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()
        ),
    )
    path = export_catalog(tmp_path, tmp_path / "out", ["tabletop"], multiplier=10)
    c = Catalog(path)
    assert len(c.tasks()) == 30
    assert all(t.max_steps == 8000 for t in c.tasks())
    assert c.reset(c.tasks()[0])["dr_state_dict"]["scene"]["uid"] == "scene3"
    archive.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="archive"):
        export_catalog(tmp_path, tmp_path / "bad", ["tabletop"])
    assert not (tmp_path / "bad").exists()


def test_standard_embodied_agent_preset_loads_without_calling_provider():
    config = AgentConfig.from_file(
        Path(__file__).resolve().parents[4] / "robots/simple/agent.yaml"
    )
    assert config.factory == "roborsi.agents.config:make_embodied_agent"
    assert config.document["runtime"]["trace"]["mode"] == "full"
    assert config.document["agent"]["planner"] == "api"


def test_service_imports_without_simulator_or_agent_sdk(tmp_path):
    code = """
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'isaacsim','mujoco','torch','openai','pydantic_ai','anthropic','mcp'}:
   raise AssertionError('unexpected optional import: '+fullname)
sys.meta_path.insert(0,Guard())
import robots.simple.benchmark_service
import robots.simple.worker
import robots.simple.native
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stderr
