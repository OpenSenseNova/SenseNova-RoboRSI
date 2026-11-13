# Copyright 2026 The HEART Authors.
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

"""Exercise the real shared CLI backend with an offline file-protocol process."""

import json
import shutil
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from roborsi.agents.config import AgentConfig
from roborsi.embodied_agent import EmbodiedAgent
from roborsi.evaluation.media import CaseMediaChannel
from roborsi.evaluation.messages import (
    NativeResult,
    Provenance,
    StartResult,
    StepResult,
)
from roborsi.evaluation.runner import CaseRunner
from roborsi.evaluation.store import ResultStore
from robots.dexjoco.agents.baseline.factory import make_agent
from robots.dexjoco.agents.baseline.session import CliSession
from robots.dexjoco.benchmark_service import make_task
from tests.unit_tests.robots.dexjoco.test_policy_lifecycle import observation

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def cli_agent(tmp_path, monkeypatch, upstream, request):
    monkeypatch.setenv("ROBORSI_DEXJOCO_AGENT_ASSETS", str(tmp_path / "assets"))
    source = tmp_path / "source"
    shutil.copytree(ROOT / "robots/dexjoco", source / "robots/dexjoco")
    preset = source / "robots/dexjoco/agent.yaml"
    if getattr(request, "param", False):
        import yaml

        folder = source / "robots/dexjoco/agents/baseline"
        (folder / "custom.md").write_text("CUSTOM_POLICY: observe before grasping.")
        (folder / "skill.md").write_text(
            "CUSTOM_SKILL: verify the grasp in the wrist image."
        )
        preset.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "extends": "./agents/baseline/agent.yaml",
                    "prompt_file": "agents/baseline/custom.md",
                    "skills": ["agents/baseline/skill.md"],
                }
            )
        )
    config = AgentConfig.from_file(preset)
    (tmp_path / "case").mkdir()
    agent = make_agent(tmp_path / "case", config=config)
    agent.task = make_task(upstream, "water_plant", 100000)
    agent.task = agent.task.model_copy(update={"max_steps": 20})
    agent.dual = False
    agent.standard_limit = 2
    monkeypatch.setattr(agent, "_verify_payload", lambda: None)
    # The model executable is fake; the planner, launcher, stdin, file transport,
    # events, parent-death behavior and lifecycle are real. No model is called.
    cli = tmp_path / "fake-cli"
    cli.write_text(
        "#!"
        + sys.executable
        + "\n"
        + r"""
import ctypes, json, os, signal, sys, time
from pathlib import Path
if "--version" in sys.argv:
    print("codex-cli 0.154.0")
    raise SystemExit
assert ctypes.CDLL(None).prctl(1, signal.SIGKILL) == 0
root = Path.cwd()
(root / "seen-prompt.txt").write_text(sys.stdin.read())
(root / "seen-argv.json").write_text(json.dumps(sys.argv[1:]))
home = Path(os.environ["CODEX_HOME"])
(home / "sessions").mkdir()
(home / "sessions/turn.jsonl").write_text(json.dumps({
    "type": "turn_context", "payload": {"model": "gpt-6-astra"}
}) + "\n")
print(json.dumps({"type": "thread.started", "thread_id": "offline-test"}), flush=True)
last = None
while not (root / "input/episode_done.json").exists() or (root / "ignore-done").exists():
    source = root / "input/observation.json"
    if source.exists():
        obs = json.loads(source.read_text())
        if obs["decision"] != last:
            last = obs["decision"]
            request = {"decision": last, "payload": {"actions": [[0.0] * 22] * min(2, obs["remaining_steps"])}}
            transport = Path(os.environ["TEST_TRANSPORT"])
            temp = transport / "request.tmp"
            temp.write_text(json.dumps(request))
            temp.replace(transport / "request.json")
    time.sleep(.01)
failure = root / "fail-on-done"
if failure.exists():
    if failure.read_text() == "turn_failed":
        print(json.dumps({"type": "turn.failed", "error": {"message": "offline provider failed"}}), flush=True)
    raise SystemExit(1)
print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 7, "output_tokens": 3}}), flush=True)
"""
    )
    cli.chmod(0o700)
    auth = tmp_path / "auth.json"
    auth.write_text("{}")
    monkeypatch.setenv("ROBORSI_DEXJOCO_CODEX", str(cli))
    monkeypatch.setenv("ROBORSI_DEXJOCO_AUTH", str(auth))
    monkeypatch.setenv("ROBORSI_DEXJOCO_POLICY_PYTHON", str(tmp_path / "numerical"))
    wrapper = tmp_path / "fake-bwrap"
    wrapper.write_text(
        "#!"
        + sys.executable
        + "\n"
        + "import os, sys\n"
        + "if sys.argv[2] != '/opt/codex': raise SystemExit(0)\n"
        + "os.environ['CODEX_HOME'] = sys.argv[3]\n"
        + "os.environ['TEST_TRANSPORT'] = "
        + repr(str(agent.private / "action-transport"))
        + "\n"
        + "os.execv("
        + repr(str(cli))
        + ", ["
        + repr(str(cli))
        + "] + sys.argv[4:])\n"
    )
    wrapper.chmod(0o700)
    monkeypatch.setattr(agent.ref.isolation, "probe", lambda *a: None)
    # Supply each session's home as the first argument after /opt/codex.
    # Production mounts use real bubblewrap; this stand-in tests argv forwarding.

    def mounts(workspace, home, interface, binary, **kwargs):
        bridge = tmp_path / (home.parent.name + "-bridge")
        # Use Python for portable argument slicing, without shell expansion.
        bridge.write_text(
            "#!"
            + sys.executable
            + "\nimport os,sys\nos.execv("
            + repr(str(wrapper))
            + ", ["
            + repr(str(wrapper))
            + "] + sys.argv[1:3] + ["
            + repr(str(home))
            + "] + sys.argv[3:])\n"
        )
        bridge.chmod(0o700)
        return [str(bridge)]

    monkeypatch.setattr(agent.ref.isolation, "mounts", mounts)
    yield agent, config
    expected_error = getattr(agent, "_expected_close_error", None)
    if expected_error:
        with pytest.raises(RuntimeError, match=expected_error):
            agent.close()
    else:
        agent.close()


def wait_ready(session):
    deadline = time.monotonic() + 10
    while not (session.agent.workspace / "seen-prompt.txt").exists():
        if session.poll() is not None or time.monotonic() > deadline:
            raise AssertionError("offline CLI did not start")
        time.sleep(0.01)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux parent-death semantics")
def test_model_process_survives_request_thread(cli_agent):
    agent, _ = cli_agent
    errors = []

    def request():
        try:
            agent._start()
            wait_ready(agent.session)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=request)
    thread.start()
    thread.join(timeout=15)
    assert not thread.is_alive()
    assert not errors
    session = agent.session
    time.sleep(0.1)
    assert session.poll() is None
    agent.close()
    assert session.episode.done
    assert session.record["cleanup"]["status"] == "complete"
    assert session.record["stop_reason"] == "completed"
    assert session.record["tokens"]["total_tokens"] == 10
    assert not (session.home / "auth.json").exists()
    assert session.close() == session.record


def test_backend_cancellation_removes_process_and_auth(cli_agent, monkeypatch):
    agent, _ = cli_agent
    (agent.workspace / "ignore-done").touch()
    monkeypatch.setattr(CliSession, "FINISH_GRACE_S", 0.05)
    agent._start()
    session = agent.session
    wait_ready(session)
    agent.close()
    assert session.episode.done
    assert session.record["stop_reason"] == "closed"
    assert session.record["cleanup"]["status"] == "complete"
    assert session.record["tokens"]["status"] != "complete"
    assert not (session.home / "auth.json").exists()


def test_real_backend_file_feedback_and_continuation(cli_agent, monkeypatch):
    agent, _ = cli_agent
    calls = []
    start_episode = EmbodiedAgent.start_episode

    def start(embodied, prompt, **kwargs):
        calls.append((embodied, prompt, kwargs))
        return start_episode(embodied, prompt, **kwargs)

    monkeypatch.setattr(EmbodiedAgent, "start_episode", start)
    agent.act(observation(agent, 0))
    first = agent.session
    assert (first.call_dir / "input.txt").read_text() == agent.prompts["initial"]
    assert (agent.workspace / "seen-prompt.txt").read_text() == agent.prompts["initial"]
    agent.act(observation(agent, 1))
    assert agent.session is first
    agent.act(observation(agent, 2))
    second = agent.session
    assert second is not first and first.episode.done
    assert (agent.workspace / "seen-prompt.txt").read_text() == agent.prompts[
        "continuation"
    ]
    assert (second.call_dir / "input.txt").read_text() == agent.prompts["continuation"]
    assert not (first.home / "auth.json").exists()
    agent.close()
    assert len(agent.records) == 2
    assert [item[1] for item in calls] == [
        agent.prompts["initial"],
        agent.prompts["continuation"],
    ]
    assert all(item[2] == {"system_prompt": ""} for item in calls)
    assert all(isinstance(item[0], EmbodiedAgent) for item in calls)
    assert all(
        r["backend"] == "roborsi.embodied_agent.codex_cli" for r in agent.records
    )
    assert all(r["cleanup"]["status"] == "complete" for r in agent.records)
    argv = json.loads((agent.workspace / "seen-argv.json").read_text())
    assert argv[0] == "exec" and argv[-1] == "-"
    assert "--json" in argv and "--ignore-user-config" in argv
    assert 'model_reasoning_effort="max"' in argv
    assert not any("mcp_servers.roborsi" in arg for arg in argv)


@pytest.mark.parametrize("cli_agent", [True], indirect=True)
def test_configured_prompt_and_skills_reach_actual_cli_stdin_both_phases(cli_agent):
    agent, config = cli_agent
    for step in (0, 1, 2):
        agent.act(observation(agent, step))
        if step in (0, 2):
            captured = (agent.workspace / "seen-prompt.txt").read_text()
            assert captured == (agent.session.call_dir / "input.txt").read_text()
            assert captured == agent.prompts[agent.phase]
            assert "CUSTOM_POLICY: observe before grasping." in captured
            assert "CUSTOM_SKILL: verify the grasp in the wrist image." in captured
            assert "The pinned public robot model is at" not in captured
            assert "input/observation.json" in captured
    assert agent.config_fingerprint == config.fingerprint
    agent.close()


def test_startup_failure_removes_auth(cli_agent, monkeypatch):
    agent, _ = cli_agent

    def fail(*args):
        raise RuntimeError("isolation probe failed")

    monkeypatch.setattr(agent.ref.isolation, "probe", fail)
    with pytest.raises(RuntimeError, match="isolation probe failed"):
        agent._start()
    assert agent.session is None
    assert not (agent.private / "codex-home/auth.json").exists()


@pytest.mark.parametrize("failure", ["turn_failed", "process_failed"])
def test_terminal_cli_failure_is_propagated_after_last_action(cli_agent, failure):
    agent, _ = cli_agent
    (agent.workspace / "fail-on-done").write_text(failure)
    agent.act(observation(agent, 0))
    session = agent.session
    agent._expected_close_error = failure
    terminal = StepResult(
        observation=observation(agent, 1),
        terminated=True,
        truncated=False,
        termination_reason="native_success",
        reward=1,
    )
    with pytest.raises(RuntimeError, match=failure):
        agent.observe_end(terminal)
    assert session.episode.done
    assert session.record["stop_reason"] == failure
    assert session.record["error_kind"] == "infrastructure"
    assert session.record["cleanup"]["status"] == "complete"
    assert not (session.home / "auth.json").exists()
    assert json.loads((agent.private / "sessions.json").read_text()) == [session.record]
    with pytest.raises(RuntimeError, match=failure):
        agent.close()
    assert agent.session is session and not agent.closed
    assert len(agent.records) == 1


def test_runner_cannot_score_native_success_after_final_cli_failure(
    cli_agent, tmp_path
):
    agent, _ = cli_agent
    store = ResultStore(tmp_path / "results")
    directory = store.claim("run", "case", agent.task, Provenance(), episode_id="one")
    agent.media = CaseMediaChannel(directory / "media")
    (agent.workspace / "fail-on-done").write_text("turn_failed")
    agent._expected_close_error = "turn_failed"
    start = StartResult(task=agent.task, observation=observation(agent, 0))
    terminal = StepResult(
        observation=observation(agent, 1),
        terminated=True,
        truncated=False,
        termination_reason="native_success",
        reward=1,
    )
    scores_read = []

    def finish(**kwargs):
        scores_read.append(True)
        return NativeResult(task=agent.task, episode_id="one", success=True, score=1)

    client = SimpleNamespace(
        reset=lambda *args, **kwargs: None,
        act=lambda value, **kwargs: agent.act(value),
        observe_end=lambda value, **kwargs: agent.observe_end(value),
        close=agent.close,
    )
    adapter = SimpleNamespace(
        start=lambda *args, **kwargs: start,
        step=lambda *args, **kwargs: terminal,
        collect_observation=lambda *args, **kwargs: None,
        finish=finish,
        close=lambda: None,
    )
    result = CaseRunner(store).run(
        "run",
        "case",
        "one",
        agent.task,
        Provenance(),
        client,
        adapter,
        preclaimed=True,
        media_channel=agent.media,
    )
    assert result.status == "agent_error"
    assert result.score is None and result.success is None
    assert result.error.details["stage"] == "agent.observe_end"
    assert "turn_failed" in result.error.message
    assert scores_read == []
    assert agent.session.episode.done
    assert not (agent.session.home / "auth.json").exists()


@pytest.mark.parametrize("direct_session_close", [False, True])
def test_done_marker_write_failure_cleans_up_and_close_can_retry(
    cli_agent, monkeypatch, direct_session_close
):
    agent, _ = cli_agent
    agent._start()
    session = agent.session
    wait_ready(session)
    write_json = agent.ref.write_json

    def fail_marker(path, value):
        if Path(path).name == "episode_done.json":
            raise OSError("injected full disk at done marker")
        return write_json(path, value)

    monkeypatch.setattr(agent.ref, "write_json", fail_marker)
    with pytest.raises(OSError, match="full disk at done marker"):
        (session if direct_session_close else agent).close()
    assert session.episode.closed
    assert session.episode.done
    assert session.record["cleanup"]["status"] == "complete"
    assert session.record["owner_requested_cancel"]
    assert not (session.home / "auth.json").exists()
    assert agent.session is session and not agent.closed
    monkeypatch.setattr(agent.ref, "write_json", write_json)
    agent.close()
    assert agent.closed and agent.session is None
    assert agent.records == [session.record]
    assert json.loads((agent.private / "sessions.json").read_text()) == agent.records


def test_session_record_write_failure_retains_session_for_retry(cli_agent, monkeypatch):
    agent, _ = cli_agent
    agent._start()
    session = agent.session
    wait_ready(session)
    write_json = agent.ref.write_json

    def fail_record(path, value):
        if Path(path).name == "session_result.json":
            raise OSError("injected session record write failure")
        return write_json(path, value)

    monkeypatch.setattr(agent.ref, "write_json", fail_record)
    with pytest.raises(OSError, match="session record write failure"):
        agent.close()
    assert session.episode.done
    assert not (session.home / "auth.json").exists()
    assert agent.session is session and not agent.closed
    monkeypatch.setattr(agent.ref, "write_json", write_json)
    agent.close()
    assert agent.closed and agent.session is None
    assert len(agent.records) == 1
    assert (
        json.loads((session.private / "session_result.json").read_text())
        == session.record
    )


def test_final_observation_write_failure_still_closes_session(cli_agent, monkeypatch):
    agent, _ = cli_agent
    agent.act(observation(agent, 0))
    session = agent.session
    terminal = StepResult(
        observation=observation(agent, 1),
        terminated=True,
        truncated=False,
        termination_reason="native_success",
        reward=1,
    )

    def fail_save(observation):
        raise OSError("injected terminal observation write failure")

    monkeypatch.setattr(agent, "_save", fail_save)
    with pytest.raises(OSError, match="terminal observation write failure"):
        agent.observe_end(terminal)
    assert session.episode.done
    assert not (session.home / "auth.json").exists()
    assert agent.closed and agent.session is None


def test_cleanup_failure_is_propagated_even_after_owner_cancellation(
    cli_agent, monkeypatch
):
    from roborsi.planner import codex_cli

    agent, _ = cli_agent
    (agent.workspace / "ignore-done").touch()
    monkeypatch.setattr(CliSession, "FINISH_GRACE_S", 0.05)
    terminate = codex_cli._terminate_process

    def fail_cleanup(process, deadline):
        terminate(process, deadline)
        raise OSError("injected process cleanup failure")

    agent._start()
    session = agent.session
    wait_ready(session)
    monkeypatch.setattr(codex_cli, "_terminate_process", fail_cleanup)
    agent._expected_close_error = "cleanup failed"
    with pytest.raises(RuntimeError, match="cleanup failed"):
        agent.close()
    assert session.episode.done
    assert session.record["stop_reason"] == "cleanup_failed"
    assert session.record["owner_requested_cancel"]
    assert not (session.home / "auth.json").exists()
    assert agent.records == [session.record]
