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

"""Public Agent ownership of real offline CLI file-protocol subprocesses."""

import json
import os
import sys
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from roborsi import embodied_agent as module
from roborsi.embodied_agent import EmbodiedAgent, LocalToolSpec, McpServer
from roborsi.planner.base import PlannerResult
from roborsi.planner.codex_cli_config import CodexCliConfig

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux CLI runtime")


@pytest.fixture
def external_agent(tmp_path):
    executable = tmp_path / "offline-cli"
    executable.write_text(
        f"#!{sys.executable}\n"
        + r"""
import ctypes, json, os, signal, sys, time
from pathlib import Path
if sys.argv[1:] == ["--version"]:
    print("codex-cli test.external")
    raise SystemExit
assert ctypes.CDLL(None).prctl(1, signal.SIGKILL) == 0
Path("stdin.txt").write_text(sys.stdin.read())
Path("argv.json").write_text(json.dumps(sys.argv[1:]))
Path("pid").write_text(str(os.getpid()))
print(json.dumps({"type": "thread.started", "thread_id": "external-test"}), flush=True)
while not Path("done").exists():
    if Path("feedback.txt").exists():
        Path("action.txt").write_text(Path("feedback.txt").read_text())
    time.sleep(.01)
if Path("failure").exists():
    print(json.dumps({"type": "turn.failed", "error": {"message": "offline failure"}}), flush=True)
    raise SystemExit(1)
print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 7, "output_tokens": 3}}), flush=True)
"""
    )
    executable.chmod(0o700)
    return EmbodiedAgent(
        mcp_servers=(),
        output_dir=tmp_path / "output",
        planner="codex_cli",
        model="test-model",
        max_turns=None,
        max_tokens=None,
        planner_timeout_s=0,
        working_dir=tmp_path,
        codex_cli=CodexCliConfig(
            executable=str(executable),
            expected_version="codex-cli test.external",
            tools="external",
        ),
    )


def wait_file(episode, path):
    deadline = time.monotonic() + 10
    while not path.exists():
        if episode.done:
            raise AssertionError(
                f"episode ended before file feedback: {episode.wait()}"
            )
        if time.monotonic() > deadline:
            raise AssertionError(f"no external feedback at {path.name}")
        time.sleep(0.01)


def test_external_episode_preserves_text_and_exchanges_files(
    external_agent, tmp_path, monkeypatch
):
    def no_toolkit(*args, **kwargs):
        raise AssertionError("external sessions must not construct an MCP toolkit")

    monkeypatch.setattr(module, "_McpToolkit", no_toolkit)
    prompt = "original protocol\n\nobserve before grasping\n"
    with external_agent.start_episode(prompt, system_prompt="") as episode:
        wait_file(episode, tmp_path / "pid")
        assert (tmp_path / "stdin.txt").read_text() == prompt
        (tmp_path / "feedback.txt").write_text("observation-1")
        wait_file(episode, tmp_path / "action.txt")
        assert (tmp_path / "action.txt").read_text() == "observation-1"
        (tmp_path / "done").touch()
        result = episode.wait(timeout=10)
        assert result.error is None and result.stop_reason == "completed"
        assert result.stats["llm_usage"]["total_tokens"] == 10
    assert not any(
        "mcp_servers" in value
        for value in json.loads((tmp_path / "argv.json").read_text())
    )
    runtime = json.loads(
        (external_agent.output_dir / "codex-cli/call-0001/runtime.json").read_text()
    )
    assert runtime["tools"] == "external"
    assert runtime["cleanup"]["status"] == "complete"


def test_external_worker_survives_starting_request_thread(external_agent, tmp_path):
    sessions = []

    def request():
        sessions.append(external_agent.start_episode("task", system_prompt=""))

    thread = threading.Thread(target=request)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive() and len(sessions) == 1
    episode = sessions[0]
    try:
        wait_file(episode, tmp_path / "pid")
        time.sleep(0.05)
        assert not episode.done
        os.kill(int((tmp_path / "pid").read_text()), 0)
        episode.cancel()
        result = episode.wait(timeout=10)
        assert result.error_kind == "cancelled" and result.stop_reason == "closed"
    finally:
        episode.close(timeout=10)
    with pytest.raises(ProcessLookupError):
        os.kill(int((tmp_path / "pid").read_text()), 0)


def test_external_failure_remains_visible_after_terminal_feedback(
    external_agent, tmp_path
):
    with external_agent.start_episode("task", system_prompt="") as episode:
        wait_file(episode, tmp_path / "pid")
        (tmp_path / "failure").touch()
        (tmp_path / "done").touch()
        result = episode.wait(timeout=10)
        assert result.error_kind == "infrastructure"
        assert result.stop_reason == "turn_failed"
        assert "offline failure" in result.error


def test_external_run_loads_skills_without_mcp(external_agent, tmp_path):
    skill = tmp_path / "skill.md"
    skill.write_text("# Grasp\nCheck the wrist camera.")
    (tmp_path / "done").touch()
    result = external_agent.run("task", system_prompt="rules", skills=[skill])
    assert result.error is None
    text = (tmp_path / "stdin.txt").read_text()
    assert "rules" in text and "Check the wrist camera." in text and "task" in text


@pytest.mark.parametrize("kind", ["mcp", "local", "deferred", "finish"])
def test_external_protocol_rejects_mixed_tool_modes(external_agent, kind):
    if kind == "mcp":
        external_agent.mcp_servers = [
            McpServer(name="robot", url="http://127.0.0.1:1/mcp")
        ]
    elif kind == "local":
        external_agent.local_tools = [LocalToolSpec("move", "Move", {"type": "object"})]
    with pytest.raises(ValueError, match="external protocol"):
        if kind in {"mcp", "local"}:
            external_agent.run("task", system_prompt="")
        else:
            external_agent.start_episode(
                "task",
                system_prompt="",
                deferred_tools=["move"] if kind == "deferred" else (),
                include_finish=kind == "finish",
            )


def test_mcp_sessions_still_require_deferred_tools(external_agent):
    external_agent.codex_cli = replace(external_agent.codex_cli, tools="mcp")
    with pytest.raises(ValueError, match="at least one deferred tool"):
        external_agent.start_episode("task", system_prompt="rules")


def test_external_preflight_failure_is_observable_and_releases_worker(
    external_agent, tmp_path
):
    external_agent.codex_cli = replace(
        external_agent.codex_cli, expected_version="other"
    )
    with external_agent.start_episode("task", system_prompt="") as episode:
        with pytest.raises(RuntimeError, match="embodied episode failed") as failure:
            episode.wait(timeout=10)
        assert "version" in str(failure.value.__cause__).lower()
        assert episode.done
    assert not (tmp_path / "pid").exists()
    assert not external_agent._run_lock.locked()


def test_external_close_timeout_remains_retryable(external_agent, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def solve(**kwargs):
        entered.set()
        assert release.wait(10)
        return PlannerResult()

    monkeypatch.setattr(
        module,
        "build_planner",
        lambda *args, **kwargs: SimpleNamespace(preflight=lambda: None, solve=solve),
    )
    episode = external_agent.start_episode("task", system_prompt="")
    try:
        assert entered.wait(5)
        with pytest.raises(TimeoutError, match="cleanup is still running"):
            episode.close(timeout=0)
        assert episode.closed and not episode.done
        with pytest.raises(RuntimeError, match="already has an active episode"):
            external_agent.run("second", system_prompt="")
    finally:
        release.set()
        episode.close(timeout=5)
    assert episode.done and not external_agent._run_lock.locked()
    assert external_agent.run("next", system_prompt="").error is None
