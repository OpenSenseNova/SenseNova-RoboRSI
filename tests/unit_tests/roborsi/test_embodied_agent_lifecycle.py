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

"""Failure recovery and concurrent ownership of the public Agent facade."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest

from roborsi import embodied_agent as module
from roborsi.embodied_agent import EmbodiedAgent, LocalToolSpec
from roborsi.planner.base import PlannerResult


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "create_trace", lambda *a, **kw: None)
    monkeypatch.setattr(
        module,
        "build_planner",
        lambda *a, **kw: SimpleNamespace(solve=lambda **kw: PlannerResult()),
    )
    return EmbodiedAgent(
        mcp_servers=[],
        local_tools=[
            LocalToolSpec("observe", "Observe", {"type": "object"}, lambda _: {})
        ],
        output_dir=tmp_path,
    )


def test_duplicate_tools_can_be_fixed_and_retried_on_same_agent(agent):
    agent.local_tools = (*agent.local_tools, *agent.local_tools)
    with pytest.raises(ValueError, match="unique"):
        agent.run("task", system_prompt="rules")
    agent.local_tools = agent.local_tools[:1]
    assert agent.run("task", system_prompt="rules").error is None


@pytest.mark.parametrize("failure", ["state", "start", "solve", "close"])
def test_failure_releases_lock_and_closes_only_constructed_toolkit(
    agent, monkeypatch, failure
):
    closed = []
    original_close = module._McpToolkit.close

    def close(toolkit):
        closed.append(toolkit)
        original_close(toolkit)
        if failure == "close":
            raise RuntimeError("close failed")

    def fail(*a, **kw):
        raise RuntimeError(f"{failure} failed")

    with monkeypatch.context() as patch:
        patch.setattr(module._McpToolkit, "close", close)
        if failure == "state":
            patch.setattr(module, "EnvState", fail)
        elif failure == "start":
            patch.setattr(module._McpToolkit, "start", fail)
        elif failure == "solve":
            patch.setattr(
                module, "build_planner", lambda *a, **kw: SimpleNamespace(solve=fail)
            )
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            agent.run("task", system_prompt="rules")
    assert len(closed) == (0 if failure == "state" else 1)
    assert agent.run("task", system_prompt="rules").error is None


def test_running_episode_rejects_concurrent_use_and_allows_later_run(
    agent, monkeypatch
):
    entered, release = Event(), Event()

    def solve(**kwargs):
        entered.set()
        assert release.wait(5)
        return PlannerResult()

    monkeypatch.setattr(
        module, "build_planner", lambda *a, **kw: SimpleNamespace(solve=solve)
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(agent.run, "task", system_prompt="rules")
        try:
            assert entered.wait(5)
            with pytest.raises(RuntimeError, match="already has an active episode"):
                agent.run("task", system_prompt="rules")
        finally:
            release.set()
        assert future.result(timeout=5).error is None
    assert agent.run("task", system_prompt="rules").error is None
