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

"""External HTTP Agents use the same case runner without owning their server."""

from __future__ import annotations

from pathlib import Path

import pytest

from roborsi.evaluation.http_agent import HttpAgentLauncher
from roborsi.utils.daemon import pick_free_port
from tests.unit_tests.roborsi.evaluation.test_orchestration import (
    _AdapterExecutor,
    _orchestrator,
    _run,
)
from tests.unit_tests.roborsi.evaluation.test_services import _service


@pytest.mark.parametrize(
    "endpoint",
    [
        "ftp://example.org",
        "http://user:password@example.org",
        "http://example.org/call",
        "http://example.org:bad",
        "http://example.org:0",
    ],
)
def test_external_agent_requires_an_http_origin(endpoint: str) -> None:
    with pytest.raises(ValueError, match="origin URL"):
        HttpAgentLauncher(endpoint)


def test_external_agent_runs_case_and_leaves_borrowed_server_alive(
    tmp_path: Path,
) -> None:
    with _service("agent") as rpc:
        launcher = HttpAgentLauncher(rpc._base_url, startup_timeout_s=3.0)
        executor = _AdapterExecutor()
        orchestrator, store = _orchestrator(tmp_path, executor, launcher)

        result = _run(orchestrator)

        assert result.status == "success" and result.score == 0.75
        assert executor.closed and store.load("run1", "case1") == result
        assert rpc.call("healthz", timeout_s=1.0) == {
            "status": "ok",
            "roborsi_protocol_versions": [1],
        }


def test_unreachable_external_agent_is_recorded_as_agent_startup_error(
    tmp_path: Path,
) -> None:
    executor = _AdapterExecutor()
    launcher = HttpAgentLauncher(
        f"http://127.0.0.1:{pick_free_port()}", startup_timeout_s=0.2
    )
    orchestrator, store = _orchestrator(tmp_path, executor, launcher)

    result = _run(orchestrator)

    assert result.status == "agent_error"
    assert result.error is not None and result.error.details["stage"] == "agent.startup"
    assert executor.closed and store.load("run1", "case1") == result
