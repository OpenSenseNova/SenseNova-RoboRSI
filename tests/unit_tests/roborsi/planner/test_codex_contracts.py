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

from __future__ import annotations

import json
import os
import queue
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic_ai.messages import BinaryContent

import roborsi.planner.codex as codex_module
from roborsi.dashboard.events import TranscriptEvent, UsageEvent
from roborsi.planner.codex import (
    PROVIDER_ENV_KEY,
    PROVIDER_ID,
    CodexPlanner,
    _codex_mcp_config_overrides,
    _interrupt,
)
from roborsi.planner.utils.http_mcp_server import (
    HttpMcpServer,
    _toolkit_to_mcp_content,
)
from roborsi.tools.toolkit import ToolResult


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    @property
    def enabled(self) -> bool:
        return True

    def emit(self, event: Any) -> None:
        self.events.append(event)


class FakeToolkit:
    def __init__(self) -> None:
        self.cancel_calls = 0

    def get_tools_spec(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "finish",
                "description": "Finish the task.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "summary": {"type": "string"},
                    },
                },
            }
        ]

    def execute_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        return ToolResult(name, {"name": name, "args": args})

    def cancel_active_and_wait(self) -> None:
        self.cancel_calls += 1


class FakeMcpServer:
    instances: list[FakeMcpServer] = []

    def __init__(self, toolkit: FakeToolkit) -> None:
        self.toolkit = toolkit
        self.started = False
        self.stopped = False
        self.instances.append(self)

    def start(self) -> str:
        self.started = True
        return "http://fake.invalid/mcp/"

    def stop(self) -> None:
        self.stopped = True


class FakeTurn:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.events = events
        self.interrupt_calls = 0
        self.steered: list[str] = []

    def stream(self):
        yield from self.events

    def interrupt(self) -> None:
        self.interrupt_calls += 1

    def steer(self, text: str) -> None:
        self.steered.append(text)


class FakeThread:
    def __init__(self, turn: FakeTurn) -> None:
        self.fake_turn = turn
        self.turn_prompts: list[tuple[str, dict[str, Any]]] = []

    def turn(self, prompt: str, **options: Any) -> FakeTurn:
        self.turn_prompts.append((prompt, options))
        return self.fake_turn


class FakeCodex:
    instances: list[FakeCodex] = []
    events: list[dict[str, Any]] = []

    def __init__(self, config: Any) -> None:
        self.config = config
        self.closed = False
        self.thread = FakeThread(FakeTurn(list(self.events)))
        self.instances.append(self)

    def __enter__(self) -> FakeCodex:
        return self

    def __exit__(self, *args: Any) -> None:
        self.closed = True

    def thread_start(self, **options: Any) -> FakeThread:
        self.thread_options = options
        return self.thread

    def close(self) -> None:
        self.closed = True


def make_planner(tmp_path: Path, sink: RecordingSink, *, timeout_s: float = 1):
    return CodexPlanner(
        output_dir=str(tmp_path),
        repo_root=tmp_path,
        timeout_s=timeout_s,
        extra_dirs=[str(tmp_path / "memory")],
        output_path=tmp_path / "codex.out",
        model="fake-codex",
        dashboard_events=sink,
    )


def install_fake_backend(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []

    def codex_config(**kwargs: Any) -> dict[str, Any]:
        configs.append(kwargs)
        return kwargs

    FakeMcpServer.instances.clear()
    FakeCodex.instances.clear()
    monkeypatch.setattr(FakeCodex, "events", [])
    monkeypatch.setattr(codex_module, "HttpMcpServer", FakeMcpServer)
    monkeypatch.setattr(codex_module.openai_codex, "CodexConfig", codex_config)
    monkeypatch.setattr(codex_module.openai_codex, "Codex", FakeCodex)
    return configs


@pytest.mark.parametrize("large_feedback", [False, True])
def test_episode_input_and_native_terminal_boundary(
    tmp_path, monkeypatch, large_feedback
):
    install_fake_backend(monkeypatch)
    toolkit = FakeToolkit()
    calls = []

    def execute(name, args):
        calls.append(name)
        payload = {"_finish": True, "status": "success", "summary": "native"}
        if large_feedback:
            payload["observation"] = "x" * 61000
        return ToolResult(name, payload)

    toolkit.execute_tool = execute

    def stream(turn):
        guarded = FakeMcpServer.instances[-1].toolkit
        terminal = guarded.execute_tool("finish", {})
        guarded.execute_tool("finish", {})
        yield {
            "method": "item/completed",
            "payload": {
                "item": {
                    "type": "mcpToolCall",
                    "tool": "finish",
                    "status": "completed",
                    "arguments": {},
                    "result": {"content": terminal.content_blocks},
                }
            },
        }
        yield {
            "method": "turn/completed",
            "payload": {"turn": {"status": "interrupted"}},
        }

    monkeypatch.setattr(FakeTurn, "stream", stream)
    result = make_planner(tmp_path, RecordingSink()).solve(
        system_prompt="system",
        user_message=[
            "observation",
            BinaryContent(data=b"pixels", media_type="image/png"),
        ],
        toolkit=toolkit,
        max_turns=3,
    )
    prompt = FakeCodex.instances[-1].thread.turn_prompts[0][0]
    assert prompt[0].text == "system"
    assert prompt[1].text == "observation"
    assert prompt[2].url == "data:image/png;base64,cGl4ZWxz"
    assert calls == ["finish"]
    assert result.error is None
    assert result.finish_result["summary"] == "native"
    assert "native" in json.dumps(result.messages)


def test_episode_cancel_wakes_pending_before_tool_cleanup(tmp_path, monkeypatch):
    install_fake_backend(monkeypatch)
    cancel = threading.Event()
    cancel.set()
    toolkit = FakeToolkit()
    events = []
    toolkit.cancel_pending = lambda: events.append("pending")
    toolkit.cancel_active_and_wait = lambda: events.append("active")
    result = make_planner(tmp_path, RecordingSink()).solve(
        system_prompt="",
        user_message="task",
        toolkit=toolkit,
        max_turns=3,
        cancel_event=cancel,
    )
    assert result.error_kind == "cancelled"
    assert result.stop_reason == "closed"
    assert events[:2] == ["pending", "active"]


@pytest.mark.parametrize(
    "item_type",
    [
        "mcpToolCall",
        "webSearch",
        "imageView",
        "sleep",
        "imageGeneration",
        "collabAgentToolCall",
    ],
)
def test_tool_only_stream_respects_sdk_step_budget(tmp_path, monkeypatch, item_type):
    install_fake_backend(monkeypatch)
    FakeCodex.events = [
        {
            "method": "item/completed",
            "payload": {
                "item": {
                    "type": item_type,
                    "tool": "inspect_scene",
                    "status": "completed",
                    "arguments": {},
                    "result": "visible",
                }
            },
        }
        for _ in range(5)
    ]
    result = make_planner(tmp_path, RecordingSink()).solve(
        system_prompt="",
        user_message="task",
        toolkit=FakeToolkit(),
        max_turns=2,
    )
    assert result.stop_reason == "max_turns"
    assert result.stats["sdk_steps"] == 2
    assert result.stats["tool_calls"] == 2
    assert FakeCodex.instances[-1].thread.fake_turn.interrupt_calls >= 1


def test_fatal_tool_failure_keeps_its_category(tmp_path, monkeypatch):
    from roborsi.planner.base import PlannerFailure

    install_fake_backend(monkeypatch)
    toolkit = FakeToolkit()

    def execute(name, args):
        raise PlannerFailure(
            "connection lost", kind="infrastructure", reason="bridge_lost"
        )

    toolkit.execute_tool = execute

    def stream(turn):
        FakeMcpServer.instances[-1].toolkit.execute_tool("finish", {})
        yield {"method": "turn/completed", "payload": {"turn": {"status": "completed"}}}

    monkeypatch.setattr(FakeTurn, "stream", stream)
    result = make_planner(tmp_path, RecordingSink()).solve(
        system_prompt="",
        user_message="task",
        toolkit=toolkit,
        max_turns=3,
    )
    assert result.error == "connection lost"
    assert result.error_kind == "infrastructure"
    assert result.stop_reason == "bridge_lost"


def test_actual_nonterminal_finish_result_owns_boundary(tmp_path, monkeypatch):
    install_fake_backend(monkeypatch)

    def stream(turn):
        FakeMcpServer.instances[-1].toolkit.execute_tool("finish", {})
        yield {
            "method": "item/completed",
            "payload": {
                "item": {
                    "type": "mcpToolCall",
                    "tool": "finish",
                    "status": "completed",
                    "arguments": {"status": "success", "summary": "claimed"},
                }
            },
        }

    monkeypatch.setattr(FakeTurn, "stream", stream)
    result = make_planner(tmp_path, RecordingSink()).solve(
        system_prompt="",
        user_message="task",
        toolkit=FakeToolkit(),
        max_turns=3,
    )
    assert result.finish_result is None


@pytest.mark.parametrize("acknowledge", [True, False])
def test_terminal_feedback_waits_for_sdk_acknowledgment(
    tmp_path, monkeypatch, acknowledge
):
    import time

    install_fake_backend(monkeypatch)
    toolkit = FakeToolkit()
    toolkit.execute_tool = lambda name, args: ToolResult(
        name, {"_finish": True, "status": "success"}
    )

    def stream(turn):
        terminal = FakeMcpServer.instances[-1].toolkit.execute_tool("finish", {})
        time.sleep(0.12)
        if acknowledge:
            assert turn.interrupt_calls == 0
            yield {
                "method": "item/completed",
                "payload": {
                    "item": {
                        "type": "mcpToolCall",
                        "tool": "finish",
                        "status": "completed",
                        "arguments": {},
                        "result": {"content": terminal.content_blocks},
                    }
                },
            }
        else:
            time.sleep(0.2)

    monkeypatch.setattr(FakeTurn, "stream", stream)
    result = make_planner(tmp_path, RecordingSink(), timeout_s=0.25).solve(
        system_prompt="",
        user_message="task",
        toolkit=toolkit,
        max_turns=3,
    )
    if acknowledge:
        assert result.error is None
    else:
        assert result.error_kind == "infrastructure"
        assert result.stop_reason == "timeout"


def test_sdk_close_failure_is_not_masked_by_terminal_success(tmp_path, monkeypatch):
    install_fake_backend(monkeypatch)
    toolkit = FakeToolkit()
    toolkit.execute_tool = lambda name, args: ToolResult(
        name, {"_finish": True, "status": "success"}
    )

    def stream(turn):
        terminal = FakeMcpServer.instances[-1].toolkit.execute_tool("finish", {})
        yield {
            "method": "item/completed",
            "payload": {
                "item": {
                    "type": "mcpToolCall",
                    "tool": "finish",
                    "status": "completed",
                    "arguments": {},
                    "result": {"content": terminal.content_blocks},
                }
            },
        }

    def close(client, *args):
        raise RuntimeError("sdk close failed")

    monkeypatch.setattr(FakeTurn, "stream", stream)
    monkeypatch.setattr(FakeCodex, "__exit__", close)
    result = make_planner(tmp_path, RecordingSink()).solve(
        system_prompt="",
        user_message="task",
        toolkit=toolkit,
        max_turns=3,
    )
    assert result.error_kind == "infrastructure"
    assert result.stop_reason == "cleanup_failed"
    assert result.finish_result["status"] == "success"


def test_installed_sdk_root_model_terminal_event_acknowledges(tmp_path, monkeypatch):
    from openai_codex.generated.v2_all import ItemCompletedNotification

    install_fake_backend(monkeypatch)
    toolkit = FakeToolkit()
    toolkit.execute_tool = lambda name, args: ToolResult(
        name, {"_finish": True, "status": "success"}
    )

    def stream(turn):
        terminal = FakeMcpServer.instances[-1].toolkit.execute_tool("finish", {})
        payload = ItemCompletedNotification.model_validate(
            {
                "completedAtMs": 1,
                "threadId": "thread",
                "turnId": "turn",
                "item": {
                    "type": "mcpToolCall",
                    "id": "native",
                    "server": "roborsi",
                    "tool": "finish",
                    "status": "completed",
                    "arguments": {},
                    "result": {"content": terminal.content_blocks},
                },
            }
        )
        yield SimpleNamespace(method="item/completed", payload=payload)

    monkeypatch.setattr(FakeTurn, "stream", stream)
    result = make_planner(tmp_path, RecordingSink()).solve(
        system_prompt="",
        user_message="task",
        toolkit=toolkit,
        max_turns=3,
    )
    assert result.error is None
    assert result.finish_result["status"] == "success"


def test_mcp_content_conversion_preserves_text_images_and_error_status() -> None:
    plain, plain_error = _toolkit_to_mcp_content("plain")
    assert plain_error is False
    assert plain[0].type == "text"
    assert plain[0].text == "plain"

    result = ToolResult(
        "finish",
        {"error": "finish refused", "_image_bytes": b"image bytes"},
    )
    content, is_error = _toolkit_to_mcp_content(result)

    assert [block.type for block in content] == ["text", "image"]
    assert json.loads(content[0].text) == {"error": "finish refused"}
    assert content[1].mimeType == "image/png"
    assert is_error is True


def test_http_mcp_readiness_ignores_environment_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    server = HttpMcpServer(FakeToolkit())

    try:
        url = server.start(ready_timeout_s=3.0)
    finally:
        server.stop()

    assert url.startswith("http://127.0.0.1:")


def test_config_overrides_normalize_provider_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CODEX_MODEL_CONTEXT_WINDOW", raising=False)
    monkeypatch.delenv("CODEX_AUTO_COMPACT_TOKEN_LIMIT", raising=False)
    assert _codex_mcp_config_overrides(
        mcp_url="http://fake.invalid/mcp/",
        base_url=None,
    ) == ['mcp_servers.roborsi.url="http://fake.invalid/mcp/"']

    overrides = _codex_mcp_config_overrides(
        mcp_url="http://fake.invalid/mcp/",
        base_url="https://provider.invalid/root/",
    )

    assert overrides == [
        'mcp_servers.roborsi.url="http://fake.invalid/mcp/"',
        f'model_provider="{PROVIDER_ID}"',
        f'model_providers.{PROVIDER_ID}.name="{PROVIDER_ID}"',
        f'model_providers.{PROVIDER_ID}.base_url="https://provider.invalid/root/v1"',
        f'model_providers.{PROVIDER_ID}.wire_api="responses"',
        f'model_providers.{PROVIDER_ID}.env_key="{PROVIDER_ENV_KEY}"',
    ]


def test_config_overrides_include_optional_context_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEX_MODEL_CONTEXT_WINDOW", "262144")
    monkeypatch.setenv("CODEX_AUTO_COMPACT_TOKEN_LIMIT", "230000")

    overrides = _codex_mcp_config_overrides(mcp_url=None, base_url=None)

    assert overrides == [
        "model_context_window=262144",
        "model_auto_compact_token_limit=230000",
    ]


def test_build_config_translates_environment_without_contacting_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEX_BASE_URL", "https://provider.invalid")
    monkeypatch.setenv("CODEX_API_KEY", "contract-key")
    monkeypatch.setenv("CODEX_BIN", "/fake/codex")
    captured: list[dict[str, Any]] = []

    def codex_config(**kwargs: Any) -> dict[str, Any]:
        captured.append(kwargs)
        return kwargs

    monkeypatch.setattr(codex_module.openai_codex, "CodexConfig", codex_config)
    planner = make_planner(tmp_path, RecordingSink())

    config = planner._build_config("http://fake.invalid/mcp/")

    assert config is captured[0]
    assert config["cwd"] == str(tmp_path)
    assert config["experimental_api"] is True
    assert config["codex_bin"] == "/fake/codex"
    assert config["env"][PROVIDER_ENV_KEY] == "contract-key"
    assert f'model_provider="{PROVIDER_ID}"' in config["config_overrides"]


def test_build_config_constructs_with_the_installed_codex_sdk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEX_BASE_URL", "https://provider.invalid")
    monkeypatch.setenv("CODEX_API_KEY", "contract-key")
    monkeypatch.delenv("CODEX_BIN", raising=False)
    planner = make_planner(tmp_path, RecordingSink())

    config = planner._build_config("http://fake.invalid/mcp/")

    assert isinstance(config, codex_module.openai_codex.CodexConfig)
    assert config.cwd == str(tmp_path)
    assert config.experimental_api is True
    assert config.env[PROVIDER_ENV_KEY] == "contract-key"
    assert f'model_provider="{PROVIDER_ID}"' in config.config_overrides
    assert "project_doc_max_bytes=0" in config.config_overrides
    assert not any(
        entry.startswith("skills.config=") for entry in config.config_overrides
    )


def test_build_config_disables_symlinked_project_skills(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CODEX_BIN", raising=False)
    skill = tmp_path / "shared" / "review-代码" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: review-pr\ndescription: Review a PR.\n---\n")
    linked_skill = tmp_path / ".agents" / "skills" / "review-pr"
    linked_skill.parent.mkdir(parents=True)
    linked_skill.symlink_to(skill.parent, target_is_directory=True)
    planner = make_planner(tmp_path, RecordingSink())

    config = planner._build_config("http://fake.invalid/mcp/")

    assert config.cwd == str(tmp_path)
    assert "project_doc_max_bytes=0" in config.config_overrides
    assert (
        "skills.config=["
        f"{{ path = {json.dumps(str(skill.resolve()), ensure_ascii=False)}, enabled = false }}]"
    ) in config.config_overrides


def test_build_config_scopes_loopback_no_proxy_to_codex_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setenv("NO_PROXY", "metadata.example.invalid")
    monkeypatch.setenv("no_proxy", ".corp.invalid")
    captured: list[dict[str, Any]] = []

    def codex_config(**kwargs: Any) -> dict[str, Any]:
        captured.append(kwargs)
        return kwargs

    monkeypatch.setattr(codex_module.openai_codex, "CodexConfig", codex_config)
    planner = make_planner(tmp_path, RecordingSink())

    config = planner._build_config("http://127.0.0.1:8765/mcp/")

    child_env = config["env"]
    assert child_env["HTTP_PROXY"] == "http://proxy.invalid:8080"
    assert child_env["HTTPS_PROXY"] == "http://proxy.invalid:8080"
    assert child_env["NO_PROXY"].split(",") == [
        "metadata.example.invalid",
        ".corp.invalid",
        "127.0.0.1",
        "localhost",
    ]
    assert child_env["no_proxy"] == child_env["NO_PROXY"]
    assert os.environ["NO_PROXY"] == "metadata.example.invalid"
    assert os.environ["no_proxy"] == ".corp.invalid"


def test_planner_forwards_configured_service_tier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_backend(monkeypatch)
    monkeypatch.setenv("CODEX_SERVICE_TIER", "fast")
    FakeCodex.events = [
        {
            "method": "turn/completed",
            "payload": {"turn": {"status": "completed", "duration_ms": 10}},
        },
    ]

    make_planner(tmp_path, RecordingSink()).solve(
        system_prompt="system rules",
        user_message="user task",
        toolkit=FakeToolkit(),
        max_turns=1,
    )

    fake_codex = FakeCodex.instances[0]
    assert fake_codex.thread_options["service_tier"] == "fast"
    assert fake_codex.thread.turn_prompts[0][1]["service_tier"] == "fast"


def test_successful_fake_codex_lifecycle_uses_fake_mcp_and_accounts_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configs = install_fake_backend(monkeypatch)
    embedded = "data:image/png;base64,embedded-image-secret"
    FakeCodex.events = [
        {
            "method": "item/completed",
            "payload": {"item": {"type": "userMessage", "content": embedded}},
        },
        {
            "method": "item/completed",
            "payload": {"item": {"type": "agentMessage", "text": "working"}},
        },
        {
            "method": "item/completed",
            "payload": {
                "item": {
                    "type": "mcpToolCall",
                    "tool": "mcp__roborsi__finish",
                    "status": "completed",
                    "arguments": {"status": "success", "summary": "done"},
                    "result": "accepted",
                }
            },
        },
        {
            "method": "thread/tokenUsage/updated",
            "payload": {
                "token_usage": {
                    "total": {
                        "input_tokens": 11,
                        "cached_input_tokens": 4,
                        "output_tokens": 6,
                        "reasoning_output_tokens": 2,
                    }
                }
            },
        },
        {
            "method": "turn/completed",
            "payload": {"turn": {"status": "completed", "duration_ms": 1250}},
        },
    ]
    sink = RecordingSink()
    result = make_planner(tmp_path, sink).solve(
        system_prompt="system rules",
        user_message="user task",
        toolkit=FakeToolkit(),
        max_turns=3,
    )

    assert configs
    assert FakeMcpServer.instances[0].started is True
    assert FakeMcpServer.instances[0].stopped is True
    fake_codex = FakeCodex.instances[0]
    assert fake_codex.closed is True
    assert fake_codex.thread.turn_prompts[0][0] == "system rules\n\nuser task"
    assert result.finish_result == {
        "_finish": True,
        "status": "success",
        "summary": "done",
    }
    assert result.error is None
    assert result.stats["backend"] == "codex_sdk"
    assert result.stats["turns_used"] == 1
    assert result.stats["tool_calls"] == 1
    assert result.stats["total_input_tokens"] == 11
    assert result.stats["total_output_tokens"] == 6
    transcript = json.dumps(result.messages)
    assert "<image omitted>" in transcript
    assert "embedded-image-secret" not in transcript
    assert (tmp_path / "codex.out.last").read_text() == "working"
    assert any(isinstance(event, TranscriptEvent) for event in sink.events)
    assert any(isinstance(event, UsageEvent) for event in sink.events)


def test_rejected_finish_item_is_not_promoted() -> None:
    from roborsi.planner.codex import _Recorder

    recorder = _Recorder(max_turns=2, dashboard_events=RecordingSink())

    rendered = recorder.observe(
        {
            "method": "item/completed",
            "payload": {
                "item": {
                    "type": "mcpToolCall",
                    "tool": "mcp__roborsi__finish",
                    "status": "failed",
                    "arguments": {"status": "success", "summary": "too early"},
                    "error": "finish refused",
                }
            },
        }
    )

    assert recorder.finish_result is None
    assert recorder.tool_calls == 1
    assert "finish" in rendered


def test_fake_codex_backend_failure_stops_mcp_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_backend(monkeypatch)

    class FailingCodex:
        def __init__(self, config: Any) -> None:
            del config

        def __enter__(self):
            raise RuntimeError("codex unavailable")

        def __exit__(self, *args: Any) -> None:
            return None

    monkeypatch.setattr(codex_module.openai_codex, "Codex", FailingCodex)
    result = make_planner(tmp_path, RecordingSink()).solve(
        system_prompt="",
        user_message="task",
        toolkit=FakeToolkit(),
        max_turns=1,
    )

    assert result.error == "RuntimeError: codex unavailable"
    assert FakeMcpServer.instances[0].stopped is True


def test_timeout_interrupts_without_starting_a_worker_or_socket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_backend(monkeypatch)

    class TimeoutThread:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.join_calls: list[float | None] = []

        def start(self) -> None:
            return None

        def join(self, timeout: float | None = None) -> None:
            self.join_calls.append(timeout)

        def is_alive(self) -> bool:
            return True

    monkeypatch.setattr(
        codex_module,
        "threading",
        SimpleNamespace(Thread=TimeoutThread),
    )
    result = make_planner(
        tmp_path,
        RecordingSink(),
        timeout_s=0.01,
    ).solve(
        system_prompt="",
        user_message="task",
        toolkit=FakeToolkit(),
        max_turns=1,
    )

    assert result.error == "Codex SDK timed out after 0.01s"
    assert FakeMcpServer.instances[0].stopped is True


def test_terminal_timeout_cancels_active_toolkit_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_backend(monkeypatch)

    class TimeoutThread:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def start(self) -> None:
            return None

        def join(self, timeout: float | None = None) -> None:
            del timeout

        def is_alive(self) -> bool:
            return True

    monkeypatch.setattr(
        codex_module,
        "threading",
        SimpleNamespace(Thread=TimeoutThread),
    )
    toolkit = FakeToolkit()
    result = make_planner(
        tmp_path,
        RecordingSink(),
        timeout_s=0.01,
    ).solve(
        system_prompt="",
        user_message="task",
        toolkit=toolkit,
        max_turns=1,
    )

    assert result.error == "Codex SDK timed out after 0.01s"
    assert FakeMcpServer.instances[0].stopped is True
    assert toolkit.cancel_calls == 1


def test_queue_and_dashboard_are_rejected_before_mcp_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnexpectedMcpServer:
        def __init__(self, toolkit: Any) -> None:
            raise AssertionError(f"MCP should not be constructed: {toolkit}")

    monkeypatch.setattr(codex_module, "HttpMcpServer", UnexpectedMcpServer)
    planner = make_planner(tmp_path, RecordingSink())

    with pytest.raises(ValueError, match="cannot be used together"):
        planner.solve(
            system_prompt="",
            user_message="task",
            toolkit=FakeToolkit(),
            max_turns=1,
            input_queue=queue.Queue(),
            dashboard_interaction=object(),
        )


def test_interrupt_attempts_both_turn_and_codex_cleanup() -> None:
    events: list[str] = []

    class Turn:
        def interrupt(self) -> None:
            events.append("turn-interrupt")
            raise RuntimeError("already complete")

    class Codex:
        def close(self) -> None:
            events.append("codex-close")

    _interrupt({"turn": Turn(), "codex": Codex()})

    assert events == ["turn-interrupt", "codex-close"]


# ---------------------------------------------------------------------------
# Connectivity probe: SDK turn-consumption contract
# ---------------------------------------------------------------------------


def test_turn_handle_is_not_iterable_and_exposes_run_and_stream() -> None:
    """Pin the SDK contract the probe got wrong.

    ``thread.turn(...)`` returns a ``TurnHandle``. It is NOT iterable: the only
    two ways to consume one are ``.stream()`` (an event iterator, used by the
    planner) and ``.run()`` (blocking, returns a ``TurnResult``, used by the
    probe). Iterating the handle itself raises
    ``TypeError: 'TurnHandle' object is not iterable``.
    """
    from openai_codex import TurnHandle

    assert callable(TurnHandle.run)
    assert callable(TurnHandle.stream)
    assert not hasattr(TurnHandle, "__iter__")
    assert not hasattr(TurnHandle, "__aiter__")


def test_turn_result_carries_the_fields_the_probe_reads() -> None:
    from openai_codex import TurnResult

    annotations = TurnResult.__annotations__
    for field in ("status", "error", "final_response"):
        assert field in annotations, f"TurnResult lost {field!r}"


class FakeProbeTurn:
    """A turn handle shaped like the real one: run/stream, never iterable."""

    def __init__(
        self,
        result: Any = None,
        *,
        blocks: threading.Event | None = None,
        raises: Exception | None = None,
    ) -> None:
        self.result = result
        self.blocks = blocks
        self.raises = raises
        self.interrupt_calls = 0

    def run(self) -> Any:
        if self.raises is not None:
            raise self.raises
        if self.blocks is not None:
            self.blocks.wait(timeout=10)
        return self.result

    def stream(self):
        raise AssertionError("the probe must use run(), not stream()")

    def interrupt(self) -> None:
        self.interrupt_calls += 1


class FakeProbeCodex:
    """Minimal Codex session double for the probe path."""

    instances: list[FakeProbeCodex] = []
    turn_factory: Any = None

    def __init__(self, config: Any) -> None:
        self.config = config
        self.closed = False
        self.thread_options: dict[str, Any] = {}
        self.turn_prompts: list[tuple[str, dict[str, Any]]] = []
        self.fake_turn = type(self).turn_factory()
        FakeProbeCodex.instances.append(self)

    def __enter__(self) -> FakeProbeCodex:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.closed = True

    def close(self) -> None:
        self.closed = True

    def thread_start(self, **options: Any) -> FakeProbeCodex:
        self.thread_options = options
        return self

    def turn(self, prompt: str, **options: Any) -> FakeProbeTurn:
        self.turn_prompts.append((prompt, options))
        return self.fake_turn


@pytest.fixture
def probe_codex(monkeypatch: pytest.MonkeyPatch):
    """Install FakeProbeCodex and hand back its instance list."""
    FakeProbeCodex.instances = []

    def _install(turn_factory: Any) -> list[FakeProbeCodex]:
        FakeProbeCodex.turn_factory = turn_factory
        monkeypatch.setattr(codex_module.openai_codex, "Codex", FakeProbeCodex)
        return FakeProbeCodex.instances

    return _install


def test_probe_consumes_the_turn_with_run_and_returns_final_response(
    probe_codex: Any,
) -> None:
    result = SimpleNamespace(status="completed", error=None, final_response="ok")
    instances = probe_codex(lambda: FakeProbeTurn(result))

    reply = codex_module.run_probe_turn(
        object(), prompt="ping", model="gpt-5.5", timeout_s=30
    )

    assert reply == "ok"
    codex = instances[0]
    assert codex.turn_prompts[0][0] == "ping"
    assert codex.thread_options["model"] == "gpt-5.5"
    assert codex.closed is True


def test_probe_omits_model_when_none_so_the_sdk_default_applies(
    probe_codex: Any,
) -> None:
    result = SimpleNamespace(status="completed", error=None, final_response="ok")
    instances = probe_codex(lambda: FakeProbeTurn(result))

    codex_module.run_probe_turn(object(), prompt="ping", model=None, timeout_s=30)

    assert "model" not in instances[0].thread_options


def test_probe_runs_read_only_and_denies_approvals(probe_codex: Any) -> None:
    result = SimpleNamespace(status="completed", error=None, final_response="ok")
    instances = probe_codex(lambda: FakeProbeTurn(result))

    codex_module.run_probe_turn(object(), prompt="ping", model=None, timeout_s=30)

    options = instances[0].thread_options
    assert options["approval_mode"] is codex_module.openai_codex.ApprovalMode.deny_all
    assert options["sandbox"] is codex_module.openai_codex.Sandbox.read_only


def test_probe_forwards_configured_service_tier(
    probe_codex: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEX_SERVICE_TIER", "fast")
    result = SimpleNamespace(status="completed", error=None, final_response="ok")
    instances = probe_codex(lambda: FakeProbeTurn(result))

    codex_module.run_probe_turn(object(), prompt="ping", model=None, timeout_s=30)

    assert instances[0].thread_options["service_tier"] == "fast"
    assert instances[0].turn_prompts[0][1]["service_tier"] == "fast"


def test_probe_config_attaches_no_mcp_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CODEX_BASE_URL", raising=False)
    monkeypatch.delenv("CODEX_API_KEY", raising=False)
    monkeypatch.delenv("CODEX_MODEL_CONTEXT_WINDOW", raising=False)
    monkeypatch.delenv("CODEX_AUTO_COMPACT_TOKEN_LIMIT", raising=False)

    overrides = _codex_mcp_config_overrides(mcp_url=None, base_url=None)

    assert overrides == []
    assert not any("mcp_servers" in entry for entry in overrides)


def test_probe_config_keeps_the_provider_override_without_mcp() -> None:
    overrides = _codex_mcp_config_overrides(mcp_url=None, base_url="https://gw.example")

    assert not any("mcp_servers" in entry for entry in overrides)
    assert any("model_providers" in entry for entry in overrides)


def test_probe_timeout_interrupts_the_turn_and_closes_the_session(
    probe_codex: Any,
) -> None:
    release = threading.Event()
    turns: list[FakeProbeTurn] = []

    def factory() -> FakeProbeTurn:
        turn = FakeProbeTurn(None, blocks=release)
        turns.append(turn)
        return turn

    instances = probe_codex(factory)
    try:
        with pytest.raises(TimeoutError) as excinfo:
            codex_module.run_probe_turn(
                object(), prompt="ping", model=None, timeout_s=1
            )
        assert "1s" in str(excinfo.value)
        assert turns[0].interrupt_calls == 1
        assert instances[0].closed is True
    finally:
        release.set()


def test_probe_raises_when_the_turn_status_is_failed(probe_codex: Any) -> None:
    result = SimpleNamespace(
        status="failed",
        error=SimpleNamespace(message="model provider rejected the request"),
        final_response=None,
    )
    probe_codex(lambda: FakeProbeTurn(result))

    with pytest.raises(RuntimeError, match="model provider rejected"):
        codex_module.run_probe_turn(object(), prompt="ping", model=None, timeout_s=30)


def test_probe_propagates_sdk_exceptions_for_classification(probe_codex: Any) -> None:
    probe_codex(lambda: FakeProbeTurn(raises=RuntimeError("401 Unauthorized")))

    with pytest.raises(RuntimeError, match="401 Unauthorized"):
        codex_module.run_probe_turn(object(), prompt="ping", model=None, timeout_s=30)


def test_probe_returns_empty_string_when_the_model_said_nothing(
    probe_codex: Any,
) -> None:
    result = SimpleNamespace(status="completed", error=None, final_response=None)
    probe_codex(lambda: FakeProbeTurn(result))

    reply = codex_module.run_probe_turn(
        object(), prompt="ping", model=None, timeout_s=30
    )

    assert reply == ""
