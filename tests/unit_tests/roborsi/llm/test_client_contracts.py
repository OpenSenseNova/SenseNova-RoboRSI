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

"""Offline contracts for provider selection and usage accounting."""

from __future__ import annotations

import asyncio
import builtins
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import BinaryContent
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.messages import (
    CachePoint,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage, RunUsage

from roborsi.dashboard.events import NullDashboardEventSink
from roborsi.llm import LLMClient, LLMConfig, LLMUsage, RetryPolicy
from roborsi.llm.client import (
    _mark_selected_function_outputs,
    build_model_settings,
)
from roborsi.llm.retry import RetryLoggingModel
from roborsi.planner.api_loop import _build_stats, _prune_history_images
from roborsi.planner.base import build_planner


@pytest.mark.parametrize(
    ("config", "model_type"),
    [
        (LLMConfig("openai", "gpt-4o", api_key="test"), "RoboRSIResponsesModel"),
        (
            LLMConfig("openai", "gpt-4o", api_key="test", openai_format="chat"),
            "OpenAIChatModel",
        ),
        (LLMConfig("anthropic", "claude-test", api_key="test"), "AnthropicModel"),
    ],
)
def test_provider_configuration_builds_selected_model(
    config: LLMConfig, model_type: str
) -> None:
    model = config.build_model()
    assert type(model).__name__ == model_type
    assert model.model_name == config.model


def test_model_factory_constructs_a_fresh_model_and_can_omit_token_limit():
    created = []

    def factory():
        model = FunctionModel(lambda *_: ModelResponse(parts=[TextPart("ok")]))
        created.append(model)
        return model

    config = LLMConfig("openai", "custom", model_factory=factory)
    first, second = config.build_model(), config.build_model()
    assert first is not second
    assert created == [first, second]
    assert "max_tokens" not in build_model_settings(first, None)
    assert build_model_settings(first, 8192)["max_tokens"] == 8192


@pytest.mark.parametrize("extra", [None, {"reasoning_effort": "medium"}])
def test_llm_config_preserves_legacy_positional_retry_and_factory(extra):
    retry = RetryPolicy(max_retries=1)
    model = FunctionModel(lambda *_: ModelResponse(parts=[TextPart("ok")]))

    def factory():
        return model

    # Preserve all positional fields supported before extra_body was added.
    config = LLMConfig(
        "openai",
        "custom",
        "test",
        "http://test.invalid/v1",
        "chat",
        None,
        None,
        False,
        None,
        0,
        None,
        retry,
        factory,
        extra_body=extra,
    )
    assert config.retry is retry
    assert config.model_factory is factory
    assert config.extra_body == extra
    assert config.build_model() is model


def test_model_factory_rejects_invalid_constructor_and_result():
    with pytest.raises(ValueError, match="model_factory must be callable"):
        LLMConfig("openai", "custom", model_factory="not callable")
    with pytest.raises(TypeError, match="must return a Pydantic AI Model"):
        LLMConfig("openai", "custom", model_factory=lambda: object()).build_model()


@pytest.mark.parametrize("timeout, expected", [(None, 17), (0, None), (9, 9)])
def test_api_planner_timeout_can_be_explicitly_disabled(
    tmp_path, monkeypatch, timeout, expected
):
    monkeypatch.setenv("CELL_TIMEOUT_S", "17")
    config = LLMConfig(
        "openai",
        "custom",
        model_factory=lambda: FunctionModel(
            lambda *_: ModelResponse(parts=[TextPart("ok")])
        ),
    )
    planner = build_planner(
        "api",
        output_dir=tmp_path,
        recipe_tag="test",
        robot_name="test",
        llm_config=config,
        planner_timeout_s=timeout,
        dashboard_events=NullDashboardEventSink(),
    )
    assert planner._timeout_s == expected


def test_openai_settings_do_not_import_anthropic_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = LLMConfig("openai", "gpt-test", api_key="test").build_model()
    real_import = builtins.__import__

    def import_without_anthropic(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pydantic_ai.models.anthropic":
            raise ImportError("anthropic is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_anthropic)
    assert build_model_settings(model, 100)["max_tokens"] == 100


def test_responses_explicit_cache_settings_and_breakpoint_wire_format() -> None:
    config = LLMConfig(
        "openai",
        "gpt-6-astra/example/deployment",
        api_key="test",
        prompt_cache_key="robodojo-stable-v1",
        prompt_cache_mode="explicit",
        image_history_groups=2,
        parallel_tool_calls=False,
    )
    model = config.build_model()
    assert type(model).__name__ == "RoboRSIResponsesModel"
    settings = build_model_settings(model, 128)
    assert settings["openai_prompt_cache_key"] == "robodojo-stable-v1"
    assert settings["openai_prompt_cache_options"] == {"mode": "explicit"}
    assert settings["parallel_tool_calls"] is False
    mapped = asyncio.run(
        model._map_user_prompt(UserPromptPart(content=["stable goal", CachePoint()]))
    )
    assert mapped["content"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    selected_settings = {**settings, "roborsi_cache_tool_call_ids": ("call-1",)}
    _, wire = asyncio.run(
        model._map_messages(
            [
                ModelRequest(
                    parts=[UserPromptPart(content=["stable goal", CachePoint()])]
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name="move_eef",
                            content="new robot state",
                            tool_call_id="call-1",
                        )
                    ]
                ),
            ],
            selected_settings,
            ModelRequestParameters(),
        )
    )
    assert wire[1]["output"] == [
        {
            "type": "input_text",
            "text": "new robot state",
            "prompt_cache_breakpoint": {"mode": "explicit"},
        }
    ]
    outputs = [
        {"type": "function_call_output", "call_id": str(index), "output": str(index)}
        for index in range(42)
    ]
    _mark_selected_function_outputs(outputs, ("2",))
    assert outputs[0]["output"] == "0"
    assert outputs[1]["output"] == "1"
    assert outputs[2]["output"] == [
        {
            "type": "input_text",
            "text": "2",
            "prompt_cache_breakpoint": {"mode": "explicit"},
        }
    ]


def test_explicit_cache_keeps_user_selected_breakpoint_when_old_image_ages_out() -> (
    None
):
    model = LLMConfig(
        "openai",
        "gpt-6-astra/example/deployment",
        api_key="test",
        prompt_cache_mode="explicit",
        image_history_groups=2,
    ).build_model()
    settings = build_model_settings(model, 128)
    messages = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        "stable goal",
                        CachePoint(),
                        "demonstration camera:",
                        BinaryContent(data=b"demonstration", media_type="image/jpeg"),
                        "camera 0:",
                        BinaryContent(data=b"frame-0", media_type="image/jpeg"),
                    ]
                )
            ]
        ),
        *(
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            f"camera {index}:",
                            BinaryContent(
                                data=f"frame-{index}".encode(),
                                media_type="image/jpeg",
                            ),
                        ]
                    )
                ]
            )
            for index in range(1, 4)
        ),
    ]
    _, before = asyncio.run(
        model._map_messages(messages, settings, ModelRequestParameters())
    )
    pruned = _prune_history_images(
        messages, max_groups=2, preserve_initial_image_count=1
    )
    _, after = asyncio.run(
        model._map_messages(pruned, settings, ModelRequestParameters())
    )

    # The selected marker remains, while camera labels are not auto-marked.
    for index in range(4):
        label_before = before[index]["content"][-2]
        label_after = after[index]["content"][-2]
        assert label_before == label_after
        assert "prompt_cache_breakpoint" not in label_after
    assert after[0]["content"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert after[0]["content"][2]["type"] == "input_image"
    assert after[0]["content"][-1]["type"] == "input_text"
    assert after[1]["content"][-1]["type"] == "input_text"
    assert after[2]["content"][-1]["type"] == "input_image"
    assert after[3]["content"][-1]["type"] == "input_image"


def test_explicit_cache_marks_only_selected_tool_feedback() -> None:
    wire = [
        {"type": "function_call_output", "call_id": "move", "output": "robot state"},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "latest camera:"},
                {"type": "input_image", "image_url": "data:image/jpeg;base64,YQ=="},
            ],
        },
    ]
    _mark_selected_function_outputs(wire, ("move",))
    assert wire[0]["output"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert "prompt_cache_breakpoint" not in wire[1]["content"][0]


def test_selected_multimodal_tool_feedback_marks_final_content_block() -> None:
    model = LLMConfig(
        "openai",
        "gpt-6-astra/example/deployment",
        api_key="test",
        prompt_cache_mode="explicit",
    ).build_model()
    settings = {
        **build_model_settings(model, 128),
        "roborsi_cache_tool_call_ids": ("snapshot",),
    }
    _, wire = asyncio.run(
        model._map_messages(
            [
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            "snapshot",
                            [
                                "state",
                                BinaryContent(data=b"frame", media_type="image/png"),
                            ],
                            "snapshot",
                        )
                    ]
                )
            ],
            settings,
            ModelRequestParameters(),
        )
    )
    assert wire[0]["output"][-1]["type"] == "input_image"
    assert wire[0]["output"][-1]["prompt_cache_breakpoint"] == {"mode": "explicit"}


def test_explicit_mode_does_not_add_unselected_breakpoints() -> None:
    model = LLMConfig(
        "openai",
        "gpt-6-astra/example/deployment",
        api_key="test",
        prompt_cache_mode="explicit",
    ).build_model()
    _, wire = asyncio.run(
        model._map_messages(
            [
                ModelRequest(parts=[UserPromptPart("task")]),
                ModelRequest(parts=[ToolReturnPart("move", "state", "call-1")]),
            ],
            build_model_settings(model, 128),
            ModelRequestParameters(),
        )
    )
    assert wire[1]["output"] == "state"
    assert not any("prompt_cache_breakpoint" in str(item) for item in wire)


def test_static_prefix_breakpoint_is_configured_separately() -> None:
    model = LLMConfig(
        "openai",
        "gpt-6-astra/example/deployment",
        api_key="test",
        prompt_cache_mode="explicit",
        prompt_cache_static_prefix=True,
    ).build_model()
    from pydantic_ai.messages import InstructionPart

    assert build_model_settings(model, 128)["roborsi_cache_static_prefix"] is True

    instructions, wire = asyncio.run(
        model._map_messages(
            [ModelRequest(parts=[UserPromptPart("task")])],
            build_model_settings(model, 128),
            ModelRequestParameters(
                instruction_parts=[
                    InstructionPart("stable instructions"),
                    InstructionPart("changing state", dynamic=True),
                ]
            ),
        )
    )
    assert wire[0]["role"] == "developer"
    assert wire[0]["content"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert wire[0]["content"][0]["text"] == "stable instructions"
    assert wire[1]["content"][0]["text"] == "changing state"
    assert "prompt_cache_breakpoint" not in wire[1]["content"][0]
    assert "stable instructions" not in str(instructions)


def test_prompt_cache_configuration_rejects_incompatible_endpoints() -> None:
    with pytest.raises(ValueError, match="OpenAI Responses"):
        LLMConfig("anthropic", "claude-test", prompt_cache_mode="explicit")
    with pytest.raises(ValueError, match="OpenAI Responses"):
        LLMConfig("openai", "gpt-test", openai_format="chat", prompt_cache_key="stable")


def test_direct_calls_report_per_call_and_cumulative_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def model(messages: list[Any], info: Any) -> ModelResponse:
        assert info.instructions == "Be concise."
        assert info.model_settings["max_tokens"] == 128
        assert messages
        return ModelResponse(
            parts=[TextPart("done")],
            usage=RequestUsage(
                input_tokens=10,
                output_tokens=5,
                cache_read_tokens=3,
                cache_write_tokens=2,
                details={"reasoning_tokens": 1},
            ),
        )

    monkeypatch.setattr(LLMConfig, "build_model", lambda self: FunctionModel(model))
    client = LLMClient(LLMConfig("openai", "offline", api_key="test"))
    sync = client.generate_sync("first", system_prompt="Be concise.", max_tokens=128)
    async_result = asyncio.run(
        client.generate("second", system_prompt="Be concise.", max_tokens=128)
    )

    assert sync.text == async_result.text == "done"
    assert sync.usage.as_dict() == {
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_read_tokens": 3,
        "cache_write_tokens": 2,
        "reasoning_output_tokens": 1,
        "requests": 1,
        "cost_usd": None,
        "total_tokens": 15,
    }
    assert client.total_usage.input_tokens == 20
    assert client.total_usage.output_tokens == 10
    assert client.total_usage.cache_read_tokens == 6
    assert client.total_usage.requests == 2


def test_api_planner_uses_explicit_llm_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart("ok")]))
    monkeypatch.setattr(LLMConfig, "build_model", lambda self: model)
    config = LLMConfig("anthropic", "offline", api_key="test")
    planner = build_planner(
        "api",
        output_dir=tmp_path,
        recipe_tag="test",
        robot_name="test",
        llm_config=config,
        dashboard_events=NullDashboardEventSink(),
    )
    assert isinstance(planner._model, RetryLoggingModel)
    assert planner._model.wrapped is model
    assert planner._model.log_path == tmp_path / "llm_errors.jsonl"


@pytest.mark.parametrize(
    ("status", "expected_attempts"),
    [(400, 1), (401, 1), (429, 3), (503, 3)],
)
def test_http_failures_have_bounded_retries_and_sanitized_logs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    status: int,
    expected_attempts: int,
) -> None:
    attempts = 0

    def model(messages: list[Any], info: Any) -> ModelResponse:
        nonlocal attempts
        attempts += 1
        raise ModelHTTPError(status, "offline", body={"prompt": "private text"})

    monkeypatch.setattr(LLMConfig, "build_model", lambda self: FunctionModel(model))
    log_path = tmp_path / "llm_errors.jsonl"
    client = LLMClient(
        LLMConfig(
            "openai",
            "offline",
            api_key="secret-key",
            retry=RetryPolicy(max_retries=2, initial_delay_s=0, max_delay_s=0),
        ),
        log_path=log_path,
    )
    with pytest.raises(ModelHTTPError):
        client.generate_sync("private text")

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert attempts == expected_attempts
    assert len(records) == expected_attempts
    assert [record["will_retry"] for record in records] == [
        *([True] * (expected_attempts - 1)),
        False,
    ]
    assert all(record["status_code"] == status for record in records)
    assert all(
        record["response_body"]["body"] == {"prompt": "[redacted]"}
        for record in records
    )
    assert (log_path.stat().st_mode & 0o777) == 0o600
    assert "private text" not in log_path.read_text()
    assert "secret-key" not in log_path.read_text()
    request_records = [
        json.loads(line)
        for line in (tmp_path / "llm_requests.jsonl").read_text().splitlines()
    ]
    assert len(request_records) == expected_attempts
    assert {record["request_id"] for record in request_records} == {
        request_records[0]["request_id"]
    }
    assert [record["attempt"] for record in request_records] == list(
        range(1, expected_attempts + 1)
    )
    assert all(record["status_code"] == status for record in request_records)
    assert all(record["request_shape"]["text_chars"] > 0 for record in request_records)
    request_log = (tmp_path / "llm_requests.jsonl").read_text()
    assert "private text" not in request_log
    assert "secret-key" not in request_log


def test_transient_failure_recovers_without_replaying_completed_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    attempts = 0

    def model(messages: list[Any], info: Any) -> ModelResponse:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ModelHTTPError(429, "offline")
        return ModelResponse(
            parts=[TextPart("recovered")],
            usage=RequestUsage(input_tokens=4, output_tokens=2),
        )

    monkeypatch.setattr(LLMConfig, "build_model", lambda self: FunctionModel(model))
    log_path = tmp_path / "llm_errors.jsonl"
    client = LLMClient(
        LLMConfig(
            "anthropic",
            "offline",
            api_key="test",
            retry=RetryPolicy(max_retries=2, initial_delay_s=0, max_delay_s=0),
        ),
        log_path=log_path,
    )
    result = asyncio.run(client.generate("task"))
    assert attempts == 2
    assert result.text == "recovered"
    assert result.usage.requests == 1
    assert len(log_path.read_text().splitlines()) == 1
    request_records = [
        json.loads(line)
        for line in (tmp_path / "llm_requests.jsonl").read_text().splitlines()
    ]
    assert [record["outcome"] for record in request_records] == ["error", "success"]
    assert request_records[1]["usage"]["input_tokens"] == 4
    assert request_records[1]["usage"]["output_tokens"] == 2


def test_failed_multimodal_request_logs_only_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def model(messages: list[Any], info: Any) -> ModelResponse:
        raise ModelHTTPError(500, "offline", body={"prompt": "private text"})

    monkeypatch.setattr(LLMConfig, "build_model", lambda self: FunctionModel(model))
    client = LLMClient(
        LLMConfig(
            "openai",
            "offline",
            api_key="secret-key",
            retry=RetryPolicy(max_retries=0),
        ),
        log_path=tmp_path / "llm_errors.jsonl",
    )
    with pytest.raises(ModelHTTPError):
        client._agent("private instruction", 128).run_sync(
            [
                "private text",
                BinaryContent(data=b"jpeg-secret", media_type="image/jpeg"),
            ]
        )
    record = json.loads((tmp_path / "llm_requests.jsonl").read_text().splitlines()[0])
    assert record["status_code"] == 500
    assert record["request_shape"]["image_count"] == 1
    assert record["request_shape"]["image_bytes"] == len(b"jpeg-secret")
    assert record["request_shape"]["text_chars"] >= len("private text") + len(
        "private instruction"
    )
    assert record["request_shape"]["max_output_tokens"] == 128
    log = (tmp_path / "llm_requests.jsonl").read_text()
    assert "private text" not in log
    assert "private instruction" not in log
    assert "jpeg-secret" not in log
    assert "secret-key" not in log


def test_http_500_logs_server_body_with_redaction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def model(messages: list[Any], info: Any) -> ModelResponse:
        raise ModelHTTPError(
            500,
            "offline",
            body={
                "error": {
                    "code": "server_error",
                    "type": "upstream_timeout",
                    "message": "Internal Server Error: upstream timed out",
                    "prompt": "private task text",
                }
            },
            headers={"x-request-id": "request-123", "authorization": "secret-key"},
        )

    monkeypatch.setattr(LLMConfig, "build_model", lambda self: FunctionModel(model))
    client = LLMClient(
        LLMConfig(
            "openai", "offline", api_key="secret-key", retry=RetryPolicy(max_retries=0)
        ),
        log_path=tmp_path / "llm_errors.jsonl",
    )
    with pytest.raises(ModelHTTPError):
        client.generate_sync("private task text")
    record = json.loads((tmp_path / "llm_errors.jsonl").read_text().splitlines()[0])
    assert record["server_error"]["code"] == "server_error"
    assert record["server_error"]["type"] == "upstream_timeout"
    assert record["server_error"]["x_request_id"] == "request-123"
    assert (
        record["server_error"]["message"] == "Internal Server Error: upstream timed out"
    )
    assert len(record["server_error"]["body_sha256"]) == 64
    assert record["response_body"]["available"] is True
    assert record["response_body"]["body"]["error"]["code"] == "server_error"
    assert record["response_body"]["body"]["error"]["prompt"] == "[redacted]"
    assert "private task text" not in (tmp_path / "llm_errors.jsonl").read_text()
    assert "secret-key" not in (tmp_path / "llm_requests.jsonl").read_text()


@pytest.mark.parametrize(
    ("error", "expected_attempts"),
    [(ModelAPIError("offline", "connection failed"), 3), (ValueError("bad data"), 1)],
)
def test_non_http_failures_are_logged_and_only_connection_errors_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error: Exception,
    expected_attempts: int,
) -> None:
    attempts = 0

    def model(messages: list[Any], info: Any) -> ModelResponse:
        nonlocal attempts
        attempts += 1
        raise error

    monkeypatch.setattr(LLMConfig, "build_model", lambda self: FunctionModel(model))
    log_path = tmp_path / "llm_errors.jsonl"
    client = LLMClient(
        LLMConfig(
            "openai",
            "offline",
            api_key="test",
            retry=RetryPolicy(max_retries=2, initial_delay_s=0, max_delay_s=0),
        ),
        log_path=log_path,
    )
    with pytest.raises(type(error)):
        client.generate_sync("task")
    assert attempts == expected_attempts
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert all("response_body" in record for record in records)
    if isinstance(error, ModelAPIError):
        assert all(record["response_body"]["available"] is False for record in records)
    assert len(log_path.read_text().splitlines()) == expected_attempts


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_provider_sdk_retries_are_disabled_for_explicit_policy(provider: str) -> None:
    client = LLMClient(LLMConfig(provider, "offline", api_key="test"))
    assert client._model.wrapped.provider.client.max_retries == 0


def test_planner_usage_normalizes_cache_without_double_counting() -> None:
    api = LLMUsage.from_planner_stats(
        {
            "total_input_tokens": 10,
            "total_output_tokens": 5,
            "cache_read_tokens": 3,
            "cache_write_tokens": 2,
            "requests": 1,
        }
    )
    claude_sdk = LLMUsage.from_planner_stats(
        {
            "total_input_tokens": 5,
            "total_output_tokens": 5,
            "total_cache_read_input_tokens": 3,
            "total_cache_creation_input_tokens": 2,
        }
    )
    codex_sdk = LLMUsage.from_planner_stats(
        {
            "total_input_tokens": 10,
            "total_output_tokens": 5,
            "total_cached_input_tokens": 3,
            "total_reasoning_output_tokens": 1,
        }
    )
    assert api.input_tokens == claude_sdk.input_tokens == codex_sdk.input_tokens == 10
    assert api.total_tokens == claude_sdk.total_tokens == codex_sdk.total_tokens == 15
    assert api.cache_read_tokens == claude_sdk.cache_read_tokens == 3
    assert codex_sdk.reasoning_output_tokens == 1


@pytest.mark.parametrize("cache_write", [17, None])
def test_codex_cli_cache_write_counts_are_already_included_in_input(cache_write):
    usage = LLMUsage.from_planner_stats(
        {
            "usage_source": "codex_cli.turn.completed",
            "total_input_tokens": 100,
            "total_output_tokens": 20,
            "total_cached_input_tokens": 40,
            "total_cache_creation_input_tokens": cache_write,
        }
    )
    assert usage.input_tokens == 100
    assert usage.output_tokens == 20
    assert usage.total_tokens == 120
    assert usage.cache_read_tokens == 40
    if cache_write is not None:
        assert usage.cache_write_tokens == 17


def test_api_planner_preserves_reported_reasoning_and_cost() -> None:
    stats = _build_stats(
        RunUsage(
            input_tokens=10,
            output_tokens=5,
            details={"reasoning_tokens": 2},
            cost=0.01,
        ),
        turns=1,
        n_tool_calls=0,
    )
    usage = LLMUsage.from_planner_stats(stats)
    assert usage.reasoning_output_tokens == 2
    assert usage.cost_usd == 0.01


@pytest.mark.parametrize(
    "extra",
    [
        None,
        {
            "reasoning_effort": "medium",
            "chat_template_kwargs": {
                "enable_thinking": True,
                "reasoning_effort": "medium",
            },
        },
    ],
)
def test_chat_extra_body_reaches_wire_without_changing_default(monkeypatch, extra):
    import importlib

    from pydantic_ai import Agent
    from pydantic_ai.capabilities import Thinking

    captured = []
    for module_name in ("httpx", "httpx2"):
        try:
            transport = importlib.import_module(module_name)
        except ImportError:
            continue

        def make_capture(http):
            async def capture(self, request, *args, **kwargs):
                payload = json.loads(request.content)
                captured.append(payload)
                return http.Response(
                    200,
                    request=request,
                    json={
                        "id": "test",
                        "object": "chat.completion",
                        "created": 1,
                        "model": payload["model"],
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "ok"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "total_tokens": 2,
                        },
                    },
                )

            return capture

        monkeypatch.setattr(transport.AsyncClient, "send", make_capture(transport))
    model = LLMConfig(
        "openai",
        "custom-27b",
        api_key="test",
        base_url="http://test.invalid/v1",
        openai_format="chat",
        extra_body=extra,
    ).build_model()
    agent = Agent(model, capabilities=[Thinking(effort="medium")])
    asyncio.run(agent.run("test"))
    assert len(captured) == 1
    if extra is None:
        assert "chat_template_kwargs" not in captured[0]
    else:
        assert captured[0]["chat_template_kwargs"] == extra["chat_template_kwargs"]
        assert captured[0]["reasoning_effort"] == "medium"


@pytest.mark.parametrize("extra", [[], {"value": float("nan")}, {"value": object()}])
def test_chat_extra_body_rejects_non_json(extra):
    with pytest.raises(ValueError, match="extra_body"):
        LLMConfig("openai", "custom", openai_format="chat", extra_body=extra)
