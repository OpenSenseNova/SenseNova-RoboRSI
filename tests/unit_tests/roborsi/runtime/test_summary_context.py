# Copyright 2026 The RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Tests for the built-in token-budgeted summary context policy."""

import asyncio
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import (
    BinaryContent,
    CachePoint,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from roborsi.runtime.context_engine import (
    ContextEngineCapability,
    ContextPolicy,
    ContextResult,
    _apply_cache_breakpoints,
    validate_context,
)
from roborsi.runtime.summary_context import _PREFIX, make_summary_engine


def _policy(**values):
    defaults = {
        "strategy": "summary",
        "context_window": 16000,
        "output_reserve": 1024,
        "overhead_reserve": 1024,
        "summary_max_tokens": 1024,
        "summary_trigger_ratio": 0.5,
        "keep_turns": 2,
    }
    return ContextPolicy(**{**defaults, **values})


def _history():
    messages = [ModelRequest(parts=[UserPromptPart("complete task A")])]
    for index in range(7):
        messages.extend(
            [
                ModelResponse(
                    parts=[ToolCallPart("move", {"index": index}, str(index))]
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart("move", "observation " + "x" * 1500, str(index))
                    ]
                ),
            ]
        )
    return messages


def test_summary_policy_uses_a_fresh_stateful_engine_per_run():
    capability = ContextEngineCapability(policy=_policy())
    assert capability.factory is not None
    assert capability.factory() is not capability.factory()


def test_summary_preserves_task_recent_feedback_and_omits_binary(monkeypatch):
    captured = []

    async def summarize(ctx, messages, **kwargs):
        del ctx, kwargs
        captured.append(str(messages))
        return "Earlier moves completed; continue task A."

    monkeypatch.setattr("roborsi.runtime.summary_context.summarize_history", summarize)
    messages = _history()
    messages[2].parts.append(
        UserPromptPart([BinaryContent(b"SECRET_BINARY" * 100, media_type="image/png")])
    )
    engine = make_summary_engine(_policy())
    result = asyncio.run(engine(SimpleNamespace(), messages))
    validate_context(messages, result)
    assert messages[0] in result
    assert messages[-4:] == result[-4:]
    assert len(result) < len(messages)
    assert captured and not any("SECRET_BINARY" in prompt for prompt in captured)


def test_below_trigger_retains_complete_text_history(monkeypatch):
    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return SimpleNamespace(ids=list(text))

    monkeypatch.setattr(
        "roborsi.runtime.summary_context._load_tokenizer",
        lambda policy: FakeTokenizer(),
    )
    policy = ContextPolicy(
        strategy="summary",
        context_window=1050000,
        output_reserve=128000,
        overhead_reserve=32768,
        summary_max_tokens=128000,
        summary_trigger_ratio=0.75,
        adaptive_retention=True,
        keep_turns=8,
        tokenizer_file="/fake/tokenizer.json",
    )
    messages = _history()
    assert (
        asyncio.run(make_summary_engine(policy)(SimpleNamespace(), messages))
        == messages
    )


def test_summary_replaces_cached_summary_across_multiple_compressions(monkeypatch):
    async def summarize(ctx, messages, **kwargs):
        del ctx, messages, kwargs
        return "task A progress"

    monkeypatch.setattr("roborsi.runtime.summary_context.summarize_history", summarize)
    engine = make_summary_engine(
        _policy(
            keep_turns=1,
            cache_initial_task=True,
            cache_initial_context=True,
        )
    )
    messages = _history()

    for index in range(8):
        result = asyncio.run(engine(SimpleNamespace(), messages))
        assert isinstance(result, ContextResult)
        messages, _ = _apply_cache_breakpoints(
            list(result.messages), result.cache_breakpoints
        )
        messages, _ = _apply_cache_breakpoints(messages, ())

        summaries = []
        for message in messages:
            if not isinstance(message, ModelRequest) or len(message.parts) != 1:
                continue
            part = message.parts[0]
            if not isinstance(part, UserPromptPart):
                continue
            content = [part.content] if isinstance(part.content, str) else part.content
            text = "".join(item for item in content if isinstance(item, str))
            if text.startswith(_PREFIX):
                summaries.append(message)
            assert not any(isinstance(item, CachePoint) for item in content)
        assert len(summaries) == 1

        call_id = f"new-{index}"
        messages.extend(
            [
                ModelResponse(parts=[ToolCallPart("move", {"index": index}, call_id)]),
                ModelRequest(
                    parts=[
                        ToolReturnPart("move", "new observation " + "x" * 6000, call_id)
                    ]
                ),
            ]
        )


@pytest.mark.parametrize(
    "values",
    [
        {"context_window": 100, "output_reserve": 100},
        {"summary_trigger_ratio": 1.0},
        {"tokenizer_file": ""},
        {"tokenizer_sha256": "0" * 64},
        {"adaptive_retention": 1},
    ],
)
def test_invalid_summary_policy_fails_early(values):
    with pytest.raises(ValueError):
        _policy(**values)
