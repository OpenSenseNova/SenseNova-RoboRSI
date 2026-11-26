# Copyright 2026 The RoboRSI Contributors.
# Licensed under the Apache License, Version 2.0.

"""CCI transport deadlines and compaction must fit the actual provider."""

import asyncio
from copy import deepcopy

import pytest

from robots.robodojo.agents.submission import (
    cci_protocol,
    execution_profile,
    output_retry,
)
from robots.robodojo.agents.submission.runtime_config import (
    POLICY_DIR,
    resolve_cci_limits,
)
from robots.robodojo.agents.submission.skillpack.context_windows import ContextWindows


@pytest.mark.parametrize("episode", [None, 7200])
def test_slow_cci_request_is_not_silently_capped_at_five_minutes(
    tmp_path, monkeypatch, episode
):
    seen = {}

    class Adapter:
        base_url = "http://127.0.0.1:1/v1"

        def __init__(self, *args, **kwargs):
            seen["timeout"] = kwargs["request_timeout"]

        def start(self):
            pass

        def stop(self):
            pass

    class Capture:
        def __init__(self, *args, **kwargs):
            seen["input_limit"] = kwargs["context_input_token_limit"]
            seen["sampling"] = kwargs["training_metadata"]["sampling"]

        def start(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(cci_protocol, "CciAdapter", Adapter)
    monkeypatch.setattr(output_retry, "GuardedCapture", Capture)
    monkeypatch.setenv("ROBORSI_CODEX_BASE_URL", "http://127.0.0.1:2/v1")
    monkeypatch.setenv("ROBORSI_CODEX_MODEL", "fixture")
    sampling = {"temperature": 1, "top_k": 20, "top_p": 0.95}
    cfg = {
        "provider_kind": "cci",
        "episode_seconds": episode,
        "cci_request_timeout_s": 3600,
        "cci_context_window": 262144,
        "sampling": sampling,
    }
    execution_profile.start_capture(None, tmp_path / "controller", cfg).stop()
    assert seen["timeout"] == 3600
    assert seen["input_limit"] + 65536 < 262144
    assert seen["sampling"] == sampling


def test_cci_deadline_cannot_extend_the_remaining_episode_budget():
    assert (
        resolve_cci_limits({"cci_request_timeout_s": 3600, "episode_seconds": 42})[
            "request_timeout_s"
        ]
        == 42
    )


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), "3600"])
def test_invalid_cci_deadline_is_rejected(value):
    with pytest.raises(ValueError, match="cci_request_timeout_s"):
        resolve_cci_limits({"cci_request_timeout_s": value})


@pytest.mark.parametrize("value", [True, 0, 65536, 262144.0])
def test_context_window_must_fit_the_unchanged_output_allowance(value):
    with pytest.raises(ValueError, match="context window|cci_context_window"):
        resolve_cci_limits({"cci_context_window": value})


def test_context_compacts_before_the_provider_rejects_input_plus_output():
    limit = resolve_cci_limits({"cci_context_window": 262144})["input_token_limit"]
    context = ContextWindows(POLICY_DIR / "skillpack", input_token_limit=limit)
    history = [{"type": "message", "role": "user", "content": "episode"}]
    context.completed(history, history, {"input_tokens": limit})
    request = {"input": history + [{"role": "user", "content": "next"}]}
    original = deepcopy(request)
    compacted = []

    async def compact(source, previous, reasons):
        compacted.append((previous, reasons))
        assert source == original
        return "Only the provider's public summary."

    actual, evidence, raw = asyncio.run(context.prepare(request, compact))
    assert compacted == [(history, ["provider_context_limit_conservative_estimate"])]
    assert request == original and raw == original["input"]
    assert actual["input"][-1] == original["input"][-1]
    assert evidence["input_token_limit"] == limit
    assert (
        limit + cci_protocol.CCI_MAX_OUTPUT_TOKENS + len(context.prompt.encode())
        < 262144
    )


def test_default_limits_preserve_existing_deployments():
    assert resolve_cci_limits({}) == {
        "request_timeout_s": 300,
        "input_token_limit": 200000,
    }
