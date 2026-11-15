# Copyright 2026 The SenseNova-RoboRSI Authors.
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

"""SDK capture preserves the existing sync and Dashboard planner lifecycles."""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest

import roborsi.planner.codex as codex_module
from roborsi.planner.codex import (
    PROVIDER_ENV_KEY,
    PROVIDER_ID,
    CodexPlanner,
    validate_codex_provider_capture,
)
from roborsi.planner.provider_capture import ProviderCapture, ProviderCaptureConfig
from tests.unit_tests.roborsi.dashboard.test_planner_control_contracts import (
    FakeInteraction,
)
from tests.unit_tests.roborsi.planner.test_codex_contracts import (
    FakeCodex,
    FakeToolkit,
    RecordingSink,
    install_fake_backend,
)
from tests.unit_tests.roborsi.planner.test_provider_capture import (
    response_body,
    upstream,
)


def make_planner(tmp_path, capture):
    return CodexPlanner(
        output_dir=str(tmp_path),
        repo_root=tmp_path,
        output_path=tmp_path / "codex.out",
        dashboard_events=RecordingSink(),
        timeout_s=5,
        provider_capture=capture,
    )


def endpoint(config):
    prefix = f"model_providers.{PROVIDER_ID}.base_url="
    return json.loads(
        next(
            value[len(prefix) :]
            for value in config["config_overrides"]
            if value.startswith(prefix)
        )
    )


def solve(planner, **kwargs):
    return planner.solve(
        system_prompt="system",
        user_message="task",
        toolkit=FakeToolkit(),
        max_turns=3,
        **kwargs,
    )


def test_sync_sdk_requests_capture_without_changing_process_environment(
    tmp_path, monkeypatch
):
    install_fake_backend(monkeypatch)
    urls = []
    closed = []

    class HttpCodex(FakeCodex):
        def __enter__(self):
            url = endpoint(self.config)
            urls.append(url)
            key = self.config["env"][PROVIDER_ENV_KEY]
            response = httpx.post(
                url + "/responses",
                json={"model": "original"},
                headers={"Authorization": f"Bearer {key}"},
            )
            response.raise_for_status()
            return self

        def __exit__(self, *args):
            closed.append(True)
            return super().__exit__(*args)

    monkeypatch.setattr(codex_module.openai_codex, "Codex", HttpCodex)
    with upstream(response_body()) as (url, requests):
        monkeypatch.setenv("CODEX_BASE_URL", url)
        monkeypatch.setenv("CODEX_API_KEY", "SDK-SECRET")
        planner = make_planner(tmp_path, ProviderCaptureConfig(url, PROVIDER_ID))
        results = [solve(planner), solve(planner)]
        assert os.environ["CODEX_BASE_URL"] == url
        assert len(requests) == 2
        assert {k.lower(): v for k, v in requests[0][1].items()}[
            "authorization"
        ] == "Bearer SDK-SECRET"
    assert closed == [True, True]
    assert len(set(urls)) == 2
    for result in results:
        assert (
            result.stats["provider_capture"]["summary"]["nonempty_encrypted_items"] == 1
        )
        path = Path(result.stats["provider_capture"]["output_dir"])
        assert (path / "000001.response.bin").read_bytes() == response_body()
        assert "SDK-SECRET" not in (path / "000001.request.json").read_text()
    assert (
        len(list((tmp_path / "codex-provider-capture").glob("call-*/summary.json")))
        == 2
    )
    for url in urls:
        with pytest.raises(httpx.RequestError):
            httpx.post(url + "/responses", json={}, timeout=0.2)


@pytest.mark.parametrize("loopback_bypass", [False, True])
def test_sdk_capture_uses_child_proxy_and_local_bypass_policy(
    tmp_path, monkeypatch, loopback_bypass
):
    install_fake_backend(monkeypatch)
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)

    class HttpCodex(FakeCodex):
        def __enter__(self):
            response = httpx.post(
                endpoint(self.config) + "/responses",
                json={"model": "original"},
                trust_env=False,
            )
            response.raise_for_status()
            return self

    monkeypatch.setattr(codex_module.openai_codex, "Codex", HttpCodex)
    with upstream(response_body("direct")) as (direct_url, direct_requests):
        with upstream(response_body("proxy")) as (proxy_url, proxy_requests):
            monkeypatch.setenv("HTTP_PROXY", proxy_url.removesuffix("/v1"))
            target = direct_url if loopback_bypass else "http://127.0.0.2:1/v1"
            monkeypatch.setenv("CODEX_BASE_URL", target)
            environment = dict(os.environ)
            planner = make_planner(tmp_path, ProviderCaptureConfig(target, PROVIDER_ID))
            result = solve(planner)
            assert dict(os.environ) == environment
    assert result.error is None
    assert len(proxy_requests) == (0 if loopback_bypass else 1)
    assert len(direct_requests) == (1 if loopback_bypass else 0)
    capture = result.stats["provider_capture"]
    expected = response_body("direct" if loopback_bypass else "proxy")
    assert (
        Path(capture["output_dir"]) / "000001.response.bin"
    ).read_bytes() == expected
    assert capture["summary"]["nonempty_encrypted_items"] == 1


def test_dashboard_sdk_uses_same_capture_and_closes_it(tmp_path, monkeypatch):
    install_fake_backend(monkeypatch)
    closed = []
    urls = []

    class Turn:
        async def stream(self):
            yield {
                "method": "turn/completed",
                "payload": {
                    "turn": {
                        "status": "failed",
                        "error": {"message": "scripted failure"},
                    }
                },
            }

        async def interrupt(self):
            pass

    class Thread:
        async def turn(self, *args, **kwargs):
            return Turn()

    class AsyncCodex:
        def __init__(self, config):
            self.config = config

        async def thread_start(self, **kwargs):
            url = endpoint(self.config)
            urls.append(url)
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url + "/responses", json={"model": "original"}
                )
                response.raise_for_status()
            return Thread()

        async def close(self):
            closed.append(True)

    monkeypatch.setattr(codex_module.openai_codex, "AsyncCodex", AsyncCodex)
    monkeypatch.delenv("CODEX_BASE_URL", raising=False)
    with upstream(response_body()) as (url, _):
        planner = make_planner(tmp_path, ProviderCaptureConfig(url, PROVIDER_ID))
        result = solve(planner, dashboard_interaction=FakeInteraction())
    assert result.error == "scripted failure"
    assert closed == [True]
    assert result.stats["provider_capture"]["summary"]["nonempty_encrypted_items"] == 1
    with pytest.raises(httpx.RequestError):
        httpx.post(urls[0] + "/responses", json={}, timeout=0.2)


def test_capture_cleanup_failure_preserves_sdk_error(tmp_path, monkeypatch):
    install_fake_backend(monkeypatch)
    monkeypatch.delenv("CODEX_BASE_URL", raising=False)

    class FailedCodex(FakeCodex):
        def __enter__(self):
            raise RuntimeError("original SDK failure")

    original_stop = ProviderCapture.stop

    def failed_stop(self, **kwargs):
        original_stop(self, **kwargs)
        raise RuntimeError("cleanup detail")

    monkeypatch.setattr(codex_module.openai_codex, "Codex", FailedCodex)
    monkeypatch.setattr(ProviderCapture, "stop", failed_stop)
    planner = make_planner(
        tmp_path, ProviderCaptureConfig("http://localhost:1/v1", PROVIDER_ID)
    )
    result = solve(planner)
    assert "original SDK failure" in result.error
    assert "capture cleanup" in result.error
    assert result.error_kind == "infrastructure"
    assert result.stop_reason == "cleanup_failed"


@pytest.mark.parametrize(
    "capture,base_url",
    [
        (ProviderCaptureConfig("http://localhost/v1", "other"), None),
        (ProviderCaptureConfig("http://localhost/custom", PROVIDER_ID), None),
        (
            ProviderCaptureConfig("http://localhost/v1", PROVIDER_ID),
            "http://different.example.invalid/v1",
        ),
    ],
)
def test_sdk_capture_rejects_silent_provider_changes(capture, base_url):
    with pytest.raises(ValueError):
        validate_codex_provider_capture(capture, base_url)


def test_sdk_capture_accepts_existing_equivalent_base_url():
    config = ProviderCaptureConfig("http://localhost/v1/", PROVIDER_ID)
    assert validate_codex_provider_capture(config, "http://localhost") == config
    assert validate_codex_provider_capture(None, "http://localhost") is None
