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

"""Retry limits apply to actual provider HTTP attempts, without SDK nesting."""

import asyncio

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from roborsi.llm.retry import RetryLoggingModel, RetryPolicy


@pytest.mark.parametrize(
    ("status", "failures", "expected", "succeeds"),
    [(503, 3, 4, True), (429, 9, 4, False), (401, 9, 1, False), (400, 9, 1, False)],
)
def test_provider_transport_attempt_limit(
    status: int, failures: int, expected: int, succeeds: bool
) -> None:
    attempts = []

    def handle(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) <= failures:
            return httpx.Response(
                status,
                json={
                    "error": {
                        "message": "temporary provider error",
                        "type": "server_error",
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "response-1",
                "object": "chat.completion",
                "created": 1,
                "model": "offline",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "complete"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            async with AsyncOpenAI(
                api_key="test",
                base_url="http://offline.invalid/v1",
                http_client=http,
                max_retries=8,
            ) as client:
                model = RetryLoggingModel(
                    OpenAIChatModel(
                        "offline", provider=OpenAIProvider(openai_client=client)
                    ),
                    policy=RetryPolicy(initial_delay_s=0, max_delay_s=0),
                )
                assert client.max_retries == 0
                return await Agent(model).run("local transport only")

    if succeeds:
        assert asyncio.run(run()).output == "complete"
    else:
        with pytest.raises(ModelHTTPError):
            asyncio.run(run())
    assert len(attempts) == expected
