# Copyright 2026 The SenseNova-RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RoboRSI example compacts complete exchanges without rolling its prefix each turn."""

from functools import partial

import pytest
from pydantic_ai import Tool
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from examples.roborsi.model_agent import BudgetedCacheContext
from roborsi.runtime import RuntimeConfig
from roborsi.runtime.context_engine import validate_context
from roborsi.runtime.factory import build_runtime_agent


def run_dialogue(*, tokens=None, threshold=100000, turns=10):
    histories, markers = [], []

    def model(messages, info):
        validate_context(messages, messages)
        histories.append(
            [
                part.tool_call_id
                for message in messages
                for part in message.parts
                if isinstance(part, ToolReturnPart)
            ]
        )
        markers.append(
            (info.model_settings or {}).get("roborsi_cache_tool_call_ids", ())
        )
        request = len(histories)
        return ModelResponse(
            parts=[ToolCallPart("read", {}, f"call-{request}")]
            if request <= turns
            else [TextPart("done")],
            usage=RequestUsage(input_tokens=tokens.get(request, 100))
            if tokens is not None
            else RequestUsage(),
        )

    agent = build_runtime_agent(
        model=FunctionModel(model),
        runtime=RuntimeConfig(
            context_engine_factory=partial(
                BudgetedCacheContext,
                max_input_tokens=threshold,
                max_feedback_turns=4,
                keep_feedback_turns=2,
            )
        ),
        tools=[Tool(lambda: "observation", name="read")],
        capabilities=[],
        system_prompt="task reader",
        max_tokens=100,
    )
    assert agent.run_sync("Read the latest observation.").output == "done"
    return histories, markers


def test_turn_budget_cuts_in_batches_and_retains_previous_cache_boundary():
    histories, markers = run_dialogue()
    assert histories[4] == [f"call-{n}" for n in range(1, 5)]
    assert histories[5] == ["call-4", "call-5"]
    assert histories[6] == ["call-4", "call-5", "call-6"]
    assert histories[7] == [f"call-{n}" for n in range(4, 8)]
    assert histories[8] == ["call-7", "call-8"]
    # The old tool prefix changes on compression; the following round can reuse
    # the newly established prefix, and old discarded exchanges stay absent.
    assert markers[5] == ("call-5",)
    assert set(markers[6]) == {"call-5", "call-6"}
    assert markers[8] == ("call-8",)
    assert all("call-1" not in history for history in histories[5:])


def test_reported_input_budget_can_compact_before_turn_limit():
    histories, markers = run_dialogue(tokens={4: 2000}, threshold=1000, turns=6)
    assert histories[3] == ["call-1", "call-2", "call-3"]
    assert histories[4] == ["call-3", "call-4"]
    assert histories[5] == ["call-3", "call-4", "call-5"]
    assert markers[4] == ("call-4",)
    assert set(markers[5]) == {"call-4", "call-5"}


@pytest.mark.parametrize(
    "values",
    [
        {"max_input_tokens": True},
        {"max_feedback_turns": 0},
        {"keep_feedback_turns": -1},
        {"max_feedback_turns": 8, "keep_feedback_turns": 8},
    ],
)
def test_invalid_user_budgets_fail_before_model_request(values):
    with pytest.raises(ValueError):
        BudgetedCacheContext(**values)


def test_image_budget_keeps_initial_and_latest_observation_before_dispatch():
    from pydantic_ai.messages import BinaryContent, ModelRequest, UserPromptPart

    from roborsi.runtime import CacheState

    def image(label):
        return BinaryContent(label.encode(), media_type="image/png")

    messages = [ModelRequest(parts=[UserPromptPart(["task", image("initial")])])]
    for i in range(8):
        messages.extend(
            [
                ModelResponse(parts=[ToolCallPart("read", {}, str(i))]),
                ModelRequest(
                    parts=[
                        ToolReturnPart("read", "observation", str(i)),
                        UserPromptPart(
                            [image(f"camera-{i}-a"), image(f"camera-{i}-b")]
                        ),
                    ]
                ),
            ]
        )
    policy = BudgetedCacheContext(max_images=5, keep_feedback_turns=4)
    output = policy(None, messages, cache_state=CacheState(None, None))
    validate_context(messages, output.messages)
    assert policy.image_count(output.messages) == 5
    assert output.messages[0] == messages[0]
    assert output.messages[-1] == messages[-1]
    assert [
        part.tool_call_id
        for message in output.messages
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ] == ["6", "7"]
    messages.extend(
        [
            ModelResponse(parts=[ToolCallPart("read", {}, "8")]),
            ModelRequest(parts=[ToolReturnPart("read", "text only", "8")]),
        ]
    )
    # Full SDK history must not bring discarded images back on the next turn.
    output = policy(None, messages, cache_state=CacheState(None, None))
    assert policy.image_count(output.messages) == 5
    assert output.messages[-1] == messages[-1]
    too_small = BudgetedCacheContext(max_images=2)
    with pytest.raises(ValueError, match="initial/current observation"):
        too_small(None, messages, cache_state=CacheState(None, None))
