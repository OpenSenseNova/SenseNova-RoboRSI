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

"""A single EmbodiedAgent factory for any Benchmark with a native action schema."""

from dataclasses import dataclass, field, replace
from pathlib import Path

from pydantic_ai import RunContext
from pydantic_ai.messages import (
    BinaryContent,
    ImageUrl,
    ModelMessage,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from roborsi.embodied_agent import EmbodiedAgent
from roborsi.runtime import (
    CacheBreakpoint,
    CacheState,
    ContextResult,
    RuntimeConfig,
)
from roborsi.runtime.context_engine import recent_turns
from roborsi.runtime.trace import current_trace


def cache_context(
    ctx: RunContext, messages: list[ModelMessage], *, cache_state: CacheState
) -> ContextResult:
    """Keep stable task boundaries and advance sequential tool-feedback caching.

    Resolve against the selected history so compression invalidates old prefixes.
    """
    points = []
    for mi, message in enumerate(messages):
        for pi, part in enumerate(message.parts):
            if isinstance(part, UserPromptPart):
                items = (
                    [part.content] if isinstance(part.content, str) else part.content
                )
                if items:
                    points.append(CacheBreakpoint(mi, pi, 0))
                    if len(items) > 1:
                        points.append(CacheBreakpoint(mi, pi, len(items) - 1))
                break
        if points:
            break
    previous_tools = [
        match.position
        for match in cache_state.resolve(messages)
        if match.reusable_prefix and match.boundary.tool_call_id is not None
    ]
    if previous_tools:
        points.append(previous_tools[-1])
    for mi in range(len(messages) - 1, -1, -1):
        for pi in range(len(messages[mi].parts) - 1, -1, -1):
            if isinstance(messages[mi].parts[pi], ToolReturnPart):
                points.append(CacheBreakpoint(mi, pi))
                return ContextResult(messages, tuple(dict.fromkeys(points)))
    return ContextResult(messages, tuple(points))


@dataclass
class BudgetedCacheContext:
    """Compact in batches, then keep the selected prefix stable while it grows.

    Users choose a soft threshold based on the preceding request's reported
    input tokens and a turn limit for missing usage. Neither estimates the next
    request's exact token count. Keep complete exchanges, initial task images,
    and the latest user input; discarded older feedback is not summarized.
    Instantiate through context_engine_factory to isolate each dialogue.
    """

    max_input_tokens: int = 12000
    max_images: int = 48
    max_feedback_turns: int = 32
    keep_feedback_turns: int = 8
    _anchor: tuple | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        for value in (
            self.max_input_tokens,
            self.max_images,
            self.max_feedback_turns,
            self.keep_feedback_turns,
        ):
            if type(value) is not int or value < 1:
                raise ValueError("context budgets must be positive integers")
        if self.keep_feedback_turns >= self.max_feedback_turns:
            raise ValueError("keep_feedback_turns must be below max_feedback_turns")

    @staticmethod
    def _identity(message: ModelResponse) -> tuple:
        # Usage can be updated after dispatch, so it is not part of the anchor.
        return (
            message.timestamp,
            message.provider_response_id,
            tuple(
                part.tool_call_id
                for part in message.parts
                if isinstance(part, ToolCallPart)
            ),
        )

    @staticmethod
    def image_count(messages: list[ModelMessage]) -> int:
        """Count image blocks before dispatch, including nested tool feedback."""

        def count(value: object) -> int:
            if isinstance(value, ImageUrl):
                return 1
            if isinstance(value, BinaryContent):
                return int(value.media_type.startswith("image/"))
            if isinstance(value, dict):
                return sum(count(item) for item in value.values())
            if isinstance(value, (list, tuple)):
                return sum(count(item) for item in value)
            return 0

        return sum(
            count(getattr(part, "content", None))
            for message in messages
            for part in message.parts
        )

    def __call__(
        self, ctx: RunContext, messages: list[ModelMessage], *, cache_state: CacheState
    ) -> ContextResult:
        responses = [m for m in messages if isinstance(m, ModelResponse)]
        selected = messages
        reason = None
        if self._anchor is not None:
            matches = [
                i
                for i, message in enumerate(responses)
                if self._identity(message) == self._anchor
            ]
            if len(matches) == 1:
                selected = recent_turns(
                    ctx, messages, keep_turns=len(responses) - matches[0]
                )
            else:
                reason = "anchor_missing_or_ambiguous"
        previous = cache_state.previous
        input_tokens = (
            previous.usage.input_tokens
            if previous is not None and previous.usage is not None
            else None
        )
        turn_count = sum(isinstance(m, ModelResponse) for m in selected)
        if input_tokens is not None and input_tokens >= self.max_input_tokens:
            reason = reason or "reported_input_tokens"
        if turn_count > self.max_feedback_turns:
            reason = reason or "feedback_turns"
        if self.image_count(selected) > self.max_images:
            reason = "image_count"
        if reason is not None:
            keep = self.keep_feedback_turns
            while True:
                compacted = recent_turns(ctx, selected, keep_turns=keep)
                if self.image_count(compacted) <= self.max_images:
                    selected = compacted
                    break
                if keep == 1:
                    raise ValueError(
                        "initial/current observation images exceed max_images; "
                        "increase the budget or supply a policy that reduces images"
                    )
                keep -= 1
            if len(responses) >= keep:
                self._anchor = self._identity(responses[-keep])
            trace = current_trace()
            if trace is not None:
                matches = cache_state.resolve(selected)
                trace.emit(
                    "context_budget",
                    {
                        "reason": reason,
                        "previous_input_tokens": input_tokens,
                        "max_input_tokens": self.max_input_tokens,
                        "max_images": self.max_images,
                        "max_feedback_turns": self.max_feedback_turns,
                        "keep_feedback_turns": self.keep_feedback_turns,
                        "before_messages": len(messages),
                        "after_messages": len(selected),
                        "after_images": self.image_count(selected),
                        "retained_feedback_turns": keep,
                        "unusable_previous_boundaries": sum(
                            not match.reusable_prefix for match in matches
                        ),
                    },
                )
        return cache_context(ctx, selected, cache_state=cache_state)


def make_agent(case_dir: Path, *, planner_timeout_s: int = 14400) -> EmbodiedAgent:
    """Create an Agent with an explicit timeout for the entire conversation.

    Runner independently limits each action and the complete case. Credentials
    use OPENAI_API_KEY; model/context settings stay in the source snapshot.
    """
    config = RuntimeConfig.from_file(Path(__file__).with_name("model_agent.yaml"))
    return EmbodiedAgent(
        mcp_servers=[],
        output_dir=case_dir,
        runtime=replace(config, context_engine_factory=BudgetedCacheContext),
        max_turns=20000,
        max_tokens=2048,
        planner_timeout_s=planner_timeout_s,
        reasoning_effort="low",
        action_timeout_s=180.0,
    )
