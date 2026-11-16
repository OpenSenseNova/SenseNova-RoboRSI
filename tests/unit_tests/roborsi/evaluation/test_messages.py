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

"""Concrete RoboRSI messages round-trip through strict JSON envelopes."""

from __future__ import annotations

import json

import pytest
from jsonschema import Draft202012Validator
from jsonschema import ValidationError as SchemaValidationError
from pydantic import ValidationError

from roborsi.evaluation.messages import (
    Action,
    BenchmarkQuery,
    EpisodeStop,
    EvaluationResult,
    MediaRef,
    NativeResult,
    Observation,
    Provenance,
    QueryResult,
    ServiceError,
    StartResult,
    StepResult,
    Task,
    from_envelope,
    message_schema,
    to_envelope,
)
from roborsi.evaluation.protocol import (
    InteractionEnvelope,
    interaction_envelope_schema,
)


def _task() -> Task:
    return Task(
        benchmark="demo",
        task_id="place_block",
        seed=2,
        instruction="Place the block.",
        max_steps=20,
        native={"suite": "demo_suite"},
    )


def test_messages_round_trip_and_publish_json_schema() -> None:
    task = _task()
    observation = Observation(
        benchmark="demo",
        episode_id="episode-1",
        step_index=0,
        data={"state": [0.1, 0.2]},
        media=(
            MediaRef(
                ref="media/front.jpg",
                mime_type="image/jpeg",
                sha256="a" * 64,
                size_bytes=2048,
            ),
        ),
    )
    action = Action(
        benchmark="demo",
        episode_id="episode-1",
        step_index=0,
        data={"joint_targets": [0.0, 0.3]},
    )
    step = StepResult(
        observation=observation.model_copy(update={"step_index": 1}),
        terminated=True,
        truncated=False,
        reward=1.0,
        termination_reason="native_success",
        native={"eval_success": True},
    )
    error = ServiceError(code="agent_error", message="model unavailable")
    start = StartResult(
        task=task, observation=observation, capabilities={"camera": True}
    )
    native = NativeResult(
        task=task,
        episode_id="episode-1",
        success=True,
        score=1.0,
        details={"native_success": True},
    )
    result = EvaluationResult(
        task=task,
        episode_id="episode-1",
        status="success",
        success=True,
        score=1.0,
        native_score_source="native/result.json",
        native_score_sha256="b" * 64,
        provenance=Provenance(agent_version="agent-v1"),
    )

    for kind, message in (
        ("task", task),
        ("start_result", start),
        ("observation", observation),
        ("action", action),
        (
            "benchmark_query",
            BenchmarkQuery(
                benchmark="demo",
                episode_id="episode-1",
                step_index=0,
                query_id="plan-left",
                method="plan_eef",
                arguments={"arm": "left"},
            ),
        ),
        (
            "query_result",
            QueryResult(
                benchmark="demo",
                episode_id="episode-1",
                step_index=0,
                query_id="plan-left",
                data={"status": "Fail"},
            ),
        ),
        (
            "episode_stop",
            EpisodeStop(
                benchmark="demo", episode_id="episode-1", step_index=0, reason="give_up"
            ),
        ),
        ("step_result", step),
        ("native_result", native),
        ("evaluation_result", result),
        ("error", error),
    ):
        envelope = to_envelope(message, benchmark="demo", episode_id="episode-1")
        decoded = InteractionEnvelope.from_dict(
            json.loads(json.dumps(envelope.as_dict()))
        )
        Draft202012Validator(interaction_envelope_schema()).validate(envelope.as_dict())
        assert from_envelope(decoded) == message
        schema = message_schema(kind)
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(message.as_dict())

    invalid_result = result.as_dict()
    invalid_result["success"] = False
    with pytest.raises(SchemaValidationError):
        Draft202012Validator(message_schema("evaluation_result")).validate(
            invalid_result
        )


def test_message_boundaries_reject_invalid_identity_and_payload() -> None:
    task = _task()
    with pytest.raises(ValidationError, match="seed"):
        Task(benchmark="demo", task_id="task", seed=-1, instruction="x", max_steps=1)
    with pytest.raises(ValidationError, match="finite JSON"):
        Task(
            benchmark="demo",
            task_id="task",
            seed=0,
            instruction="x",
            max_steps=1,
            native={"reward": float("nan")},
        )
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Task.model_validate({**task.as_dict(), "unknown": "field"})
    with pytest.raises(ValidationError, match="schema_version"):
        Task.from_dict({**task.as_dict(), "schema_version": 2})
    with pytest.raises(ValueError, match="identity"):
        to_envelope(task, benchmark="other", episode_id="episode-1")

    envelope = InteractionEnvelope(
        "action",
        "demo",
        "episode-1",
        Action(
            benchmark="other", episode_id="episode-1", step_index=0, data={}
        ).as_dict(),
    )
    with pytest.raises(ValueError, match="identity"):
        from_envelope(envelope)

    with pytest.raises(ValidationError, match="case-relative"):
        MediaRef(
            ref="../secret.jpg",
            mime_type="image/jpeg",
            sha256="a" * 64,
            size_bytes=12,
        )

    task.native["reward"] = float("nan")
    with pytest.raises(ValueError, match="finite JSON"):
        task.as_dict()


def test_request_id_is_optional_but_validated_on_the_wire() -> None:
    envelope = to_envelope(
        _task(), benchmark="demo", episode_id="episode-1", request_id="request-1"
    )
    document = envelope.as_dict()
    Draft202012Validator(interaction_envelope_schema()).validate(document)
    assert InteractionEnvelope.from_dict(document).request_id == "request-1"
    with pytest.raises(ValueError, match="request_id"):
        InteractionEnvelope.from_dict({**document, "request_id": ""})
    with pytest.raises(ValueError, match="schema_version"):
        InteractionEnvelope.from_dict({**document, "schema_version": 2})


def test_step_and_result_require_consistent_terminal_state() -> None:
    observation = Observation(benchmark="demo", episode_id="e", step_index=0)
    with pytest.raises(ValidationError, match="termination_reason"):
        StepResult(observation=observation, terminated=True, truncated=False)
    with pytest.raises(ValidationError, match="native success and score"):
        EvaluationResult(
            task=_task(),
            episode_id="e",
            status="success",
            success=False,
            score=0.0,
            provenance=Provenance(),
        )
    with pytest.raises(ValidationError, match="unknown score"):
        EvaluationResult(
            task=_task(),
            episode_id="e",
            status="agent_error",
            success=None,
            score=0.0,
            error=ServiceError(code="agent_error", message="failed"),
            provenance=Provenance(),
        )
