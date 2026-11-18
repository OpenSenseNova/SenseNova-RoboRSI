# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pinned task identity and the continuous G1/Dex3 action boundary."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from roborsi.evaluation.messages import Task

SOURCES = json.loads(Path(__file__).with_name("sources.json").read_text())
SIMPLE_REVISION = SOURCES["simple_revision"]
ACTION_SCHEMA = {
    "type": "object",
    "required": ["actions"],
    "additionalProperties": False,
    "properties": {
        "actions": {
            "type": "array",
            "minItems": 1,
            "maxItems": 512,
            "items": {
                "type": "array",
                "minItems": 36,
                "maxItems": 36,
                "items": {"type": "number"},
            },
        }
    },
}
EVALUATION_RULES = {
    "success_metric": "native_success",
    "score_metric": "binary_success",
    "aggregation": "case_mean",
    "error_denominator": "valid",
    "score_protocol": "historical_simple_worker_defaults_not_verified_leaderboard",
}


def reset_instruction(state: dict[str, Any]) -> str:
    """Reproduce upstream's instruction from the pinned reset, without loading it."""
    try:
        template = state["dr_state_dict"]["language"]
        name = state["dr_state_dict"]["target"]["name"]
    except (KeyError, TypeError) as error:
        raise ValueError("SIMPLE reset lacks language or target name") from error
    if (
        not isinstance(template, str)
        or not template.strip()
        or template.count("{}") > 1
        or any(c in template.replace("{}", "") for c in "{}")
        or not isinstance(name, str)
        or not name.strip()
    ):
        raise ValueError("unsupported SIMPLE reset instruction template")
    return template.replace("{}", name)


def validate_task(task: Task) -> dict[str, Any]:
    """Reject task, controller, reset and budget drift before creating a simulator."""
    native = task.native
    expected = {
        "alias",
        "level",
        "episode",
        "official_steps",
        "controller",
        "reset_sha256",
        "score_protocol",
    }
    if task.benchmark != "simple" or set(native) != expected:
        raise ValueError("SIMPLE task needs the exact native identity fields")
    if not isinstance(native["alias"], str) or native["alias"] not in SOURCES["tasks"]:
        raise ValueError("unsupported SIMPLE task")
    spec = SOURCES["tasks"][native["alias"]]
    if (
        native["official_steps"] != spec["official_steps"]
        or native["controller"] != spec["controller"]
    ):
        raise ValueError("SIMPLE native budget or controller differs")
    for name, maximum in [("level", 2), ("episode", 9)]:
        if type(native[name]) is not int or not 0 <= native[name] <= maximum:
            raise ValueError("invalid SIMPLE level or episode")
    if task.max_steps not in [spec["official_steps"], 10 * spec["official_steps"]]:
        raise ValueError("SIMPLE supports explicit 1x or 10x budgets")
    expected_id = f"{native['alias']}/L{native['level']}/E{native['episode']:02d}"
    if (
        task.task_id != expected_id
        or task.capabilities.get("action_schema") != ACTION_SCHEMA
    ):
        raise ValueError("SIMPLE task or action schema differs from catalog")
    digest = native["reset_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("SIMPLE reset needs a SHA256 identity")
    if native["score_protocol"] != EVALUATION_RULES["score_protocol"]:
        raise ValueError("unsupported SIMPLE scoring protocol")
    return spec


def validate_actions(data: dict[str, Any], remaining: int) -> list[list[float]]:
    """Validate a complete chunk without clipping targets or dropping rows."""
    if set(data) != {"actions"} or not isinstance(data["actions"], list):
        raise ValueError("expected exactly an actions array")
    rows = data["actions"]
    if not 1 <= len(rows) <= min(remaining, 512):
        raise ValueError("action chunk exceeds remaining native budget")
    for row in rows:
        if not isinstance(row, list) or len(row) != 36:
            raise ValueError("each SIMPLE action needs 36 columns")
        if any(
            type(v) not in (int, float)
            or abs(v) > 3.4028234663852886e38
            or not math.isfinite(v)
            for v in row
        ):
            raise ValueError("actions must be finite float32 numbers")
    return rows
