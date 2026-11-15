# Copyright 2026 The RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Stable, dependency-free contracts for the BEHAVIOR-1K adapter."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, Sequence

ACTION_DIM = 21


class ActionValidationError(ValueError):
    """Raised when a model action cannot be sent to the simulator."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ActionBounds:
    low: tuple[float, ...] | None = None
    high: tuple[float, ...] | None = None


def _as_sequence(action: Any) -> Sequence[Any]:
    if isinstance(action, (list, tuple)):
        return action
    tolist = getattr(action, "tolist", None)
    if callable(tolist):
        converted = tolist()
        if isinstance(converted, list):
            return converted
    raise ActionValidationError(
        "invalid_type", "action must be a one-dimensional JSON array"
    )


def validate_action(
    action: Any,
    bounds: ActionBounds | None = None,
    *,
    action_dim: int = ACTION_DIM,
) -> list[float]:
    """Validate and normalize one official R1Pro action vector.

    No clipping is performed. Clipping would hide model errors and make the
    trajectory differ from the official evaluator's action semantics.
    """

    values = _as_sequence(action)
    if len(values) != action_dim:
        raise ActionValidationError(
            "invalid_length",
            f"R1Pro action must contain exactly {action_dim} values; got {len(values)}",
        )

    normalized: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ActionValidationError(
                "invalid_value",
                f"action[{index}] must be a finite number; got {type(value).__name__}",
            )
        number = float(value)
        if number != number or number in (float("inf"), float("-inf")):
            raise ActionValidationError(
                "non_finite",
                f"action[{index}] must be finite; got {value!r}",
            )
        normalized.append(number)

    if bounds is not None:
        if bounds.low is not None and len(bounds.low) != action_dim:
            raise ValueError(f"action lower bounds must have length {action_dim}")
        if bounds.high is not None and len(bounds.high) != action_dim:
            raise ValueError(f"action upper bounds must have length {action_dim}")
        for index, number in enumerate(normalized):
            if bounds.low is not None and number < bounds.low[index]:
                raise ActionValidationError(
                    "out_of_bounds",
                    f"action[{index}]={number} is below lower bound {bounds.low[index]}",
                )
            if bounds.high is not None and number > bounds.high[index]:
                raise ActionValidationError(
                    "out_of_bounds",
                    f"action[{index}]={number} is above upper bound {bounds.high[index]}",
                )
    return normalized


# The ordered public proprio_obs list in the pinned official r1pro.yaml.
R1_PROPRIO_LAYOUT = (
    ("base_qvel", 3),
    ("arm_left_qpos", 7),
    ("arm_left_qvel", 7),
    ("eef_left_pos", 3),
    ("eef_left_quat", 4),
    ("gripper_left_qpos", 2),
    ("gripper_left_qvel", 2),
    ("arm_right_qpos", 7),
    ("arm_right_qvel", 7),
    ("eef_right_pos", 3),
    ("eef_right_quat", 4),
    ("gripper_right_qpos", 2),
    ("gripper_right_qvel", 2),
    ("trunk_qpos", 4),
    ("trunk_qvel", 4),
)


def public_proprio(vector: list) -> dict:
    """Decode only the already-public sensor vector, never query hidden robot state."""
    if len(vector) != sum(size for _, size in R1_PROPRIO_LAYOUT):
        raise ValueError(
            "Official proprio vector does not match pinned R1Pro observation layout"
        )
    result, offset = {}, 0
    for name, size in R1_PROPRIO_LAYOUT:
        result[name] = vector[offset : offset + size]
        offset += size
    return result
