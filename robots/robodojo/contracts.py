# Copyright 2026 The RoboRSI Contributors.
# Licensed under the Apache License, Version 2.0.

"""ARX-X5 dimensions enforced by the pinned native runtime and submission."""


def get_robot_action_dim_info(robot: str) -> dict[str, list[int]]:
    """Return fresh dimensions for the sole supported robot configuration."""
    if robot != "arx_x5":
        raise ValueError("RoboDojo submission requires arx_x5")
    return {"arm_dim": [6, 6], "ee_dim": [1, 1]}
