# Copyright 2026 The RoboRSI Contributors.
# Licensed under the Apache License, Version 2.0.

"""Compatibility import for the Bench2Dex Agent."""

from .agents.baseline.factory import CLI_SHA256, FrozenPolicyAgent, make_agent

__all__ = ["CLI_SHA256", "FrozenPolicyAgent", "make_agent"]
