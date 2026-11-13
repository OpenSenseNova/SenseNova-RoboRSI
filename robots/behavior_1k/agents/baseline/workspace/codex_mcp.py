#!/usr/bin/env python3
# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""BEHAVIOR launcher 1.0.0 for the preset's pinned Codex CLI 0.162.0."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def main() -> None:
    """Approve only BEHAVIOR's two local MCP tools and replace this process."""
    executable = os.environ.get("BEHAVIOR_CODEX_EXECUTABLE", "codex")
    resolved = shutil.which(executable)
    if resolved is None:
        raise SystemExit(f"Codex executable not found: {executable}")
    if Path(resolved).resolve() == Path(__file__).resolve():
        raise SystemExit("BEHAVIOR_CODEX_EXECUTABLE must point to the underlying CLI")
    arguments = sys.argv[1:]
    command = 0
    while command < len(arguments):
        option, separator, _ = arguments[command].partition("=")
        if option == "--search":
            command += 1
        elif option in {"--ask-for-approval", "-a"}:
            command += 1 if separator else 2
        else:
            break
    if command < len(arguments) and arguments[command] == "exec":
        approvals = []
        for tool in ("icl_demonstration", "roborsi_action"):
            approvals.extend(
                ["-c", f'mcp_servers.roborsi.tools.{tool}.approval_mode="approve"']
            )
        arguments = [*arguments[: command + 1], *approvals, *arguments[command + 1 :]]
    # Keep the real CLI's version, streams, exit status and signal handling.
    # EmbodiedAgent owns exec, the MCP server and the episode lifecycle.
    os.execv(resolved, [resolved, *arguments])


if __name__ == "__main__":
    main()
