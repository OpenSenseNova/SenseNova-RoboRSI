# Copyright 2026 The RoboRSI Contributors.
# Licensed under the Apache License, Version 2.0.

"""Historical action envelope with an interruption-safe, process-owned lock."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import sys
import time
from pathlib import Path


def submit(
    payload_path: Path,
    workspace: Path = Path("/workspace"),
    transport: Path | None = None,
) -> dict:
    """Submit once, or wait for the same interrupted in-flight submission.

    The persistent lock inode must never be unlinked: flock ownership disappears
    when a process exits, including SIGKILL. A durable decision marker prevents a
    later helper from resubmitting an action while that chunk is still executing.
    An ambiguous interruption before publication fails closed rather than replaying.
    """
    root = workspace / "input"
    transport = transport or workspace / ".action_transport"
    payload_path = Path(payload_path)
    if payload_path.resolve().is_relative_to(transport.resolve()):
        raise ValueError(
            "Save the action payload outside the reserved transport directory"
        )
    payload = json.loads(payload_path.read_text())
    descriptor = os.open(
        transport / "submission.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "r+") as lock:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise ValueError("Submission lock must be a regular file")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("An action submission helper is already active") from exc
        if (root / "episode_done.json").exists():
            return {"episode_done": True}
        current = json.loads((root / "observation.json").read_text())
        request = {"decision": current["decision"], "payload": payload}
        wire = json.dumps(request, allow_nan=False)
        identity = hashlib.sha256(wire.encode()).hexdigest()
        journal = transport / "submission.json"
        previous = json.loads(journal.read_text()) if journal.exists() else None
        if previous is not None and previous["decision"] > current["decision"]:
            raise RuntimeError("Submission decision is ahead of the observation")
        if previous is not None and previous["decision"] == current["decision"]:
            if previous["request_sha256"] != identity:
                raise RuntimeError(
                    "Previous action submission is still in flight; wait for the "
                    "next observation before submitting a different action"
                )
            # Rejoin the wait only; never write the request a second time.
        else:
            target = transport / "request.json"
            if target.exists():
                raise RuntimeError("An action submission is already pending")
            marker = transport / "submission.tmp"
            marker.write_text(
                json.dumps(
                    {"decision": current["decision"], "request_sha256": identity}
                )
            )
            marker.replace(journal)
            temporary = transport / "request.tmp"
            temporary.write_text(wire)
            temporary.replace(target)
        while True:
            if (root / "episode_done.json").exists():
                return {"episode_done": True}
            next_observation = json.loads((root / "observation.json").read_text())
            if next_observation["decision"] > current["decision"]:
                return next_observation
            time.sleep(0.1)


if __name__ == "__main__":
    print(json.dumps(submit(Path(sys.argv[1])), allow_nan=False))
