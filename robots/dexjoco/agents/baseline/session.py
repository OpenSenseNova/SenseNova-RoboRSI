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

"""Adapt the original file interface to EmbodiedAgent's external sessions."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from roborsi.embodied_agent import EmbodiedAgent
from roborsi.planner.codex_cli import preflight_codex_cli


class CliSession:
    """Keep file feedback while EmbodiedAgent owns the model and its worker."""

    FINISH_GRACE_S = 30
    CLEANUP_TIMEOUT_S = 45

    def __init__(self, agent, phase: str, prompt: str) -> None:
        self.agent = agent
        self.phase = phase
        self.private = agent.private if phase == "initial" else agent.private / phase
        self.private.mkdir(parents=True, exist_ok=True)
        self.home = self.private / "codex-home"
        self.episode = None
        self.record = None
        self.cancel_requested = False
        self.started = time.monotonic()
        try:
            cli = agent.cli_config
            if cli is None:
                raise ValueError("Use the declared DexJoCo Agent config to start a CLI")
            executable = cli.executable
            if executable == "codex":
                executable = os.environ.get("ROBORSI_DEXJOCO_CODEX", executable)
            native = preflight_codex_cli(
                replace(cli, executable=executable), repo_root=agent.workspace
            )
            binary = Path(native["executable"])
            python_env = Path(os.environ["ROBORSI_DEXJOCO_POLICY_PYTHON"]).resolve()
            if python_env == Path(sys.prefix).resolve():
                raise ValueError(
                    "Policy Python must be separate from the Agent environment"
                )
            ref = agent.ref
            ref.isolation.prepare_auth(
                self.home, Path(os.environ["ROBORSI_DEXJOCO_AUTH"])
            )
            ref.isolation.probe(
                agent.workspace, self.home, agent.interface, binary, agent.private
            )
            mounts = ref.isolation.mounts(
                agent.workspace,
                self.home,
                agent.interface,
                binary,
                policy_python=python_env,
            )
            probe = (
                "import importlib.util,numpy,scipy,PIL; "
                "assert all(importlib.util.find_spec(n) is None for n in "
                "('dexjoco','gpt6_eval','roborsi','robots')); "
                "print('isolated numerical runtime')"
            )
            subprocess.run(
                mounts + ["--", "/opt/policy-python/bin/python3", "-c", probe],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            # This launcher only execs the existing mount namespace and forwards
            # argv/stdin unchanged. RoboRSI constructs the CLI command, reads its
            # events, applies cancellation and owns process/descendant cleanup.
            launcher = self.private / "isolated-codex"
            launcher.write_text(
                "#!/bin/sh\nexec "
                + shlex.join(mounts + ["--", "/opt/codex"])
                + ' "$@"\n'
            )
            launcher.chmod(0o700)
            isolated = replace(
                cli,
                executable=str(launcher),
                expected_sha256=hashlib.sha256(launcher.read_bytes()).hexdigest(),
            )
            ref.write_json(self.private / "native-cli.json", native)
            self.embodied = EmbodiedAgent(
                mcp_servers=(),
                output_dir=self.private,
                working_dir=agent.workspace,
                planner="codex_cli",
                model=agent.cfg["model"],
                max_turns=None,
                max_tokens=None,
                planner_timeout_s=0,
                action_timeout_s=None,
                reasoning_effort=agent.cfg["reasoning_effort"],
                codex_cli=isolated,
            )
            # Historical accounting and action validation consume these paths;
            # shared planner artifacts remain the sole source of the log bytes.
            self.call_dir = self.private / "codex-cli/call-0001"
            for old, new in (
                ("codex.events.jsonl", "events.jsonl"),
                ("codex.stderr.log", "stderr.log"),
                ("prompt.md", "input.txt"),
            ):
                (self.private / old).symlink_to(Path("codex-cli/call-0001") / new)
            # The persistent EmbodiedAgent worker also owns bubblewrap's
            # --die-with-parent relationship across temporary HTTP threads.
            self.episode = self.embodied.start_episode(prompt, system_prompt="")
        except BaseException:
            try:
                if self.episode is not None:
                    self.episode.close(timeout=self.CLEANUP_TIMEOUT_S)
            finally:
                (self.home / "auth.json").unlink(missing_ok=True)
            raise

    def poll(self) -> int | None:
        """Supply only completion state to the original action-file validator."""
        if not self.episode.done:
            return None
        result = self.episode.wait(timeout=0)
        if result.error:
            raise RuntimeError(f"RoboRSI CLI backend failed: {result.error}")
        return 0

    def actions(self, decision: int, remaining: int):
        """Retain original chunk checks and CLI-authored model identity evidence."""
        return self.agent.ref._wait_action(
            self.agent.root,
            self.agent.cfg,
            self,
            decision,
            remaining,
            self.agent.dual,
            session_private=self.private,
        )

    def close(self) -> dict:
        """Release resources and propagate failures, retaining evidence for retries."""
        ref = self.agent.ref
        marker_error = None
        deadline = time.monotonic() + self.FINISH_GRACE_S + self.CLEANUP_TIMEOUT_S
        try:
            if self.record is None:
                try:
                    ref.write_json(
                        self.agent.inputs / "episode_done.json", {"episode_done": True}
                    )
                except Exception as exc:
                    marker_error = exc
                    self.cancel_requested = True
                    self.episode.cancel()
                try:
                    result = self.episode.wait(timeout=self.FINISH_GRACE_S)
                except TimeoutError:
                    self.cancel_requested = True
                    self.episode.cancel()
                    result = self.episode.wait(timeout=self.CLEANUP_TIMEOUT_S)
                runtime = json.loads((self.call_dir / "runtime.json").read_text())
                self.record = {
                    "phase": self.phase,
                    "started_at_monotonic": self.started,
                    "ended_at": ref.utc_now(),
                    "wall_seconds": time.monotonic() - self.started,
                    "returncode": runtime["exit_code"],
                    "tokens": ref.token_usage(
                        self.private, "codex", returncode=runtime["exit_code"]
                    ),
                    "backend": "roborsi.embodied_agent.codex_cli",
                    "stop_reason": result.stop_reason,
                    "error": result.error,
                    "error_kind": result.error_kind,
                    "cleanup": runtime["cleanup"],
                    "owner_requested_cancel": self.cancel_requested,
                    "prompt_sha256": (
                        hashlib.sha256(
                            (self.call_dir / "input.txt").read_bytes()
                        ).hexdigest()
                        if (self.call_dir / "input.txt").is_file()
                        else None
                    ),
                }
            ref.write_json(self.private / "session_result.json", self.record)
            if marker_error is not None:
                raise marker_error
            self._raise_close_error()
            return self.record
        finally:
            try:
                self.episode.close(timeout=max(0, deadline - time.monotonic()))
            finally:
                (self.home / "auth.json").unlink(missing_ok=True)

    def _raise_close_error(self) -> None:
        record = self.record
        if record["cleanup"]["status"] != "complete":
            raise RuntimeError(f"RoboRSI CLI cleanup failed: {record['cleanup']}")
        cancelled_by_owner = (
            record["owner_requested_cancel"]
            and record["error_kind"] == "cancelled"
            and record["stop_reason"] == "closed"
        )
        if not cancelled_by_owner and (
            record["error"]
            or record["error_kind"]
            or record["stop_reason"] != "completed"
            or record["returncode"] != 0
        ):
            raise RuntimeError(
                f"RoboRSI CLI backend failed ({record['stop_reason']}): "
                f"{record['error'] or record['error_kind'] or record['returncode']}"
            )
