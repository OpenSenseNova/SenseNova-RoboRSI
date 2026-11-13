# Copyright 2026 The RoboRSI Contributors.
# Licensed under the Apache License, Version 2.0.

"""Keep the historical file protocol while RoboRSI owns the CLI process."""

from __future__ import annotations

import hashlib
import shlex
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

from roborsi.embodied_agent import EmbodiedAgent
from roborsi.planner.codex_cli_config import CodexCliConfig


def historical_cli_config(isolation, executable="codex") -> CodexCliConfig:
    """Describe the pinned experiment without inheriting user configuration."""
    flags = [
        "--ignore-user-config",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--sandbox",
        "workspace-write",
    ]
    for feature in isolation.DISABLED:
        flags.extend(["--disable", feature])
    return CodexCliConfig(
        executable=executable,
        expected_version="codex-cli 0.154.0",
        expected_sha256="3188814c35471432d4123203e0eb38e5bddc60226e3d7ddf0e59e649ea140022",
        tools="external",
        inherit_env=False,
        env={"PATH": "/usr/bin:/bin"},
        cli_args=tuple(flags),
        config_overrides={
            "approval_policy": "never",
            "web_search": "disabled",
            "sandbox_workspace_write.network_access": False,
            "allow_login_shell": False,
            "project_doc_max_bytes": 0,
        },
    )


class _PlannerProcess:
    """Expose completion to the unchanged action mailbox, without owning a PID."""

    def __init__(self, session):
        self.session = session

    def poll(self):
        if not self.session.done:
            return None
        result = self.session.wait(timeout=0)
        if result.error:
            raise RuntimeError(f"RoboRSI CLI planner: {result.error}")
        return 0


def episode_type(reference, cli_config: CodexCliConfig):
    """Adapt only CLI startup and cleanup; preserve the original policy protocol."""

    class FrameworkEpisode(reference.bridge.Episode):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            helper = Path(__file__).with_name("step_helper.py")
            shutil.copyfile(helper, self.interface / "step.py")
            reference.core.write_json(
                self.private / "action-transport-runtime.json",
                {
                    "protocol": "historical-decision-envelope",
                    "helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
                    "lock": "flock-with-inflight-decision-guard",
                },
            )

        def start(self):
            isolation = reference.bridge.isolation
            isolation.prepare_auth(self.home, self.auth)
            isolation.probe(
                self.workspace, self.home, self.interface, self.binary, self.private
            )
            mounts = isolation.mounts(
                self.workspace,
                self.home,
                self.interface,
                self.binary,
                numerical_runtime=self.calibrated,
            )
            probe = (
                "import importlib.metadata as m,json,sys; print(sys.version); "
                "print(json.dumps(sorted((d.metadata['Name'],d.version) "
                "for d in m.distributions())))"
            )
            if self.calibrated:
                probe = (
                    "import sys; sys.path.insert(0, '/interface'); "
                    "from kinematics_tool import RobotKinematics; " + probe
                )
            packages = subprocess.check_output(
                mounts + ["--", "/usr/bin/python3", "-c", probe]
            )
            (self.private / "policy-packages.txt").write_bytes(packages)
            # The wrapper adds only the original mount boundary. RoboRSI still
            # chooses argv, checks the entry, launches, records and reaps Codex.
            wrapper = self.private / "isolated-codex"
            wrapper.write_text(
                "#!/bin/sh\nexec "
                + shlex.join(mounts + ["--", "/opt/codex"])
                + ' "$@"\n'
            )
            wrapper.chmod(0o700)
            config = replace(
                cli_config,
                executable=str(wrapper.resolve()),
                expected_sha256=hashlib.sha256(wrapper.read_bytes()).hexdigest(),
            )
            self.embodied_agent = EmbodiedAgent(
                mcp_servers=(),
                output_dir=self.private,
                working_dir=self.workspace,
                planner="codex_cli",
                model=self.config["model"],
                reasoning_effort=self.config["reasoning_effort"],
                max_turns=None,
                max_tokens=None,
                planner_timeout_s=0,
                action_timeout_s=None,
                codex_cli=config,
            )
            (self.private / "prompt.md").write_text(self.prompt)
            # Empty system text preserves the historical initial message. The
            # public EmbodiedEpisode owns the persistent model worker and CLI.
            self.session = self.embodied_agent.start_episode(
                self.prompt, system_prompt=""
            )
            self.cancel_requested = False
            self.proc = _PlannerProcess(self.session)
            self.telemetry.model_process_started = True
            self.telemetry.flush()

        def finish(self):
            """Report planner failures and retain retryable cleanup after I/O errors."""
            if self.finished:
                return
            self.stopped.set()
            cleanup_errors = []
            planner_errors = []
            try:
                self.encoder.shutdown(wait=True, cancel_futures=True)
            except Exception as exc:
                cleanup_errors.append(exc)
            try:
                reference.core.write_json(
                    self.inputs / "episode_done.json", {"episode_done": True}
                )
            except Exception as exc:
                cleanup_errors.append(exc)

            session = getattr(self, "session", None)
            if session is not None:
                # A failed terminal write must not leave a model waiting forever.
                # A completed worker may have failed; only our own cancellation
                # request can make a cancelled result an expected outcome.
                if not cleanup_errors and not session.done:
                    try:
                        session.wait(timeout=5)
                    except TimeoutError:
                        pass
                    except Exception:
                        pass  # Collected below, after mandatory cleanup.
                if not session.done and not session.closed:
                    self.cancel_requested = True
                    session.cancel()
                try:
                    session.close(timeout=40)
                except Exception as exc:
                    cleanup_errors.append(exc)
                if session.done:
                    try:
                        result = session.wait(timeout=0)
                        self.telemetry.codex_returncode = 0 if not result.error else 1
                        if result.error and not (
                            result.error_kind == "cancelled"
                            and result.stop_reason == "closed"
                            and self.cancel_requested
                        ):
                            error = RuntimeError(f"RoboRSI CLI planner: {result.error}")
                            if result.stop_reason == "cleanup_failed":
                                cleanup_errors.append(error)
                            else:
                                planner_errors.append(error)
                    except Exception as exc:
                        self.telemetry.codex_returncode = 1
                        planner_errors.append(exc)

            # Try every cleanup operation even if an earlier one failed. Keep
            # session and owner references alive until a later close can retry.
            try:
                (self.home / "auth.json").unlink(missing_ok=True)
            except Exception as exc:
                cleanup_errors.append(exc)
            try:
                self.telemetry.flush(finished=not cleanup_errors)
            except Exception as exc:
                cleanup_errors.append(exc)
            self.finished = not cleanup_errors
            errors = cleanup_errors + planner_errors
            if errors:
                for extra in errors[1:]:
                    errors[0].add_note(f"Additional shutdown failure: {extra!r}")
                raise errors[0]

    return FrameworkEpisode
