"""Frozen registered skillpack on this checkout's native XPolicyLab ACK loop."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import threading

from roborsi.planner.codex_cli_config import CodexCliConfig

from .hybrid_agent import HybridController, HybridEmbodiedAgent
from .hybrid_launcher import HybridCodexCliPlanner
from .runtime_config import POLICY_DIR


def clean_agent_exit(planner):
    """A zero process code alone cannot certify a successful provider turn."""
    process = getattr(planner, "_last_agent_process", None)
    segment = getattr(planner, "_last_agent_segment", None)
    if process is None or process.returncode != 0 or segment is None:
        return False
    terminal = None
    try:
        with (Path(segment)/"model.jsonl").open() as stream:
            for line in stream:
                item = json.loads(line)
                kind = item.get("type")
                if kind in {"turn.failed", "error"}:
                    return False
                if kind == "turn.completed":
                    terminal = kind
    except (OSError, ValueError, TypeError, AttributeError):
        return False
    return terminal == "turn.completed"


class SkillpackCodexCliPlanner(HybridCodexCliPlanner):
    """Own MCP/capture per planner instance; no process-global launch patches."""

    def __init__(self, *, profile, context, **kwargs):
        super().__init__(**kwargs)
        self.profile, self.context = profile, context
        self.toolkit = None
        self._robot_server = self._capture = None
        self._last_agent_process = self._last_agent_segment = None
        self._resource_lock = threading.RLock()

    def _sandbox(self, cfg, run, private_home, command, child_env):
        args=super()._sandbox(cfg, run, private_home, command, child_env)
        boundary=args.index('--')
        references=[*self.profile.sandbox_references(),
                    POLICY_DIR/'runtime_records.py', POLICY_DIR/'append_journal.py']
        for source in dict.fromkeys(references):
            source=Path(source).resolve(strict=True)
            # Individual verified files expose only the registered case and
            # helper code. A repo/root bind would expose other case solutions.
            args[boundary:boundary]=['--ro-bind',str(source),str(source)]
            boundary+=3
        return args

    def launch_segment(self, cfg, run, private_home, segment, prompt, session=None):
        from .execution_profile import RobotCodeModeProfile, start_capture
        from .runtime_config import resolve_cci_limits

        with self._resource_lock:
            if self._cancelled.is_set():
                raise RuntimeError("Skillpack planner was cancelled")
            if self.toolkit is None:
                raise RuntimeError("Bind the episode's registered MCP tools before launch")
            if self._robot_server is None:
                self._robot_server = self.toolkit.mcp_server()
                self._robot_server.start()
            if self._capture is None:
                capture_profile = RobotCodeModeProfile(self.profile, self.toolkit.get_tools_spec())
                self._capture = start_capture(capture_profile, Path(run), self.context.model_cfg or {})
            input_limit = 200000
            provider_limits = None
            if (self.context.model_cfg or {}).get("provider_kind") == "cci":
                provider_limits = resolve_cci_limits(self.context.model_cfg)
                input_limit = provider_limits["input_token_limit"]
                cfg = deepcopy(cfg)
                cfg["options"]["codex_compact_tokens"] = input_limit
            overrides = dict(self.config.config_overrides)
            overrides.update({
                "model_providers.runtime.base_url": self._capture.base_url,
                "model_instructions_file": "/workspace/preparation/agent_system.md",
                "model_auto_compact_token_limit": input_limit,
                "model_context_window": input_limit,
                "mcp_servers.robot.url": self._robot_server.url,
                "mcp_servers.robot.enabled": True,
                "mcp_servers.robot.required": True,
                # Keep both call paths on the same registered MCP toolkit.
                # RobotPromptProfile still pins all model-visible declarations.
                "mcp_servers.robot.omit_tools_from": [],
                "mcp_servers.robot.startup_timeout_sec": 30,
                "mcp_servers.robot.tool_timeout_sec": 1200,
                "features.code_mode": True,
                "features.code_mode_only": False,
            })
            if (self.context.model_cfg or {}).get("provider_kind") == "cci":
                overrides["model_catalog_json"] = "/workspace/preparation/cci-model-catalog.json"
            environment = dict(self.config.env)
            for name in ("NO_PROXY", "no_proxy"):
                environment[name] = ",".join(filter(None, [environment.get(name), os.environ.get(name), "127.0.0.1", "localhost"]))
            previous = self.config
            forwarded = tuple(name for name in previous.env_vars if name not in {"NO_PROXY", "no_proxy"})
            self.config = replace(previous, config_overrides=overrides, env=environment, env_vars=forwarded)
            from .execution_profile import episode_prompt
            if not prompt.startswith(episode_prompt(self.profile, cfg, model_cfg=getattr(self.context, "model_cfg", None))):
                raise ValueError("Evaluation prompt differs from the registered profile")
            short = prompt
            Path(segment).mkdir(parents=True, exist_ok=True)
            (Path(segment)/"prompt-assets.json").write_text(json.dumps({
                "profile_id": self.profile.profile_id,
                "task": self.profile.task,
                "vla_backend": getattr(self.profile, "vla_backend", None),
                "base_prompt_sha256": getattr(self.profile, "base_prompt_sha256", None),
                "core_sha256": getattr(self.profile, "manifest", {}).get("sha256", {}).get("agent_core.md"),
                "registered_robot_tools": [s["name"] for s in self.toolkit.get_tools_spec()],
                "model_prompt_bytes": len(short.encode()),
                "operator_limits_in_prompt": False,
                "planner": "SkillpackCodexCliPlanner",
                "source": str(POLICY_DIR),
                "provider_limits": provider_limits,
            }, indent=2)+"\n")
            try:
                process = super().launch_segment(cfg, run, private_home, segment, short, session)
                self._last_agent_process = process
                self._last_agent_segment = Path(segment)
                return process
            finally:
                self.config = previous

    def cancel(self):
        super().cancel()
        with self._resource_lock:
            for name in ("_robot_server", "_capture"):
                resource = getattr(self, name)
                if resource is not None:
                    try:
                        resource.stop()
                    except Exception as error:
                        # Keep terminal acknowledgement independent of proxy cleanup.
                        path = Path(self.context.episode_dir)/"skillpack-cleanup.jsonl"
                        with path.open("a") as stream:
                            stream.write(json.dumps({"resource": name, "error_type": type(error).__name__})+"\n")
                    else:
                        # A bounded stop may time out while the capture still
                        # flushes its final response. Keep ownership on failure
                        # so controller-finally/shutdown can retry that same
                        # resource instead of abandoning an open TrainingStore.
                        setattr(self, name, None)


class SkillpackEmbodiedAgent(HybridEmbodiedAgent):
    """Keep the EmbodiedAgent owner and native controller, replacing its context."""

    def stage_workspace(self, context, workspace, run):
        if context.execution_profile is None:
            raise ValueError("The current task's frozen skillpack profile is required")
        # The legacy socket remains private infrastructure; no Python robot module
        # is published. Experiment.request also verifies the MCP operation thread.
        (Path(run)/"tools/robot.py").unlink(missing_ok=True)
        context.execution_profile.stage_live_workspace(workspace, owner=context.episode_dir)
        if (context.model_cfg or {}).get("provider_kind") == "cci":
            from .execution_profile import stage_cci_catalog
            stage_cci_catalog(workspace, self.model, reasoning_effort=self.reasoning_effort)

    def evaluation_prompt(self, context, cfg):
        from .execution_profile import episode_prompt
        prompt = episode_prompt(context.execution_profile, cfg, model_cfg=getattr(context, "model_cfg", None))
        return prompt

    def make_experiment(self, context, cfg, run):
        from .execution_profile import runtime_modules

        runtime = runtime_modules(POLICY_DIR)
        base = runtime.native_request.make_experiment_class(context.reference, allow_direct_targets=True)
        owner = self

        class RegisteredExperiment(base):
            def finalize(self, reason):
                # The original runner finalizes an exited CLI before returning
                # its summary. Route only a known clean policy exit through the
                # live native owner while the controller/ACK loop is still alive.
                planner = getattr(owner, "_hybrid_planner", None)
                process = getattr(planner, "_last_agent_process", None)
                if (reason == "agent_exited" and self.phase == "evaluation"
                        and cfg.get("options", {}).get("agent_exit_policy") == "native_finish"
                        and clean_agent_exit(planner)
                        and not self.fault and self.latest is not None and not self.latest["done"]
                        and self.active is not None and not self.active.get("recorded")):
                    from .hybrid_resume import write_json
                    write_json(self.run / "agent-exit.json", {
                        "format": "roborsi_agent_exit_v1", "reason": "policy_exit_before_native_done",
                        "policy_status": "incomplete", "native_done_before_finalization": False,
                        "control_step": self.latest["observation"]["control_step"],
                        "agent_returncode": process.returncode, "score": None,
                        "native_finalization_requested": True,
                    })
                    # accept() re-enters finalize with reason=agent_exit, then
                    # uses the normal idempotent terminal recording path.
                    self.accept(self.backend.request(op="finish"), "agent_exit")
                    return
                return super().finalize(reason)

            def checkpoint(self, name):
                from .training_store import storage_mode
                if storage_mode(context.model_cfg) == 'raw':
                    return super().checkpoint(name)
                # Recovery uses the live workspace and native ACK/action chain.
                # Copying every generated image into workspace snapshots doubles
                # storage; terminal archives only need the compact source rows.
                path = self.run/'checkpoints'/name
                path.mkdir(parents=True,exist_ok=True)
                (path/'files.json').write_text(json.dumps({'storage':'live_workspace',
                    'workspace':str(self.run/'workspace')})+'\n')
                return str(path.relative_to(self.run))

            def request(self, req):
                toolkit = getattr(owner, "_skillpack_toolkit", None)
                if toolkit is None or toolkit.operation_thread != threading.get_ident():
                    raise context.reference.runner.InvalidAction("Use the registered robot MCP tools")
                if (getattr(context, "model_cfg", None) or {}).get("vla_backend") == "disabled" and isinstance(req, dict) and (
                    req.get("op") in {"vla_propose", "vla_review"} or "proposal_id" in req
                ):
                    raise context.reference.runner.InvalidAction("VLA is disabled in this TCP-only episode")
                return super().request(req)

        return RegisteredExperiment(cfg, run, context.backend)

    def make_planner(self, context, run, workspace):
        return SkillpackCodexCliPlanner(
            profile=context.execution_profile, context=context,
            reference=context.reference, bubblewrap=context.bubblewrap,
            bubblewrap_sha256=context.bubblewrap_sha256, output_dir=run,
            repo_root=workspace, model=self.model, timeout_s=None,
            dashboard_events=None, reasoning_effort=self.reasoning_effort, config=self.codex_cli,
        )

    def bind_experiment(self, context, planner, experiment, run):
        from .execution_profile import start_toolkit

        proposal_ids = None
        if context.resume:
            # Carry only private ID bookkeeping; old pending motion is not restored.
            for source in dict.fromkeys((context.resume.get("workspace_source"), context.resume["source"])):
                if source:
                    path = Path(source)/"agent/controller/proposal_ids.json"
                    if path.is_file():
                        from .runtime_records import read
                        proposal_ids = read(path)
                        break
        self._skillpack_toolkit = start_toolkit(
            experiment, run, context.episode_dir, context.model_cfg or {}, proposal_ids_seed=proposal_ids)
        planner.toolkit = self._skillpack_toolkit


class SkillpackController(HybridController):
    def cancel(self, timeout=15.0):
        return super().cancel(timeout)

    close = cancel


def make_agent(case_dir, *, config):
    options = dict(config.document.get("agent", {}))
    if (options.get("planner") != "codex_cli" or options.get("reasoning_effort") not in {"medium", "xhigh"}
            or options.get("max_turns") is not None or options.get("max_tokens") is not None):
        raise ValueError("Skillpack requires the external Codex CLI, medium/xhigh and no output-token cap")
    if config.document.get("prompt_file") or config.document.get("skills") or config.document.get("runtime"):
        raise ValueError("The frozen execution profile owns prompts and skills")
    options["codex_cli"] = CodexCliConfig.from_dict(options.get("codex_cli", {}))
    if options["codex_cli"].tools != "external":
        raise ValueError("Skillpack requires its registered robot MCP tools")
    return SkillpackEmbodiedAgent(mcp_servers=(), output_dir=case_dir, local_tools=(), **options)
