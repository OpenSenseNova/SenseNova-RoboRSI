"""Stage shared robot skills and registered tools."""
from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_modules(policy_dir):
    from . import install_tool_imports
    install_tool_imports()
    root = Path(policy_dir).resolve()/"skillpack"
    frozen = json.loads((root/"frozen_common.json").read_text())
    for relative, digest in frozen.items():
        path = (root/relative).resolve(strict=True)
        if not path.is_relative_to(root) or _hash(path) != digest:
            raise ValueError("Frozen shared interface differs: "+relative)
    runtime = root/"robot_runtime"
    for name in ("robot_tools", "tcp_pose", "gripper_monitor", "proposal_diagnostics", "execution", "native_request", "prompt_profile"):
        loaded = sys.modules.get(name)
        if loaded is not None and Path(loaded.__file__).resolve() != runtime/(name+".py"):
            raise ValueError("Robot runtime already imported from another source: "+name)
    if str(runtime) not in sys.path:
        sys.path.insert(0, str(runtime))
    return SimpleNamespace(**{name: importlib.import_module(name) for name in (
        "robot_tools", "execution", "native_request", "prompt_profile")})


def _tool_availability(profile, vla_enabled):
    if not vla_enabled and not getattr(profile, "tcp_only", False):
        profile.tcp_only = True
        profile.profile_id += ":tcp_only"
        profile.core = ("EVALUATION CONDITION: VLA is unavailable. Use observe, model-supplied TCP targets, "
                        "and the registered target-calculation helpers. Do not call vla_propose or execute "
                        "proposal_id. VLA alternatives are unavailable. Native task "
                        "conditions, scoring and movement budgets are unchanged.\n\n" + profile.core)
        profile.expected_tools = set(profile.expected_tools) - {"vla_propose"}
        profile.deferred = [s for s in profile.deferred if s["name"] != "mcp__robot__vla_propose"]
        for declaration in profile.deferred:
            if declaration["name"] == "mcp__robot__act":
                body = declaration["description"].split("exec tool declaration:\n", 1)[1]
                body = body.replace("  proposal_id?: string;\n", "")
                body = "\n".join(line for line in body.split("\n")
                                 if not line.strip().startswith("// Copy the current short VLA"))
                body = body.replace("Shared native-step budget for TCP tracking, default 80. Omit for original VLA acceptance; a redundant value at least steps is ignored with a warning, and a smaller value is rejected.",
                                    "Shared native-step budget for TCP tracking, default 80.")
                body = body.replace("Consecutive prefix length; VLA allows 5..50, default 5.",
                                    "Number of supplied TCP waypoints; omit to derive from targets.")
                body = body.replace("Without proposal_id/steps/targets", "Without steps/targets")
                body = body.replace("; no proposal_id or reject flag", "")
                declaration["description"] = _TCP_ACT_DESCRIPTION + "\n\nexec tool declaration:\n" + body
    return profile


_TCP_ACT_DESCRIPTION = (
    "Track model-supplied dual-arm TCP targets with measured-state feedback, or record a due "
    "checkpoint without motion. TCP targets use continuous gripper_opening (0 closed, 1 open). "
    "Return actual native steps, residual errors, current RGB and measured TCP state. "
    "Pose arrival does not verify grasp. Keep reason/assessment concise."
)


def load_skillpack_profile(policy_dir, task, owner, *, vla_backend="pi05", vla_enabled=None,
                           skill_library_mode="general_only"):
    """Load the shared robot skills for the current native task."""
    from .agent_track import validate_mode, load_profile
    validate_mode(skill_library_mode)
    return load_profile(policy_dir, task, owner,
                        vla_backend="disabled" if vla_enabled is False else vla_backend)


def prepare_skillpack_context(policy_dir, task, episode_dir, *, native_identity=None,
                              vla_backend="pi05", vla_enabled=None, skill_library_mode="general_only"):
    """Stage the shared skills in an isolated episode workspace."""
    root = Path(episode_dir).resolve(strict=True)
    profile = load_skillpack_profile(policy_dir, task, root, vla_backend=vla_backend,
                                     vla_enabled=vla_enabled, skill_library_mode=skill_library_mode)
    workspace = root/"prepared-skills"
    workspace.mkdir(exist_ok=True)
    receipt = profile.stage_live_workspace(workspace, owner=root)
    index = json.loads((Path(policy_dir)/"skillpack/task_catalog.json").read_text())
    return SimpleNamespace(workspace=workspace, source_identity={
        "skill_library_mode": skill_library_mode,
        "format": "roborsi_general_skills_v1", "mapping": index["tasks"][task],
        "runtime_subagents_enabled": True, "spatial_memory_enabled": False,
        "profile_id": profile.profile_id, "assets": receipt["assets"],
        "vla_backend": profile.vla_backend, "base_prompt_sha256": profile.base_prompt_sha256,
        "task": task,
    })


def extend_reference(reference):
    """Keep original metadata and ACK protocol, extending only this adapter class."""
    runtime = runtime_modules(Path(__file__).parent)
    from .runtime_records import write as append_state
    # Storage extension only: the pinned native rules/physics are unchanged.
    # Patch every already-imported reference alias, rather than editing its
    # verified source or letting a per-step latest.json rename block actions.
    modules=[reference.runner,reference.hybrid_host,
             sys.modules.get('policylab.common'),sys.modules.get(reference.HybridAdapter.__module__)]
    for module in modules:
        if module is not None and hasattr(module,'write_json'):
            module.write_json=append_state
    return replace(reference, HybridAdapter=runtime.execution.adapter_class(reference.HybridAdapter))


def episode_prompt(profile, cfg, *, model_cfg=None):
    """Use the current task's shared-skill prompt."""
    return profile.episode_prompt(cfg)


def start_toolkit(experiment, run, owner, model_cfg, *, proposal_ids_seed=None):
    """Keep recording and optional operator stop outside the model tool schema."""
    runtime = runtime_modules(Path(__file__).parent)
    RobotTools, ToolInputError = runtime.robot_tools.RobotTools, runtime.robot_tools.ToolInputError
    run, owner = Path(run), Path(owner)
    limit = model_cfg.get("smoke_act_limit")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("smoke_act_limit must be a positive operator-only act count")

    if model_cfg.get("vla_backend") == "disabled" and proposal_ids_seed:
        raise ValueError("TCP-only episodes cannot import VLA proposal IDs")

    class RecordedTools(RobotTools):
        def __init__(self):
            self.operation_thread = None
            self.serial = threading.RLock()
            self.executed_act_calls = self.sequence = 0
            self.closed = False
            from .runtime_config import resolve_vla_config
            vla = resolve_vla_config(model_cfg)
            # Retain the existing multi-waypoint TCP tracker when VLA is disabled.
            setting = "simate_vla" if vla["backend"] == "disabled" else vla["setting"]
            super().__init__(experiment.request, setting=setting, direct_act_enabled=True,
                             image_root=run/"observations", workspace_root=run/"workspace",
                             proposal_ids_path=run/"proposal_ids.json",
                             proposal_ids_seed=proposal_ids_seed, code_first_motion=True,
                             spatial_config=model_cfg.get("spatial_localization"))
            if model_cfg.get("vla_backend") == "disabled":
                self._specs.pop("vla_propose", None)
                spec = self._specs["act"]
                spec["input_schema"]["properties"].pop("proposal_id", None)
                spec["input_schema"]["properties"]["max_native_steps"]["description"] = (
                    "Shared native-step budget for TCP tracking, default 80.")
                spec["input_schema"]["anyOf"] = [{"required": ["targets"]}, {"required": ["review"]}]
                spec["description"] = _TCP_ACT_DESCRIPTION

        def _public(self, result):
            public = super()._public(result)
            if model_cfg.get("skill_library_mode") == "general_only":
                from .agent_track import expose_current_calibration
                public = expose_current_calibration(public, result, observations_root=run/"observations")
            return public

        def dispatch(self, name, arguments):
            if self.closed:
                raise ToolInputError("Operator-authorized execution has ended")
            if model_cfg.get("vla_backend") == "disabled" and (
                name == "vla_propose" or isinstance(arguments, dict) and "proposal_id" in arguments
            ):
                raise ToolInputError("VLA is disabled in this TCP-only episode")
            return super().dispatch(name, arguments)

        def execute_tool(self, name, arguments):
            with self.serial:
                self.operation_thread = threading.get_ident()
                before = experiment.latest["observation"]["control_step"]
                self.sequence += 1
                record = {"sequence": self.sequence, "tool": name, "arguments": deepcopy(arguments),
                          "at": time.time(), "native_step_before": before}
                try:
                    result = super().execute_tool(name, arguments)
                    record["result"] = deepcopy(result.result)
                    return result
                except BaseException as error:
                    record.update(error_type=type(error).__name__, error=str(error))
                    raise
                finally:
                    try:
                        after = experiment.latest["observation"]["control_step"]
                        if name == "act" and after > before:
                            self.executed_act_calls += 1
                        record.update(native_step_after=after, executed_act_calls=self.executed_act_calls)
                        with (run/"tool_calls.jsonl").open("a") as stream:
                            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")
                        if limit is not None and self.executed_act_calls >= limit and not self.closed:
                            self.closed = True
                            status = {"reason": "user_smoke_act_limit", "executed_act_calls": self.executed_act_calls,
                                      "native_step": after, "score": None, "full_episode_evaluation": False,
                                      "native_reset": False, "at": time.time()}
                            for target in (run/"smoke_status.json", owner/"user_smoke_complete.json"):
                                target.write_text(json.dumps(status, indent=2)+"\n")
                            with experiment.lock:
                                try:
                                    experiment.end_episode("user_smoke_act_limit")
                                finally:
                                    experiment.segment_done = True
                    finally:
                        self.operation_thread = None
    return RecordedTools()


class RobotCodeModeProfile:
    """Expose the registered robot schemas on both direct and JS call paths."""
    def __init__(self, profile, tools):
        self.profile = profile
        names = [tool["name"] for tool in tools]
        if len(names) != len(set(names)) or set(names) != set(profile.expected_tools):
            raise ValueError("JS robot bindings must match the registered episode tools")
        self.robot_schemas = {tool["name"]: deepcopy(tool["input_schema"]) for tool in tools}
        self.robot_namespace = dict(type="namespace", name="mcp__robot",
            description="Tools in the mcp__robot namespace.", tools=[dict(
                type="function", name=tool["name"], description=tool.get("description", ""),
                strict=False, parameters=deepcopy(self.robot_schemas[tool["name"]]))
                for tool in sorted(tools, key=lambda tool: tool["name"])])
        self.schema_sha256 = hashlib.sha256(json.dumps(self.robot_namespace,
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def __getattr__(self, name):
        return getattr(self.profile, name)

    def transform(self, request):
        from .turn_output_retry import compaction_request

        groups = [request.get("tools", [])] + [
            item.get("tools", []) for item in request.get("input", [])
            if isinstance(item, dict) and item.get("type") == "additional_tools"]
        if compaction_request(request) and not any(groups):
            # The pinned CLI's explicit text compaction advertises no robot
            # tools. Validate its unchanged core using a per-call profile copy.
            compact_profile = copy(self.profile)
            compact_profile.expected_tools = set()
            result, evidence = compact_profile.transform(deepcopy(request))
            evidence["compact_is_explicitly_tool_free"] = True
            return result, evidence
        normalized = deepcopy(request)
        restored = []
        for index, item in enumerate(normalized.get("input", [])):
            if item.get("type") != "additional_tools" or not item.get("tools"):
                continue
            if not any(tool.get("name") == "mcp__robot" for tool in item["tools"]):
                # Code-mode discovery omits the direct namespace. Reconstruct
                # its declaration from this same MCP toolkit, never
                # from model output or a second set of tool descriptions.
                item["tools"].append(deepcopy(self.robot_namespace))
                restored.append(index)
        result, evidence = self.profile.transform(normalized)
        constrained = set()

        def restore_schemas(specs, path=()):
            for spec in specs:
                full = path + (spec.get("name", ""),)
                if isinstance(spec.get("tools"), list):
                    restore_schemas(spec["tools"], full)
                else:
                    name = spec.get("name", "").rsplit("__", 1)[-1]
                    if name in self.robot_schemas and "robot" in ".".join(full):
                        # CLI 0.160.1 drops maxItems, numeric bounds and other
                        # validation keywords. Restore the complete registered
                        # schema even when the direct namespace was present.
                        spec["parameters"] = deepcopy(self.robot_schemas[name])
                        constrained.add(name)

        for item in result.get("input", []):
            if item.get("type") == "additional_tools":
                restore_schemas(item.get("tools", []))
        evidence["robot_native_namespace_restored_for_code_mode"] = restored
        evidence["robot_registered_schemas_restored"] = sorted(constrained)
        evidence["robot_constraint_schema_version"] = 2
        evidence["robot_constraint_schema_sha256"] = self.schema_sha256
        evidence["changed"] = result != request
        return result, evidence


class CciPromptProfile:
    """Normalize a null wire content field without altering the frozen profile."""
    def __init__(self, profile):
        self.profile = profile

    def transform(self, request):
        normalized = deepcopy(request)
        nulls = []
        for index, item in enumerate(normalized.get("input", [])):
            if "content" in item and item["content"] is None:
                item["content"] = []
                nulls.append(index)
        result, evidence = self.profile.transform(normalized)
        evidence["cci_null_content_as_empty_array"] = nulls
        return result, evidence


def start_capture(profile, run, model_cfg):
    """Capture and normalize actual wire input using the same upstream and key."""
    from roborsi.planner.provider_capture import ProviderCaptureConfig
    from .output_retry import GuardedCapture
    from .runtime_config import resolve_base_url, resolve_model
    from .training_store import storage_mode
    from .hybrid_resume import read_json
    mode=storage_mode(model_cfg)
    owner = Path(run).parent
    upstream = resolve_base_url()
    cci = None
    native_messages_transport = None
    provider_kind = model_cfg.get("provider_kind", "responses")
    if provider_kind == "cci":
        from .cci_protocol import CciAdapter
        seconds = model_cfg.get("episode_seconds")
        cci = CciAdapter(upstream, resolve_model(), owner/"cci-wire",
                         request_timeout=300 if seconds is None else min(300, seconds),
                         sampling=model_cfg.get("sampling"),
                         enable_thinking=model_cfg.get("enable_thinking", True), save_raw=mode=='raw',
                         reasoning_directory=owner/'training/provider_reasoning' if mode=='training' else owner/'cci-wire/provider_reasoning')
        cci.start()
        upstream = cci.base_url
        profile = CciPromptProfile(profile)
    elif provider_kind == "anthropic_messages":
        from .anthropic_messages import AnthropicMessagesTransport, MODEL
        from .fable_profile import FableCliPrefixProfile
        if resolve_model() != MODEL or model_cfg.get('reasoning_effort', 'xhigh') != 'xhigh':
            raise ValueError('Fable Native Messages requires its explicit model and xhigh profile')
        profile = FableCliPrefixProfile(profile)
        native_messages_transport = AnthropicMessagesTransport(
            state_sources=model_cfg.get('native_messages_state_sources', ()),
            prefix_mismatch_behavior=model_cfg.get('native_prefix_policy', 'audit_only'),
            request_deadline_seconds=model_cfg.get('native_request_deadline_seconds', 300))
    elif provider_kind != "responses":
        raise ValueError("provider_kind must be responses, cci or anthropic_messages")
    prior_rows = []
    for usage_file in model_cfg.get("prior_usage_files", []):
        prior_rows.extend(json.loads(line) for line in Path(usage_file).read_text().splitlines() if line)
    capture = GuardedCapture(
        ProviderCaptureConfig(upstream, "runtime", request_encrypted_content=True),
        owner/"api", case_dir=owner, model=resolve_model(), model_label=owner.name,
        expected_effort=model_cfg.get("reasoning_effort", "xhigh"),
        cohort_root=owner, prompt_profile=profile, enforce_paid_cache=provider_kind != "cci",
        case_deadline=model_cfg.get("case_wall_deadline"),
        prior_rows=prior_rows,
        storage_mode=mode,
        native_messages_transport=native_messages_transport,
        training_metadata={'task_id':model_cfg.get('task_name',profile.task if hasattr(profile,'task') else None),
            'model':resolve_model(),'reasoning_effort':model_cfg.get('reasoning_effort','xhigh'),
            'provider_kind':provider_kind,'sampling':model_cfg.get('sampling'),
            **({'provider_summary_manifest':'provider_summaries/manifest.json',
                'provider_summary_semantics':'provider_returned_summary_separate_from_reasoning_content'}
               if provider_kind=='anthropic_messages' and mode=='training' else {}),
            **({'provider_reasoning_manifest':'provider_reasoning/manifest.json'} if provider_kind=='cci' and mode=='training' else {}),
            'enable_thinking':model_cfg.get('enable_thinking',True),
            'source_runtime':str(owner),'harness_commit':model_cfg.get('harness_commit'),
            'source_identity':read_json(owner.parent/'episode-owner.json') if (owner.parent/'episode-owner.json').is_file() else None},
    )
    try:
        capture.start()
    except BaseException:
        if cci is not None:
            cci.stop()
        raise
    if cci is not None:
        original_stop = capture.stop
        def stop():
            try:
                original_stop()
            finally:
                cci.stop()
        capture.stop = stop
    return capture


def stage_cci_catalog(workspace, model, *, reasoning_effort="xhigh"):
    """Declare the custom model's exec/wait transport, without extra instructions."""
    template = Path(__file__).parent/"cci_model_template.json"
    value = json.loads(template.read_text())
    value["models"][0].update(slug=model, display_name=model, default_reasoning_level=reasoning_effort)
    path = Path(workspace)/"preparation/cci-model-catalog.json"
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")
    return path
