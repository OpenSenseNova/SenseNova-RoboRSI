"""Robot observation, proposal, target calculation and execution tools.

The bridge retains full proposals, permissions, IK, native state and budgets.
The public tool interface uses grasp-centre TCP poses; native bridge/IK poses
remain link6. The adapter also samples trajectories for display only.
"""

from __future__ import annotations

import base64
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import socket
import threading
from typing import Any, Callable

from spatial_tool.contracts import tool_spec as spatial_tool_spec
from spatial_tool.service import SpatialLocalizationTool
from spatial_tool.snapshot import public_snapshot

SPATIAL_MODE = "pixel"
SPATIAL_TOOL_NAME = "spatial_localize_pixel"

from proposal_diagnostics import PREFIX_FIELDS, PREVIEW_STEPS
from tcp_pose import (
    POSE_CONTRACT,
    link6_to_public_pose,
    link6_to_tcp_target,
    tcp_to_link6_target,
    validate_gripper_opening,
)


class ToolInputError(ValueError):
    """A recoverable public tool argument error."""


class RobotBridge:
    """Send one request to an existing episode; never retry uncertain calls."""

    def __init__(self, socket_path: str, timeout: float = 1200):
        self.socket_path = socket_path
        self.timeout = timeout

    def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(self.timeout)
            connection.connect(self.socket_path)
            connection.sendall((json.dumps(request, allow_nan=False) + "\n").encode())
            with connection.makefile("rb") as stream:
                line = stream.readline(16 * 1024 * 1024 + 1)
        if not line or len(line) > 16 * 1024 * 1024:
            raise RuntimeError("Incomplete robot reply; observe before any further motion")
        result = json.loads(line)
        if "error" in result:
            # This is a complete, explicit rejection from the original bridge.
            raise ToolInputError(result["error"])
        return result


def _object(properties: dict, required: list[str]) -> dict:
    return dict(type="object", properties=properties, required=required,
                additionalProperties=False)


TEXT = dict(type="string", minLength=1)
ARM_NAME = dict(type="string", enum=["left", "right"])
TOOL_NAMES = frozenset(("observe", "vla_propose", "act", "hold_current",
                        "close_in_place", "move_relative", SPATIAL_TOOL_NAME))
REVIEW = _object({key: TEXT for key in (
    "observed_state", "demo_comparison", "why_not_done", "next_plan", "ending_check"
)}, ["observed_state", "demo_comparison", "why_not_done", "next_plan", "ending_check"])
ARM = _object({
    "position": dict(type="array", items=dict(type="number"), minItems=3, maxItems=3,
                     description="Absolute gripping-face-centre TCP XYZ in environment-origin metres."),
    "pitch_deg": dict(type="number", minimum=-180, maximum=180,
                      description="Degrees about environment +X, relative to the downward reference."),
    "roll_deg": dict(type="number", minimum=-90, maximum=90,
                     description="Degrees about environment +Y, applied after pitch."),
    "yaw_deg": dict(type="number", minimum=-180, maximum=180,
                    description="Degrees about environment +Z, applied after roll."),
    "gripper_opening": dict(type="number", minimum=0, maximum=1,
                            description="Continuous normalized opening: 0 fully closed, 1 fully open."),
}, ["position", "pitch_deg", "roll_deg", "yaw_deg", "gripper_opening"])
DUAL = _object({"left": ARM, "right": ARM}, ["left", "right"])
# The private bridge retains native quaternion poses and continuous openings.
NATIVE_ARM = _object({
    "position": dict(type="array", items=dict(type="number"), minItems=3, maxItems=3),
    "quaternion_wxyz": dict(type="array", items=dict(type="number"), minItems=4, maxItems=4),
    "gripper_opening": dict(type="number", minimum=0, maximum=1),
}, ["position", "quaternion_wxyz", "gripper_opening"])
NATIVE_DUAL = _object({"left": NATIVE_ARM, "right": NATIVE_ARM}, ["left", "right"])


def tool_specs() -> list[dict[str, Any]]:
    """Expose robot operations and read-only target calculations."""
    return [
        dict(name="observe", description="Read current RGB, measured TCP positions, pitch/roll/yaw degrees relative to the downward reference, permissions and budgets without advancing physics. Display floats use at most three decimal places.", input_schema=_object({}, [])),
        dict(name="vla_propose", description="Predict without moving. Prefer localized code-computed TCP motion when geometry is usable. VLA is also available when localization is unavailable or TCP tracking stalls; no localized point or distance threshold is required. Choose the accepted prefix from the preview and current feedback within the native 5..50-step range. Keep the same short visual instruction for the same subgoal; put metric corrections in TCP targets. Give a short English instruction for the selected work, naming the observed object and destination when needed. Follow the supplied subgoal wording rules; use separate short sentences and necessary conditions. Omit reasoning, code and unrelated future plans. Return proposal_id and preview at steps 5,10,15,20,25,30,35,40,45,50 only. Each arm has pos_m (TCP XYZ), angles_deg ([pitch about +X, roll about +Y, yaw about +Z] from downward), grip (0 closed, 1 open), and measured-start prefix diagnostics: delta_cm, path_cm, max_step_cm, max_rotation_step_deg, max_joint_step_rad, grip_delta and grip_range (including start). Row bounds are cumulative existing warnings through that step. Statistics include every original step in the prefix. Read with RGB; predictions do not establish grasp or safety. No full 50-point inspection. A new proposal replaces the old one; act uses its stored original joint prefix.", input_schema=_object({"subgoal": {**TEXT, "maxLength": 2000}}, ["subgoal"])),
        dict(name="act", description="Accept stored original VLA joint steps, or track model-supplied TCP targets with internal measured-state feedback and IK. TCP targets use continuous gripper_opening (0 closed, 1 open). Supplying targets declines the proposal. Return status, actual native steps, residual errors, current RGB and measured TCP state. Inspect interrupted results before choosing another action; an interrupted VLA proposal cannot be reused. Pose arrival does not verify grasp. Keep reason/assessment concise. Only when observe reports review_due, act.review records legacy checkpoint analysis; it does not determine the native score.", input_schema=_object({
            "proposal_id": {**TEXT, "description": "Copy the current short VLA proposal_id, e.g. p1; omit when supplying targets."},
            "steps": dict(type="integer", minimum=1, maximum=50,
                          description="Consecutive prefix length; VLA allows 5..50, default 5."),
            "targets": dict(type="array", items=DUAL, minItems=1, maxItems=50,
                            description="Model-chosen dual-arm TCP waypoints; no proposal_id or reject flag. Harness tracks each to 3mm/3deg before advancing. A waypoint may consume multiple native steps; model-side interpolation is unnecessary."),
            "max_native_steps": dict(type="integer", minimum=1, maximum=160,
                                     description="Shared native-step budget for TCP tracking, default 80. Omit for original VLA acceptance; a redundant value at least steps is ignored with a warning, and a smaller value is rejected."),
            "reason": TEXT,
            "assessment": _object({"current_subgoal": TEXT}, ["current_subgoal"]),
            "review": {**REVIEW, "description": "Legacy harness checkpoint record, only when observe reports review_due. These fields are not native scoring conditions. Without proposal_id/steps/targets, records the due checkpoint without motion."},
        }, ["reason", "assessment"])),
        dict(name="hold_current", description="Calculate one full dual-arm TCP target from the latest measured state, preserving both poses and gripper openings. Read-only: no motion, VLA call, physics step or proposal replacement. Returns control_step, computed_only and targets; inspect them, then call act with targets to execute. Recalculate after any intervening action.", input_schema=_object({}, [])),
        dict(name="close_in_place", description="Calculate one full dual-arm TCP target from the latest measured state. Keep both poses and the other gripper unchanged; set the selected arm's opening (default 0, fully closed). Read-only; does not close or verify grasp. Returns control_step, computed_only and targets; inspect them, then call act with targets. Recalculate after any intervening action.", input_schema=_object({
            "arm": ARM_NAME,
            "opening": dict(type="number", minimum=0, maximum=1, default=0,
                            description="Selected gripper's target opening, 0 closed and 1 open."),
        }, ["arm"])),
        dict(name="move_relative", description="Calculate one full dual-arm TCP target from the latest measured state. Add delta_m to the selected arm's TCP XYZ in environment axes; preserve both orientations, gripper openings and the other arm. Read-only; no motion or collision planning. Returns control_step, computed_only and targets; inspect them, then call act with targets. Recalculate after any intervening action.", input_schema=_object({
            "arm": ARM_NAME,
            "delta_m": dict(type="array", items=dict(type="number"), minItems=3, maxItems=3,
                            description="Environment-frame XYZ offsets in metres, not tool-local axes."),
        }, ["arm", "delta_m"])),
        spatial_tool_spec(),
    ]


def sparse_proposal(proposal: dict[str, Any]) -> dict[str, Any]:
    """Join ten public targets and prefix diagnostics; retain no dense output."""
    targets = proposal.get("targets")
    if not isinstance(targets, list) or len(targets) != 50:
        raise RuntimeError("Expected the configured full 50-target proposal")
    violations = proposal.get("out_of_bounds", [[] for _ in range(50)])
    if not isinstance(violations, list) or len(violations) != 50:
        raise RuntimeError("Expected full-trajectory bound checks")
    aid = proposal.get("review_aid") or {}
    prefixes = aid.get("prefixes", {})
    available = aid.get("status") == "available" and all(
        all(key in prefixes.get(str(step), {}).get(arm, {}) for key in PREFIX_FIELDS)
        for step in PREVIEW_STEPS for arm in ("left", "right"))
    result = dict(proposal_id=proposal["proposal_id"])
    if "control_step" in proposal:
        result["control_step"] = proposal["control_step"]
    result["preview"] = []
    for step in PREVIEW_STEPS:
        row = dict(step=step)
        for arm in ("left", "right"):
            pose = link6_to_tcp_target(targets[step-1][arm])
            value = dict(pos_m=pose["position"],
                         angles_deg=[pose[key] for key in ("pitch_deg", "roll_deg", "yaw_deg")],
                         grip=pose["gripper_opening"])
            if available:
                value.update({key: deepcopy(prefixes[str(step)][arm][key]) for key in PREFIX_FIELDS})
            row[arm] = value
        # Keep warnings at skipped steps; omit duplicate occurrences only.
        row["bounds"] = list(dict.fromkeys(flag for flags in violations[:step] for flag in flags))
        result["preview"].append(row)
    if not available:
        result["diagnostics"] = "unavailable"
    if (proposal.get("fk_validation") or {}).get("passed") is False:
        result["fk_validation"] = deepcopy(proposal["fk_validation"])
    if proposal.get("gripper_clipped_values"):
        result["gripper_clipped_values"] = proposal["gripper_clipped_values"]
    return _round_display(result)


def _round_display(value: Any) -> Any:
    """Round model-facing float outputs; preserve IDs, integers and booleans."""
    if type(value).__module__.split(".")[0] == "numpy":
        value = value.tolist()
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Public output numbers must be finite")
        rounded = round(float(value), 3)
        return 0.0 if rounded == 0 else rounded
    if isinstance(value, (list, tuple)):
        return [_round_display(item) for item in value]
    if isinstance(value, dict):
        return {key: _round_display(item) for key, item in value.items()}
    return value


class RobotTools:
    """Bind public operations to one existing, original episode bridge."""

    def __init__(self, call: Callable[[dict], dict], *, setting: str,
                 direct_act_enabled: bool = False, image_root: Path | None = None,
                 workspace_root: Path | None = None,
                 proposal_ids_path: Path | None = None, proposal_ids_seed: dict | None = None,
                 code_first_motion: bool = False, spatial_config: dict | None = None):
        if setting not in ("pi05_vla", "simate_vla", "no_vla"):
            raise ValueError("Explicit pi05_vla, simate_vla or no_vla setting required")
        self.code_first_motion = code_first_motion
        self.call = call
        self.setting = setting
        self.direct_act_enabled = direct_act_enabled
        self.image_root = image_root.resolve(strict=True) if image_root else None
        self.workspace_root = workspace_root.resolve(strict=True) if workspace_root else None
        self._lock = threading.Lock()
        self._spatial = SpatialLocalizationTool(self._observe, self.image_root, config=spatial_config)
        self._specs = {spec["name"]: spec for spec in tool_specs()}
        self._proposal_ids_path = proposal_ids_path
        self._proposal_ids = dict(proposal_ids_seed or {})
        if proposal_ids_path is not None and proposal_ids_path.exists():
            from importlib import import_module
            self._proposal_ids.update(import_module('XPolicyLab.policy.SenseNova-RoboRSI.runtime_records').read(proposal_ids_path))
        if (any(not isinstance(k, str) or not k or not isinstance(v, str)
                or re.fullmatch(r"p[1-9][0-9]*", v) is None
                for k, v in self._proposal_ids.items())
                or len(set(self._proposal_ids.values())) != len(self._proposal_ids)):
            raise ValueError("Invalid private proposal ID mapping")
        self._proposal_sequence = max((int(v[1:]) for v in self._proposal_ids.values()), default=0)
        if proposal_ids_path is not None:
            self._save_proposal_ids(self._proposal_ids)

    def _save_proposal_ids(self, mapping: dict) -> None:
        if self._proposal_ids_path is not None:
            from importlib import import_module
            import_module('XPolicyLab.policy.SenseNova-RoboRSI.runtime_records').write(self._proposal_ids_path,mapping)

    def _public_proposal(self, proposal: dict) -> dict:
        public = sparse_proposal(proposal)
        native_id = proposal["proposal_id"]
        if native_id not in self._proposal_ids:
            mapping = {**self._proposal_ids, native_id: f"p{self._proposal_sequence + 1}"}
            # Persist before exposing the alias, so reconnects never recycle it.
            self._save_proposal_ids(mapping)
            self._proposal_ids = mapping
            self._proposal_sequence += 1
        public["proposal_id"] = self._proposal_ids[native_id]
        return public

    def get_tools_spec(self) -> list[dict]:
        return deepcopy(list(self._specs.values()))

    def _observe(self) -> dict:
        return self.call({"op": "observe"})

    def _public(self, result: dict) -> dict:
        public = deepcopy(result)
        observation = public.get("observation", {})
        spatial = observation.get("spatial_localization")
        if spatial and "case_path" in spatial:
            observation["spatial_localization"] = public_snapshot(spatial)
            observation["spatial_localization"]["known_support_planes"] = deepcopy(self._spatial.config["support_planes"])
        observation.get("additional_info", {}).pop("spatial_calibration", None)
        state = observation.get("state", {})
        for arm in ("left", "right"):
            key = arm + "_ee_pose"
            if key in state:
                state[key] = link6_to_public_pose(state[key])
        public["pose_contract"] = deepcopy(POSE_CONTRACT)
        if self.code_first_motion:
            public["motion_policy"] = {
                "name": "spatial_code_preferred_flexible_vla",
                "preferred_control": "localized_code_tcp_with_observed_feedback",
                "vla_fallbacks": ["localization_unavailable", "tcp_tracking_stalled", "contact_assistance"],
                "localization_required_for_vla": False,
                "vla_step_range": [5, 50],
                "required_vla_fields": ["subgoal"],
                "localization_interfaces": ["robot." + SPATIAL_TOOL_NAME],
                "next_action": "Inspect each result; use direct TCP refinement or a reviewed VLA prefix, then observe and switch back to TCP when practical.",
            }
        if public.get("pending_vla_proposal") is not None:
            public["pending_vla_proposal"] = self._public_proposal(public["pending_vla_proposal"])
        # Convert from full-precision native poses first, then quantize only
        # the model-visible copy. Native measurements and execution stay raw.
        return _round_display(public)

    def _review(self, latest: dict, review: dict | None) -> dict:
        if latest.get("observation", {}).get("review_due") and review is None:
            raise ToolInputError("Include the checkpoint evidence in act.review")
        if review is not None:
            return self.call({"op": "review", "analysis": review})
        return latest

    def _calculate_targets(self, name: str, arguments: dict) -> dict:
        """Use full-precision current state; quantize only the returned copy."""
        latest = self._observe()
        try:
            observation = latest["observation"]
            state = observation["state"]
            target = {}
            for arm in ("left", "right"):
                pose = state[arm+"_ee_pose"]
                if type(pose).__module__.split(".")[0] == "numpy":
                    pose = pose.tolist()
                target[arm] = link6_to_public_pose(pose)
                target[arm]["gripper_opening"] = validate_gripper_opening(
                    float(state[arm+"_ee_joint_state"][0]))
            if name == "close_in_place":
                target[arguments["arm"]]["gripper_opening"] = arguments.get("opening", 0.)
            elif name == "move_relative":
                arm = arguments["arm"]
                target[arm]["position"] = [value+delta for value, delta in
                                           zip(target[arm]["position"], arguments["delta_m"])]
            return _round_display(dict(control_step=observation["control_step"],
                                       computed_only=True, targets=[target]))
        except (KeyError, TypeError, ValueError, IndexError) as error:
            raise ToolInputError("Latest measured state cannot form finite dual-arm TCP targets") from error





    def dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute once, preserving native rejections and uncertain failures."""
        from jsonschema import Draft202012Validator, ValidationError

        if name not in self._specs:
            raise ToolInputError(f"Unknown robot tool: {name}")
        if name == SPATIAL_TOOL_NAME:
            with self._lock:
                return self._spatial.locate(arguments)
        try:
            json.dumps(arguments, allow_nan=False)
            Draft202012Validator(self._specs[name]["input_schema"]).validate(arguments)
        except (ValidationError, ValueError, TypeError) as error:
            raise ToolInputError(str(error)) from error
        with self._lock:
            if name == "observe":
                return self._public(self._observe())
            if name in {"hold_current", "close_in_place", "move_relative"}:
                return self._calculate_targets(name, arguments)
            if name == "vla_propose":
                if self.setting == "no_vla":
                    raise ToolInputError("VLA is disabled for this episode")
                latest = self._observe()
                if latest.get("done") or latest.get("observation", {}).get("review_due"):
                    raise ToolInputError("Episode ended or checkpoint review is due")
                pending = latest.get("pending_vla_proposal")
                if pending is not None:
                    self.call(dict(op="vla_review", proposal_id=pending["proposal_id"],
                                   decision="reject", targets=None,
                                   reason="Replace the unexecuted proposal with a new prediction",
                                   assessment={"current_subgoal": arguments["subgoal"]}))
                proposal = self._public_proposal(self.call({"op": "vla_propose", "subgoal": arguments["subgoal"]}))
                return proposal
            return self._act(arguments)

    def _act(self, args: dict) -> dict:
        if "review" in args and not any(key in args for key in ("proposal_id", "targets", "steps")):
            latest = self._observe()
            if not latest.get("observation", {}).get("review_due"):
                raise ToolInputError("A review-only act requires a due checkpoint")
            return self._public(self._review(latest, args["review"]))
        targets = args.get("targets")
        if targets is not None and "proposal_id" in args:
            raise ToolInputError("Choose proposal_id with steps, or targets without proposal_id")
        steps = args.get("steps", len(targets) if targets is not None else 5)
        warnings = []
        if targets is None and "max_native_steps" in args:
            if "proposal_id" not in args:
                raise ToolInputError("max_native_steps applies only to model TCP targets")
            if args["max_native_steps"] < steps:
                raise ToolInputError("max_native_steps is smaller than the requested VLA steps; "
                                     "reduce steps or omit max_native_steps")
            warnings.append(dict(
                code="parameter_ignored", parameter="max_native_steps",
                supplied_value=args["max_native_steps"], requested_steps=steps,
                message="VLA acceptance uses steps; redundant max_native_steps was ignored."))
        if targets is not None and len(targets) != steps:
            raise ToolInputError("steps must match the replacement target count")
        # Convert only caller-supplied TCP targets. An accepted proposal is
        # already stored natively in link6 coordinates and is never shifted.
        native_targets = None
        if targets is not None:
            try:
                native_targets = [{arm: tcp_to_link6_target(dual[arm])
                                   for arm in ("left", "right")} for dual in targets]
            except ValueError as error:
                raise ToolInputError(str(error)) from error
        if self.setting in ("pi05_vla", "simate_vla"):
            if targets is None and "proposal_id" not in args:
                raise ToolInputError("Provide proposal_id with steps, or model-chosen TCP targets")
        elif not self.direct_act_enabled or targets is None or steps != 1 or "proposal_id" in args:
            raise ToolInputError("Direct EEF requires permission and exactly one dual-arm target")
        latest = self._observe()
        latest = self._review(latest, args.get("review"))
        if latest.get("done"):
            return self._public(latest)
        if native_targets is not None and latest.get("manual_tcp_targets_enabled"):
            if not latest.get("manual_tcp_targets_enabled"):
                raise ToolInputError("This episode bridge has not enabled direct TCP fallback")
            request = dict(op="act_eef", targets=native_targets,
                           reason=args["reason"], assessment=args["assessment"])
            if "max_native_steps" in args:
                request["max_native_steps"] = args["max_native_steps"]
            result = self.call(request)
        elif self.setting in ("pi05_vla", "simate_vla") and native_targets is not None:
            raise ToolInputError("This episode bridge has not enabled direct TCP fallback")
        elif self.setting in ("pi05_vla", "simate_vla"):
            supplied_id = args["proposal_id"]
            # The native host still rejects stale/consumed UUIDs. Preserve full
            # native IDs for pre-change conversations without alias guessing.
            native_id = next((key for key, value in self._proposal_ids.items()
                              if value == supplied_id), supplied_id)
            result = self.call(dict(op="vla_review", proposal_id=native_id,
                                    decision="accept", targets=None, approved_steps=steps,
                                    reason=args["reason"], assessment=args["assessment"]))
        else:
            result = self.call(dict(op="act", repeat=1, action=dict(
                mode="eef", target=native_targets[0], reason=args["reason"], assessment=args["assessment"])))
        if warnings:
            result = {**result, "warnings": [*result.get("warnings", []), *warnings]}
        return self._public(result)

    def execute_tool(self, name: str, arguments: dict) -> Any:
        """Supply RoboRSI ToolResult objects to its existing MCP transport."""
        from roborsi.tools import ToolResult

        try:
            value = self.dispatch(name, arguments)
            if name != SPATIAL_TOOL_NAME:
                value = _round_display(value)
        except ToolInputError as error:
            return ToolResult(name=name, result={"error": str(error)})
        result = ToolResult(name=name, result=value)
        cameras = value.get("observation", {}).get("cameras", {})
        if self.image_root is not None:
            for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
                ref = cameras.get(camera)
                if not isinstance(ref, str) or not ref.startswith("/observations/"):
                    continue
                path = (self.image_root / ref.removeprefix("/observations/")).resolve(strict=True)
                media_type = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(path.suffix.lower())
                if not path.is_relative_to(self.image_root) or media_type is None:
                    raise RuntimeError("Invalid public RGB image path")
                result.content_blocks.extend([
                    {"type": "text", "text": camera + " RGB"},
                    {"type": "image", "source": {
                        "type": "base64", "media_type": media_type,
                        "data": base64.b64encode(path.read_bytes()).decode(),
                    }},
                ])
        return result

    def as_local_tools(self) -> tuple:
        """Register in EmbodiedAgent using the repository's LocalToolSpec."""
        from roborsi.embodied_agent import LocalToolSpec

        return tuple(LocalToolSpec(
            name=spec["name"], description=spec["description"],
            input_schema=spec["input_schema"],
            handler=lambda args, name=spec["name"]: self.execute_tool(name, args),
            propagate_errors=True,
        ) for spec in self.get_tools_spec())

    def mcp_server(self) -> Any:
        """Create the existing RoboRSI HTTP MCP transport; caller owns its lifetime."""
        from roborsi.planner.utils.http_mcp_server import HttpMcpServer

        return HttpMcpServer(self)
