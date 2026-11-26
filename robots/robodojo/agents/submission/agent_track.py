"""Task-agnostic Agent Track inputs and one immutable general skill library.

Task/seed/layout remain private scheduler metadata. They never select a skill,
action program or prompt. The official runtime description is the sole task
text, and every episode owns a fresh workspace.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re


MODE = "general_only"
CORE = """You are an agent carrying out the supplied embodied task. Follow the current episode instruction, public interface, measured state and RGB observations. Continue authorized work autonomously; never ask the user to approve robot actions. Read the same robo-general and manipulation skills used for every episode before acting.

Do not stop the task merely because it looks complete while native done remains false.
Keep useful state within this episode only and end your response only after native done; exiting earlier is recorded as incomplete.

Prefer calibrated spatial localization and code-computed EEF/TCP motion when geometry is usable. Use robot.spatial_localize_pixel with original-image pixels, observation_id/image_id from observe, and an independently known plane or absolute height. Inspect its environment_origin coordinates, uncertainty and assumptions. The isolated tool uses deterministic geometry and no grounding model. Never guess hidden height or annotate crop coordinates without reversing the transform. Re-observe and localize again after object motion or uncertain execution. No spatial memory is maintained. If geometry remains unavailable, use visually grounded VLA assistance when permitted. Localization never executes motion and is not a prerequisite for VLA.

Use this feedback loop: observe, choose a meaningful action, execute, inspect actual RGB and measured-state feedback, then correct or switch control mode. Direct EEF/TCP targets are supported for approach, descent, alignment, grasp closure, lifting, transport, insertion, release and retreat. Start from hold_current, preserve the inactive arm, and choose orientation, clearance and object-to-TCP offset from geometry. Use move_relative for environment-axis corrections and close_in_place for gripper-only changes, then pass computed targets to act. Helpers only calculate; act executes. Public EEF targets refer to TCP/GRASP_POINT, not raw link6. The host handles IK and interpolation. A point is not a grasp pose, and pose arrival or closed fingers do not prove possession or completion.

After stalled_reobserve, native_step_budget_exhausted, an interruption or a large residual, inspect the actual measured pose before another action. Revise the TCP target or use VLA to recover from the current configuration when its preview is suitable; do not blindly repeat an unreachable target. When localization is unavailable, VLA may also perform a visually specified approach or manipulation. After VLA, re-observe and return to direct TCP refinement when the next correction is clear. Near-target proximity never requires VLA, and VLA requires no named memory point, metric distance threshold or extra localization fields. Native permissions, proposal freshness, checkpoint reviews and motion budgets still apply.

Call vla_propose with a short English instruction naming the visible object and selected action or destination. Follow the supplied subgoal wording rules, using separate short sentences and necessary conditions. Prefer the official description and current observed object wording and keep the instruction stable throughout the same subgoal. Do not include XYZ, metric distances, internal memory IDs, reasoning or unrelated future plans. Inspect its preview, both arms, gripper changes, bounds and current RGB before accepting. Choose a native prefix of 5..50 steps according to stage and uncertainty: shorter near uncertain contact, longer only when the inspected motion is appropriate. Stop before an unverified release when necessary. There is no universal 10-step cap. Inspect each execution result before requesting a fresh proposal or computing another TCP correction.

Do not terminate an unfinished episode merely because localization failed, a TCP target stalled or a capability is assumed unavailable. A statement that a tool is unsupported must cite an actual attempted tool call, its arguments and the exact returned error. Successful observe does not establish that act failed; an unattempted tool has no observed failure. A guidance status, malformed annotation or geometric rejection is not an unsupported robot interface. Correct input errors and use an available alternative control path. After timeouts or uncertain execution, observe before retrying to avoid duplicate motion. End normally only after native done; apparent completion or a per-act tracking budget does not end the episode. If a genuine blocker leaves no permitted way forward, report the failed call and returned error, actual progress, remaining work and the alternative attempted or why it was unavailable; distinguish observations from assumptions and never invent tool failures.

Use the current official Detail Description as the sole task specification, including its public scoring table, score values and intermediate completion conditions. Use these published criteria with current observations; they do not expose a live evaluator score or authorize reading checker internals. Keep memory inside this episode. Do not access other episodes, task-specific recipes, demonstrations, evaluator internals, hidden state or saved answers.
"""
GENERAL = """---
name: robo-general
description: Operate RoboDojo with observe, vla_propose and act. Use for VLA-assisted or enabled direct TCP control.
---

# Robot operation

Follow the current episode instruction, permissions and budgets. Read the official Detail Description for this episode's objective and acceptance criteria. Use declared tool parameters; keep reasons and observed-state assessments concise.

Unless needed to diagnose a concrete issue, do not read the implementation source of existing skills or scripts; use their SKILL.md instructions and documented interfaces.

Call robot tools directly. Their results include state text and RGB image blocks automatically; no JavaScript forwarding, `text()` or `image()` is needed.

## Observe, propose and execute

1. Call `observe` first. Inspect head and both wrist RGB, measured TCP state and remaining budget. Observe again after a timeout, uncertain execution or session recovery.
Use `robot.spatial_localize_pixel` with current observation/image IDs, original-image pixels and an independently known plane or absolute height. Inspect coordinates and uncertainty before planning TCP motion. Re-localize after object motion; no spatial memory is maintained.
3. For direct refinement, compute and execute a full dual-arm TCP target, then observe; no VLA preview is required. If using a VLA proposal, read each preview row with current RGB. Both arms show `pos_m` (TCP XYZ metres), `angles_deg` ([pitch,roll,yaw] degrees), `grip` (opening 0..1), `delta_cm` (XYZ from measured start), `path_cm`, maximum consecutive-step changes (`max_step_cm`, `max_rotation_step_deg`, `max_joint_step_rad`), `grip_delta` and `grip_range` (including the measured start). Row `bounds` are cumulative existing warnings through that step. Prefix statistics include every original step, including unsampled excursions; all displayed floats have at most three decimals. If `diagnostics=unavailable`, inspect the poses and RGB without assuming zero changes. No full 50-point file or inspection command is exposed. Choose one action:
   - Accept: copy the current short `proposal_id` (e.g. `p1`) into `act` with a consecutive `steps` count (5..50, default 5; choose by stage, preview and uncertainty). Old IDs expire when replaced or consumed. The harness executes the stored original joint sequence with its native timing; FK is for preview only.
   - Choose your own motion: give `act` full dual-arm TCP `targets`. No proposal_id or reject flag is needed; these targets replace the VLA suggestion. The harness tracks each target using measured-state feedback and IK; compute only meaningful approach waypoints; let the controller generate low-level interpolation.
   - To calculate a target from the latest measured state, call `hold_current({})`, `close_in_place({"arm":"left","opening":0})`, or `move_relative({"arm":"left","delta_m":[0,0,0.03]})`. They preserve both orientations and unmodified arm/gripper fields. Relative offsets use environment XYZ metres. All three are read-only: they return `control_step`, `computed_only=true` and one full dual-arm `targets` list, without motion, VLA inference or proposal replacement. Inspect the result, then pass its `targets` to `act` with `reason` and `assessment.current_subgoal`. Recalculate after any intervening action; a returned target does not execute itself.
   - TCP tracking uses a shared `max_native_steps` budget (default 80, maximum 160), with 3 mm / 3 degree pose tolerance. Target count is not native-step count. Check reached/stalled/budget status, residuals and actual executed steps; pose arrival does not prove grasp.
4. Inspect the returned RGB and measured state after execution. Use short spans near contact, placement or uncertainty. If interrupted, inspect the current result before choosing another action; an interrupted VLA proposal cannot be reused.

Only if observe reports `review_due`, use the legacy checkpoint fields declared by `act.review` before continuing. This is harness bookkeeping, not a native scoring condition.

Use RGB, measured state and numerical evidence to choose the needed review depth:

- Target identity: confirm the instructed object on first selection, after occlusion or when the scene differs from expectation. Change view or crop when useful.
- Arm and subgoal: compare both arms' motion and gripper changes with the intended object and action. Briefly explain consequential changes in the next action reason/current_subgoal.
- Grasp: after closing or a short test lift, before transport, or when slipping/empty fingers are suspected, check whether the intended object is held and moves with the gripper.
- Path: inspect visible clearance along the path and at the endpoint, especially near surfaces, clutter or the other arm. Use a shorter span, re-observe or revise the motion when needed.
- Completion: compare actual arrangement, support, release and arm state with the official Detail Description's acceptance criteria.

Keep consequential findings and the next action concise in `act.reason` and `assessment.current_subgoal`.

Preview samples are for inspection; do not interpolate them into an execution trajectory. All poses/targets use gripping-face-centre TCP XYZ in metres and `pitch_deg`, `roll_deg`, `yaw_deg` relative to the vertical-down reference. Zero angles point down; yaw=30 means rotate that reference 30 degrees about +Z. The fixed-axis order is pitch about +X, roll about +Y, then yaw about +Z. Use TCP targets directly. TCP is not necessarily the held object centre; judge grasp offset and progress from current RGB. A command or closed gripper does not prove possession or completion.

Targets use continuous `gripper_opening` from 0 (fully closed) to 1 (fully open). Intermediate values are allowed. Proposal samples show the stored continuous openings; accepting a proposal preserves them. Measured openings are in `left_ee_joint_state` and `right_ee_joint_state`.

When VLA is disabled and direct control is enabled, use `observe` then `act` with one dual-arm TCP target and no proposal_id. Hold an inactive arm at its measured pose and intended opening.

## Code motion, local refinement and VLA recovery

Prefer code-computed full dual-arm TCP targets from measured state and usable geometry. This applies to free-space approach, descent, alignment, grasp closure, lifting, transport, insertion, release and retreat. Start from hold_current, preserve the inactive arm and select orientation, object-to-TCP offset and clearance explicitly. An object point is not a grasp pose. Execute meaningful stages and inspect returned RGB and measured state after each stage. The host performs IK and interpolation; generated code alone moves nothing.

Use move_relative for environment XYZ corrections, hold_current to preserve measured poses while selecting a different orientation, and close_in_place for gripper-only correction. Pass the calculated targets to act with reason and assessment.current_subgoal. Public EEF targets use TCP/GRASP_POINT, not raw link6. Direct act needs no proposal_id or localization metadata.

After stalled_reobserve, native_step_budget_exhausted, interruption or excessive residual, inspect current state. Revise the TCP target or request a visually grounded VLA recovery proposal; do not repeat the same unreachable pose blindly. Localization failure is also a valid reason to use VLA when enabled. VLA does not require a named memory point or a metric proximity threshold. After a VLA prefix, inspect the new state and return to direct TCP refinement when practical. Either mode may handle contact; proximity never mandates VLA. Preserve native permissions, current-proposal checks, checkpoint reviews and budgets.

Keep each VLA instruction short and stable within the same subgoal, naming visible object attributes and an action/destination. Numeric pose corrections and internal memory IDs belong in code, not the VLA instruction. Choose 5..50 native steps from the inspected preview and current stage; use short prefixes for uncertain contact and longer prefixes only for appropriate inspected motion. Stop before an unverified release when needed. There is no fixed 10-step cap. Mark moved-object memory stale and refresh it before metric reuse, without preventing visually grounded VLA assistance.

## Evidence before declaring a blocker

Continue unfinished work after recoverable input, localization or tracking errors. An unsupported-tool claim requires an actual attempted tool call, its arguments and the exact returned error. Do not infer act failure from observe, a helper error or an unattempted call. Correct malformed input; distinguish geometric rejection and tracking limits from interface failure. Use an available alternative control path, and observe after an uncertain execution before retrying. Do not debug source code in the task context.

End normally only after native done; apparent completion or a per-act tracking budget does not end the episode. If a genuine blocker prevents all permitted progress, report the failed call and exact returned error, observed progress, remaining work and the alternative attempted or why it was unavailable. Successful observe means observation worked. Never report an invented unsupported call or claim a grasp, insertion, release or task completion solely because a target pose was reached.

## Completion

Do not stop the task merely because it looks complete while native done remains false.
Keep useful state within this episode only and end your response only after native done; exiting earlier is recorded as incomplete.

Use the official Detail Description, including all supplied public scoring rows and intermediate conditions, to assess applicable checks as met / unmet / uncertain, with current RGB, measured state and native-step evidence. Use only thresholds and terminal conditions stated in that official description; do not consult checker code or hidden evaluator state. Check actual position/orientation, coverage, support, release, stability after real native steps and applicable arm state. A command, planned target, prior expectation or occluded view does not establish completion; repeated observe alone advances no physics. Continue useful inspection/corrections while requirements remain unmet or uncertain and the task permits it. Native done means termination; success and score come from the native evaluator. The harness owns enforced limits and termination.


## Structured replies and uncertain actions

Robot replies may be a public JSON object, an MCP `content` array containing JSON text and RGB blocks, or a JSON string encoding that envelope. Use `scripts/robot_reply.py:parse_reply` on the saved reply; it preserves `json_values`, text and `RGB_blocks`. Read status, done, actual executed/completed targets and the observation from the decoded JSON value, preserving zero and false. Do not regex `JSON.stringify(reply)` or assume a top-level observation: JSON inside `content[].text` is escaped. A camera label is not JSON. Prefer direct robot calls, which render state and RGB automatically.

A JavaScript parsing or display error after `robot.act` does not prove that motion failed. Inspect a fresh public observation and the native action ACK/control step before deciding on another action; never replay the motion solely because the surrounding script reported an error. `native_step_budget_exhausted` is a per-act tracking limit, not necessarily episode termination. `stalled_reobserve` does not establish contact, grasp, a pressed control or task success. An interrupt can cancel the unexecuted suffix: read the actual completed-target count before planning from the resulting pose.

## Read-only code helpers

The helpers in this skill execute no robot actions and read no task answers. Use their documented interfaces instead of writing another crop/parser script. Check the exit code and returned status before reading an output path; a failed helper or missing file is not a valid localization. Current localization is stateless: use `robot.spatial_localize_pixel`; do not invent memory filenames or reuse legacy object/point keys.

- `python3 .agents/skills/robo-general/scripts/robot_reply.py reply.json` decodes a saved public reply. `require_points(points, minimum=2, dimensions=2)` rejects empty, short or nonfinite detections before indexing, fitting or min/max reduction. A rejected detection needs a new observation/view or an available alternative; never turn NaN or an empty centroid into a target. `localization_point(result, allow_estimated=False)` accepts only a successful finite metre-valued environment-origin point; allow an estimated result only after inspecting its uncertainty/assumptions. `image_pixel([u,v], width, height)` validates exclusive original-image upper bounds before indexing or localization.
- `python3 .agents/skills/robo-general/scripts/robot_reply.py --angles 80 -100 0` returns an equivalent valid angle representation. Pitch/yaw must be within ±180 degrees and roll within ±90; these are representation ranges, not physical tilt limits. Normalize all three angles together, then inspect the intended rotation and clearance. Do not clamp roll alone: that changes the desired orientation. The helper does not establish reachability or safety.
- `python3 .agents/skills/robo-general/scripts/rgb_inspect.py --input /observations/current_cam_head.jpg --output inspect/head.png --crop 20 40 220 240 --scale 1.5 --grid` crops an ordinary current RGB image. Scale accepts positive finite fractions; `--grid` is a separate boolean flag. Crop edges are integer original-image coordinates and must be within bounds. The returned `pixel_to_original` mapping converts displayed pixel centres back to original pixels: `original = displayed * scale + offset`, per axis. Pass original-image coordinates to localization, with `0 <= u < width` and `0 <= v < height`. Do not confuse crop/resize coordinates or normalized 0..1 coordinates with original pixels. The input image is never overwritten.

When the public episode rules require stationary waiting for another actor or process, hold both measured arm poses and intended gripper openings. Staying outside a shared region alone does not establish compliance with a stillness requirement. Do not pregrasp the next object or jiggle to spend steps. `hold_current` only calculates: pass its unchanged targets to `act` for native time to advance, use short inspected spans, and resume manipulation only after current temporal observations establish that the wait condition has ended. Do not infer that condition from an object count or one still image.

## Current camera calibration

The current observation's `spatial_localization.calibration_file` points to an
existing read-only JSON under `/observations` for this episode. It contains each
camera's full-precision `K`, `T_base_camera_cv`, `distortion` and image binding.
`T_base_camera_cv` maps optical-camera coordinates into `environment_origin`;
positions use metres. The public snapshot also supplies image dimensions and
current observation/image IDs. Use the calibration belonging to the current
image and control step; reobserve after camera motion. These camera parameters
do not provide hidden object poses, depth or task answers.
"""
MANIPULATION = """---
name: manipulation
description: Reusable observation-conditioned grasp, placement, support, contact and recovery principles.
---

# Observation-conditioned manipulation

Use current object geometry, measured poses and both-arm gripper intentions.
Choose subgoals and required endings from the current official Detail
Description. No task identity selects a workflow in this skill.

**Grasp and retention.** Identify accessible opposing grasp surfaces and a free
withdrawal direction. Choose a reachable approach without crossing occupied
space. Select orientation, opening and object-to-TCP offset from current RGB.
Inspect whether the intended surface is between the jaws before closing at the
measured pose. A closed command or shortened opening alone is insufficient.
When permitted, a small clearance lift can test whether object and gripper move
together. If the object stays, slides or tips, reobserve its new pose and change
the alignment or grasp hypothesis before another attempt. Transport requires
observed retention and clearance.

**Support and release.** Distinguish the held-object geometry from the TCP.
Align the intended support surfaces and account for the grasp offset and
required orientation. Inspect contact before release. An open command does not
prove release: confirm that the object separates from the jaws, remains
supported and is not dragged during withdrawal. When needed, inspect stability
after real native steps; repeated observe alone advances no physics. Whether
the final object must stay held, be released or return elsewhere is determined
by the official description, not this skill.

**Two-arm support.** Select a recipient grasp on an exposed surface rather than
the donor jaws or functional tip. Keep at least one verified support until
recipient retention is established. If support is uncertain, stabilize with
the supporting arm or use a visible stable staging surface when permitted.
Never open both grippers to resolve uncertainty. Balanced support may be useful
for a long or flexible object; inspect both arms' clearance and load effects.

**Contact, insertion and rotation.** Use visible mating geometry to choose
alignment and approach direction. Near uncertain contact, use small
feedback-controlled corrections. Unexpected lateral motion, tilt or residual
error can indicate an edge collision: withdraw or realign from observation
instead of forcing an unseen path. A rotation about the TCP can move the held
object's functional end along an arc because of grasp offset. Verify relative
motion and engagement; rotating an entire assembly does not establish that two
parts moved relative to one another. Regrasp only with adequate support.

**Articulation and deformable material.** Distinguish motion about a visible
hinge from translation of the whole body; unexpected sliding calls for
reobservation and a revised support hypothesis. For deformable material,
visible local displacement does not guarantee that the rest followed. Inspect
the resulting shape, support and exposed edges after an action; rigid-body pose
alone cannot establish its configuration.

**Flow, containment and occlusion.** Keep a receiver accessible and supported;
passing above an opening does not prove entry. Check visible transfer and any
external spill, and allow dynamic material to settle when needed. Hidden
contents require entry history rather than an empty source alone. If later
motion occludes a relevant region, retain episode-local visual evidence while
continuing to check what remains visible. Choose transfer order and final
container state only from the official description and current observations.

**Recovery.** After slip, collision, occlusion, uncertain execution or loss of
support, inspect RGB, measured state and native ACK before choosing the next
action. Replan from the actual configuration. Do not repeat a motion solely
because a parser failed, or infer grasp/task success from reached pose alone.
"""
INSTRUCTIONS = """Read each registered skill's SKILL.md. The library is identical for all tasks.
Available skills:
- robo-general: Public observation, tools and feedback control. (file: /workspace/.agents/skills/robo-general/SKILL.md)
- manipulation: Observation-conditioned manipulation. (file: /workspace/.agents/skills/manipulation/SKILL.md)
"""


def validate_mode(value):
    if value != MODE:
        raise ValueError("skill_library_mode must be general_only")
    return value


def _downloaded_detail(task, bundle, manifest_sha256):
    root=Path(bundle).resolve(strict=True)
    raw=(root/"MANIFEST.json").read_bytes()
    if hashlib.sha256(raw).hexdigest()!=manifest_sha256:
        raise ValueError("Official description manifest changed")
    manifest=json.loads(raw)
    canonical = manifest.get("origin") == "https://robodojo-benchmark.com/leaderboard/detail-description"
    if not canonical and manifest.get("origin")!="https://robodojo-benchmark.com/doc/sim-tasks/":
        raise ValueError("Description snapshot is not the official task source")
    record=manifest["tasks"][task]
    if record.get("task")!=task or record.get("section")!=("Detail Description" if canonical else "Detailed Instruction"):
        raise ValueError("Official description identity differs")
    path=(root/record["detail_file"]).resolve(strict=True)
    if path.parent!=root:
        raise ValueError("Official description file escapes the frozen bundle")
    payload=path.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=record["detail_sha256"]:
        raise ValueError("Official description text changed")
    if canonical:
        description, scoring = record.get("description"), record.get("scoring")
        if (manifest.get("schema_version") != 2 or record.get("url") != manifest["origin"]
                or record.get("includes_public_scoring") is not True
                or not isinstance(description, str) or not description.strip()
                or not isinstance(scoring, list)
                or any(not isinstance(row, str) or not row.strip() for row in scoring)):
            raise ValueError("Canonical official description/scoring schema differs")
        expected = description
        if scoring:
            expected += "\n\nScoring\n" + "\n".join("- " + row for row in scoring)
        if payload.decode("utf-8") != expected + "\n":
            raise ValueError("Canonical official description contains unregistered additions")
    return payload.decode("utf-8"),record


def validate_resume_mode(mode, source_config):
    validate_mode(mode)
    if source_config is not None and source_config.get("options",{}).get("skill_library_mode",MODE)!=mode:
        raise ValueError("An episode cannot change its skill library while resuming")


def official_observation(obs, *, task=None, bundle=None, manifest_sha256=None):
    """Bind native episode language to the frozen official task specification."""
    extra=obs.get("extra_info")
    detail=extra.get("detailed_instruction") if isinstance(extra,dict) else None
    canonical=record=None
    if bundle is not None:
        if not isinstance(manifest_sha256,str) or len(manifest_sha256)!=64:
            raise ValueError("A downloaded official description requires a frozen manifest SHA")
        canonical,record=_downloaded_detail(task,bundle,manifest_sha256)
    native=isinstance(detail,str) and bool(detail.strip())
    runtime_detail = detail if native else None
    appended_scoring = False
    runtime_instruction = runtime_instruction_field = None
    if record is not None and record.get("requires_official_runtime_parameters"):
        canonical_source = record.get("url") == "https://robodojo-benchmark.com/leaderboard/detail-description"
        # The native environment has already expanded <target> and generated
        # the current color order/category mapping/bottle-and-bowl sequence.
        # Bind that observed language, never infer it from task/seed/layout.
        template_only = not native or " ".join(detail.split()) in {
            " ".join(record.get("description", "").split()), " ".join(canonical.split())}
        if canonical_source and template_only:
            runtime_instruction_field = "instruction" if obs.get("instruction") else "instructions"
            runtime_instruction = obs.get(runtime_instruction_field)
            if isinstance(runtime_instruction, (list, tuple)) and runtime_instruction and all(
                    isinstance(value, str) and value.strip() for value in runtime_instruction):
                runtime_instruction = "\n".join(runtime_instruction)
            if not isinstance(runtime_instruction, str) or not runtime_instruction.strip():
                raise ValueError("Official language template requires the current runtime instruction")
            if (re.search(r"<[A-Za-z_][A-Za-z0-9_-]*>", runtime_instruction)
                    or re.search(r"current language instruction", runtime_instruction, re.I)
                    or " ".join(runtime_instruction.split()) == " ".join(canonical.split())):
                raise ValueError("Official instruction contains unresolved runtime parameters")
    if not native:
        if canonical is None:
            raise ValueError("general_only requires official obs.extra_info.detailed_instruction")
        if record.get("requires_official_runtime_parameters") and runtime_instruction is None:
            raise ValueError("Official document template requires the complete runtime detailed_instruction")
        detail=canonical
    if not detail.strip():raise ValueError("Official description is empty")
    if "<target>" in detail:
        raise ValueError("Official detailed_instruction contains unresolved runtime parameters")
    if native and record is not None and record.get("includes_public_scoring") and record["scoring"]:
        normalize = lambda text: " ".join(text.replace("**", "").split())
        missing = [row for row in record["scoring"] if normalize(row) not in normalize(detail)]
        if missing:
            if re.search(r"(?im)^\s*(?:#{1,6}\s*)?Scoring\s*:?(?:\s|$)", detail):
                raise ValueError("Runtime scoring differs from the frozen official specification")
            detail += "\n\nScoring\n" + "\n".join("- " + row for row in missing) + "\n"
            appended_scoring = True
    if runtime_instruction is not None:
        detail = "Current language instruction\n" + runtime_instruction + "\n\nDetail Description\n" + detail
    provenance={"kind":("official_runtime_field_with_public_scoring" if appended_scoring else
                         "official_runtime_field" if native else "official_document_snapshot"),
                "description_sha256":hashlib.sha256(detail.encode()).hexdigest(),
                "legacy_short_instruction_used":False}
    if runtime_instruction is not None:
        provenance.update(kind=("official_runtime_template_with_current_instruction" if native else
                                "official_document_snapshot_with_current_instruction"),
                          runtime_instruction_field=runtime_instruction_field,
                          runtime_instruction_sha256=hashlib.sha256(runtime_instruction.encode()).hexdigest())
    if record is not None:
        expected_runtime = record.get("description", canonical)
        provenance.update(source_url=record["url"],manifest_sha256=manifest_sha256,
                          canonical_sha256=record["detail_sha256"],
                          runtime_matches_snapshot=(" ".join(runtime_detail.split())==" ".join(expected_runtime.split()) if native else None),
                          public_scoring_included=bool(record.get("scoring")),
                          public_scoring_appended=appended_scoring)
        if native:
            provenance["runtime_description_sha256"] = hashlib.sha256(runtime_detail.encode()).hexdigest()
    value=dict(obs, instruction=detail, extra_info={"detailed_instruction":detail},
               _roborsi_official_description_source=provenance)
    value.pop("instructions",None)
    return value


def expose_current_calibration(public, raw, *, observations_root):
    """Advertise only the existing current-episode read-only calibration file."""
    from pathlib import PurePosixPath
    observation=raw.get("observation",{})
    spatial=observation.get("spatial_localization")
    if not spatial or "case_path" not in spatial:return public
    step=observation.get("control_step")
    path=PurePosixPath(spatial["case_path"])
    if (not path.is_absolute() or path.parts[:2]!=("/","observations")
            or len(path.parts)!=4 or any(part in (".","..") for part in path.parts)
            or not path.name.endswith("_spatial_case.json")
            or type(step) is not int or spatial.get("control_step")!=step):
        raise ValueError("Camera calibration is not bound to the current observation")
    root=Path(observations_root).resolve(strict=True)
    source=(root/Path(*path.parts[2:])).resolve(strict=True)
    if not source.is_relative_to(root):
        raise ValueError("Camera calibration escapes the current episode")
    metadata=json.loads(source.with_suffix(".meta.json").read_text())
    if (metadata.get("control_step")!=step or metadata.get("case_sha256")!=_sha(source)
            or metadata.get("observation_id")!=path.parent.name+":"+path.stem):
        raise ValueError("Camera calibration snapshot identity differs")
    public["observation"]["spatial_localization"]["calibration_file"]=str(path)
    return public


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_profile(policy_dir, task, owner, *, vla_backend):
    from .execution_profile import _hash, _tool_availability, runtime_modules
    root=Path(policy_dir).resolve()
    index=json.loads((root/"skillpack/task_catalog.json").read_text())
    if task not in index["tasks"]:
        raise ValueError("Unknown public RoboDojo task")
    runtime=runtime_modules(root)
    destination=Path(owner).resolve(strict=True)/"execution-profile"

    class GeneralProfile(runtime.prompt_profile.RobotPromptProfile):
        def episode_prompt(self,cfg):
            data={"vla_backend":self.vla_backend,"reset_allowed":False,
                  "direct_act_enabled":True,
                  "native_step_limit":cfg["options"]["review_window_steps"]*cfg["options"]["review_max_windows"]}
            if cfg.get("episode_seconds") is not None:
                data["wall_seconds_budget"]=cfg["episode_seconds"]
            return ("Carry out the official Detail Description supplied by observe.\n"
                    "Episode configuration:\n"+json.dumps(data,separators=(",",":"))+"\n")

        def stage_live_workspace(self,workspace,*,owner):
            receipt=super().stage_live_workspace(workspace,owner=owner)
            # GeneralProfile owns small generated files, not external case
            # references. Keep the workspace materialized for Hybrid's boundary
            # check instead of creating unregistered links into this profile.
            for relative, digest in receipt["assets"].items():
                path=Path(workspace)/relative
                if path.is_symlink():
                    payload=path.read_bytes()
                    if hashlib.sha256(payload).hexdigest()!=digest:
                        raise ValueError("General skill changed while staging")
                    path.unlink()
                    path.write_bytes(payload)
            public=Path(workspace).parent/"tools/BENCHMARK.md"
            if public.exists():public.write_text("# Public robot interface\n\nRead the registered general skills.\n")
            allowed={"robo-general","manipulation"}
            skills=Path(workspace)/".agents/skills"
            if {p.name for p in skills.iterdir()}!=allowed:
                raise ValueError("general_only workspace contains an unregistered skill")
            if (Path(workspace)/"icl_reference").exists():
                raise ValueError("general_only cannot expose public ICL demonstrations")
            return receipt

    if destination.exists():
        profile=GeneralProfile(destination)
        if (profile.task!=task or profile.vla_backend!=vla_backend
                or profile.manifest.get("skill_library_mode")!=MODE):
            raise ValueError("Owned profile belongs to another task, backend or skill mode")
        return _tool_availability(profile,vla_backend!="disabled")
    destination.mkdir()
    source=root/"skillpack/common"
    files={"agent_core.md":CORE,"skills_instruction.md":INSTRUCTIONS,
           "skills/robo-general/SKILL.md":GENERAL,
           "skills/manipulation/SKILL.md":MANIPULATION}
    for name in ["native_tools.json","robot_tools.json","actual_robot_declarations.json",
                 "skills/robo-general/scripts/robot_reply.py",
                 "skills/robo-general/scripts/rgb_inspect.py"]:
        files[name]=(source/name).read_text()
    from .runtime_config import PI05_BASE_PROMPT_SHA256, load_pi05_base_prompt, resolve_vla_config
    if vla_backend=="pi05":files["pi05_base_prompt.txt"]=load_pi05_base_prompt(root)
    for name,value in files.items():
        p=destination/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value)
    manifest=deepcopy(json.loads((source/"profile.json").read_text()))
    for key in ["readonly_source_root","readonly_references"]:manifest.pop(key,None)
    manifest.update(task=task,profile_id="robodojo_general_only_v1:"+vla_backend,
                    skill_library_mode=MODE,skills=["robo-general","manipulation"],
                    single_markdown_skills=["robo-general","manipulation"],
                    vla_backend=vla_backend,vla_setting=resolve_vla_config({"vla_backend":vla_backend})["setting"],
                    base_prompt_sha256=PI05_BASE_PROMPT_SHA256 if vla_backend=="pi05" else None,
                    sha256={name:_hash(destination/name) for name in files})
    (destination/"profile.json").write_text(json.dumps(manifest,indent=2)+"\n")
    profile=GeneralProfile(destination)
    profile.task_selection=index["tasks"][task]
    return _tool_availability(profile,vla_backend!="disabled")


def publish_initialization_ready(obs):
    """Private scheduler event after a verified native observation; never an ACK."""
    target=Path(os.environ["ROBORSI_INITIALIZATION_READY_FILE"])
    folder=Path(os.environ["ROBORSI_TRAJECTORY_DIR"]).parent.resolve(strict=True)
    if target.name!="initialization-ready.json" or target.parent.resolve(strict=True)!=folder:
        raise ValueError("Initialization event escapes the current attempt")
    if target.exists():return
    native=obs.get("roborsi_native_identity")
    token=os.environ.get("ROBORSI_COLLECTION_LAUNCH_TOKEN")
    if not isinstance(native,dict) or not token:
        raise ValueError("Initialization event lacks verified native identity or launch token")
    value={"launch_token":token,"native_identity_verified":True,"control_step":native["control_step"]}
    temporary=target.with_name(".initialization-ready.new")
    with temporary.open("x")as stream:json.dump(value,stream);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,target)
