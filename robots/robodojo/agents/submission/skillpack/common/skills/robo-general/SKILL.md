---
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
