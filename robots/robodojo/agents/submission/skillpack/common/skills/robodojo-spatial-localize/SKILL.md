---
name: robodojo-spatial-localize
description: Localize named object points from calibrated current RGB and update task spatial memory. Use a declared support plane for known-height points or matching views for unknown height.
---

# Localize a point and update spatial memory

1. Call `robot.observe`. Keep its `observation.control_step` with the displayed RGB. The host already paired RGB with full-precision calibration in `environment_origin` metres, including wrist motion. If calibration is absent, report that capability once and keep metric positions unknown.
2. Choose the object and physical point in current RGB. Pixels are `u,v = column,row` on the **original** image, with dimensions in `spatial_localization.image_sizes` (or the original image file). Crop/preview pixels need their explicit inverse crop/resize transform before use.
3. Call `robot.localize` with that `control_step`, the original-image pixels, and an explicit geometry choice. It creates the annotation file, validates the current observation, runs geometry and updates `/workspace/space.json`. Inspect each point's `accepted`, `reason` and `position_m` before planning motion. The packaged CLI is equally supported when you prefer code; follow inputs.md for its annotation schema.

## Structured localization for one or more points

For an unknown-height point visible in two cameras, use `kind: "direct"` and the **same physical point** in both RGB images:

```json
{"control_step": 0, "targets": [{
  "object_id": "red_block", "point": "top_center",
  "description": "visible center of the red block top face", "kind": "direct",
  "observations": [
    {"camera": "cam_head", "pixel": [320, 240]},
    {"camera": "cam_left_wrist", "pixel": [280, 220]}
  ]
}]}
```

Replace the step and pixels with current evidence. For a point at an **independently known** height above a horizontal plane, set `kind: "support_plane"`, pass `plane_z_m` at the top level and `height_above_plane_m` on each plane target. Both are environment-origin metres; height `0` is valid only for actual contact. The default RoboDojo tabletop is `z=0.765`, but conveyor/overridden layouts may differ. Verify the loaded setting before declaring that prior. A visible top face or object center is not automatically on the plane; use two-view geometry when its height is unknown.

```json
{"control_step": 0, "plane_z_m": 0.765, "targets": [{
  "object_id": "red_block", "point": "table_contact",
  "description": "visible red block contact with the table", "kind": "support_plane",
  "height_above_plane_m": 0,
  "observations": [{"camera": "cam_head", "pixel": [320, 240]}]
}]}
```

Each `robot.localize` call accepts **1 to 12 targets**, with 1 to 3 camera observations per target and exactly two pixel coordinates `[u,v]` per view. For more than 12 targets, split into batches of at most 12 using the same observation and `control_step`. If any motion occurs between batches, observe again and rebuild the remaining annotations from that new observation. Never silently drop targets or relax validation. A point name is unique within an object, and each target's camera names must be distinct. The tool returns a structured rejection when the step is stale, a pixel is out of bounds, a height is unknown or an input is malformed. Correct the stated field; never clip pixels or guess a height to force acceptance.

## CLI localization

Use the packaged `run_localize.py` CLI for ordinary direct/plane localization as well as midpoint, circle or curved-surface modes. Read [the batch example in inputs.md](inputs.md#batch-annotations-for-run_localizepy), copy the exact current `observation.spatial_localization.case_path`, write `{"targets":[...]}` with `points_px:[[u,v]]`, and pass `--targets /workspace/annotations.json`. Unknown fields and incompatible frames are rejected explicitly.

Point IDs are unique **within each object**: two different objects may both have `id: top_center`. Repeating the same `(object_id,id)` in a batch is an error; combine that point's camera observations instead. For a CLI batch of known-height points on one plane, `--plane-z Z` declares the plane in the working copy; `--height H` fills only missing per-target heights. Different planes need separate batches. Keep the packaged observation unchanged.

## Results and corrections

`robot.localize` returns the exact `result` and `memory` paths. Use them rather than constructing filenames or importing the result again. For the advanced CLI, normally omit `--output`: it allocates an evidence directory. Optional `--output /workspace/localization/attempt` names a directory, not a JSON file. Reusing it allocates a fresh sibling attempt and preserves older evidence, including failed attempts.

If input is rejected, correct the reported field and retry. Shape errors and actual pixel bounds are reported separately. Do not clip pixels, guess a resize factor, remove geometric checks or project an unknown-height point onto the table to make validation pass. If geometry rejects a point, use its reason to improve physical-point matching or acquire a better overlapping view. After robot motion or a stale-observation response, call `robot.observe` again and use its new control step. Routine use requires no source-code inspection, matrix reconstruction or dependency installation.

## Task spatial memory and geometry limits

`/workspace/space.json` is shared memory for this task runtime. Use `robot.spatial_memory` with `{"operation":"show"}` to read it, or `{"operation":"stale","object_id":"red_block"}` to mark an object stale after motion, slippage or uncertain execution. Each object contains named points with environment-origin XYZ metres, uncertainty, evidence and native control step. Localization merges accepted estimates and marks rejected re-estimates stale. Avoid direct edits to the memory schema.

The `space_memory.py` CLI is also a normal supported memory interface. Reuse memory within the same episode; never reuse another task/reset's memory. Localization and memory changes do not execute robot motion. Object points are not TCP grasp poses; choose orientation, object-to-TCP offset and clearance separately.

Use only RGB, calibration and explicit geometric priors. Environment depth, hidden object poses, asset anchors, CAD/mesh, segmentation IDs and evaluation truth are not inputs. Reprojection and conditional uncertainty do not prove semantic identity or an assumed support height. Advanced modes, uncertainty parameters and host integration are in [inputs.md](inputs.md); the lower-level `localize_rgb.py` is for host diagnostics.

## Code planning and flexible VLA assistance

Prefer localization of the object and destination for code-computed TCP planning. Use accepted non-stale points with object-specific orientation, offset and clearance; object points are not grasp poses. Direct EEF/TCP targets may handle approach, final alignment, descent, grasp closure, insertion, release and subsequent corrections. Observe after each stage and refresh moved-object memory before metric reuse.

If localization remains unavailable after a targeted correction, use visually grounded VLA assistance when permitted. Likewise, a stalled TCP approach may use VLA recovery, then return to direct TCP refinement. No named memory point or distance threshold is required for VLA. Keep its short visual instruction stable within the same subgoal; numeric corrections and memory IDs belong in code/TCP targets. Inspect the proposal and choose a native 5..50-step prefix by stage and uncertainty. A geometric rejection is not an unsupported tool or a reason by itself to end the episode. If all permitted progress is genuinely blocked, cite the actual failed call and returned error rather than inventing an interface failure.
