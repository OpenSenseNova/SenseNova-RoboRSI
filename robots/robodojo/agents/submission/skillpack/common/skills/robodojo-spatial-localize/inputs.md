# RGB localization inputs

Normal task runs may use `robot.localize`, which binds the current calibrated observation and writes valid annotation JSON, or the bundled `run_localize.py` CLI for ordinary and advanced geometry modes. The host preinstalls NumPy, SciPy and OpenCV with SIFT; do not install packages in the model session. The schema below documents advanced annotations and host-side integration.

## Batch annotations for run_localize.py

Keep the host observation file unchanged. Here is a batch of two **actual plane-contact** points; replace IDs, cameras and pixels with measured evidence:

```json
{"targets": [
  {"object_id":"red_block", "id":"contact", "description":"visible red block contact point",
   "kind":"support_plane", "height_above_plane_m":0,
   "observations":{"cam_head":{"points_px":[[320,240]]}}},
  {"object_id":"blue_block", "id":"contact", "description":"visible blue block contact point",
   "kind":"support_plane", "height_above_plane_m":0,
   "observations":{"cam_head":{"points_px":[[350,240]]}}}
]}
```

```bash
python3 .agents/skills/robodojo-spatial-localize/scripts/run_localize.py \
  --case /observations/evaluation-0001/00000_spatial_case.json \
  --targets /workspace/annotations.json --plane-z 0.765
```

The plane height is an explicit agent-declared prior, valid only after confirming the loaded setting. The wrapper adds that plane to a working copy; it never edits the checksummed host case. A supplied `support_plane` name is preserved; one `--plane-z` cannot declare multiple different named planes. `--height H` fills missing per-target heights; explicit per-target heights take precedence. Unknown height requires `kind: direct` with the same physical point in two views, without plane fields.

Equivalent target containers (a list or one target object) and a single `points_px:[u,v]` are normalized. Optional top-level `frame`/`units` must match the host case. Other unknown fields are rejected with their names. The identity is `(object_id,id)`: different objects can share a point name, but one object/point cannot be duplicated. Add multiple camera entries under that point's `observations` instead.

Omit `--output` for automatic evidence paths. Always use the wrapper's returned `result` and `memory` paths. Its result lives under an attempt's `geometry/points3d.json`; that layout differs from the lower-level host CLI below. A failed attempt preserves evidence and does not block a corrected retry using the same optional output prefix.


## Camera case

```json
{
  "frame": "robot/base_link",
  "units": "meters",
  "cameras": {
    "reference": {
      "image": "reference.png",
      "K": [[500,0,320],[0,500,240],[0,0,1]],
      "T_base_camera_cv": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
      "distortion": []
    }
  }
}
```

Images are ordinary color images resolved relative to `case.json`. Transforms are right-handed SE(3), OpenCV camera axes right/down/forward. Supply at least two useful viewpoints for unconstrained geometry. The support-plane mode below can use one calibrated view. Keep object and calibration stable during acquisition.

## Targets

```json
{
  "targets": [{
    "id": "contact_center",
    "object_id": "selected_object",
    "description": "Center of the selected visible top face",
    "kind": "circle_center",
    "observations": {
      "reference": {"points_px": [[320,240]], "bbox_px": [290,210,350,270]},
      "other_view": {"points_px": [[280,240]], "bbox_px": [250,210,310,270]}
    }
  }]
}
```

Kinds:

| `kind` | Annotations and assumptions |
|---|---|
| `support_plane` | One physical point per view; a named explicit plane and `height_above_plane_m`. Single-view or multiple-view ray/plane intersection. |
| `direct` | One physical point per view, `points_px:[[u,v]]`; robust calibrated consensus. |
| `midpoint` | Two unordered physical endpoints per view; recover both in3D, then midpoint. |
| `circle_center` | One seed per view; adaptive RGB boundary ellipse refinement. No known radius or native pose. |
| `curved_strip_crown` | Set `color` to `green`, `blue`, or `yellow`; provide `bbox_px` for the same whole strip in at least three views. Fits generic horizontal rounded strip with convex parabolic cross section. Pose, sizes, rise and thickness are inferred. Raw silhouette residual does not prove this shape prior is correct. |
| `axis_surface_intersection` | Use `regions` below. Assumes a bilaterally symmetric thin curved sheet whose center is aligned with the observed vertical post. Fits dimensions and both signed curvatures; uses RGB-triangulated local surface supports when available. |

For an axis/surface construction:

```json
{
  "id": "surface_intersection",
  "object_id": "selected_support",
  "description": "Visible support axis intersecting its top surface",
  "kind": "axis_surface_intersection",
  "regions": {
    "side_view": {"axis_px": [[320,300],[320,350]]},
    "upper_view": {"polygon_px": [[250,180],[390,180],[390,300],[250,300]]}
  }
}
```

Provide at least three sufficiently resolved sheet outlines and a visible post axis in one or more calibrated views. Outline the sheet, including its thin edge, while excluding the lower base, background and other objects. The coordinates above illustrate the schema and are not a runnable calibration or scene fixture.

`surface_method` may be `parametric` (default) or `local_quadratic`. The latter intersects the recovered axis with a local quadratic fit to triangulated RGB supports; it needs at least8 supports, sufficient spatial coverage and reserved-point validation. The parametric method also incorporates local support measurements and validates held-out points. Neither mode changes the semantic point to a SIFT feature.

## Known tabletop or support plane (core mode)

The plane equation is `normal · X = offset_m` in its declared frame; `normal` must be unit length and point away from the support surface (upward for a horizontal table). Height above the plane is measured along that normal, in meters. No implicit scene height is inserted by the runtime.

A minimal **synthetic calibration example**, with the output frame explicitly named `world`, a downward-looking camera one meter above the table, and a tabletop pixel:

```json
{
  "frame": "world",
  "units": "meters",
  "support_planes": {
    "table": {
      "frame": "world",
      "normal": [0, 0, 1],
      "offset_m": 0.765,
      "height_sigma_m": 0.001,
      "source": "Confirmed current default table: center z=0.74 m, thickness=0.05 m"
    }
  },
  "cameras": {
    "head": {
      "image": "head.jpg",
      "K": [[500,0,320],[0,500,240],[0,0,1]],
      "T_base_camera_cv": [[1,0,0,0],[0,-1,0,0],[0,0,-1,1.765],[0,0,0,1]],
      "distortion": []
    }
  }
}
```

```json
{
  "targets": [{
    "id": "table_contact",
    "object_id": "selected_block",
    "description": "Visible bottom contact point on the tabletop",
    "kind": "support_plane",
    "support_plane": "table",
    "height_above_plane_m": 0,
    "observations": {"head": {"points_px": [[320,240]]}}
  }]
}
```

This synthetic center ray returns `[0, 0, 0.765]`. Replace the image and calibration with the current measured camera data; the example is not a real head-camera transform. Run the same `scripts/localize_rgb.py --case ... --targets ... --output ...` command as other modes.

For a known top face 4 cm above the support plane, annotate that face and set `height_above_plane_m: 0.04`; if its height has 2 mm uncertainty, also set `target_height_sigma_m: 0.002`. A nonnegative height is required explicitly, including zero. Unknown height is a reason to use `direct` multi-view triangulation, not a reason to assume zero. Multiple observations must identify the same physical point.

### Frames and uncertainty

If the plane is declared in `world` but `case.frame` is a named robot Base, include `T_base_plane_frame` in that plane. It maps points from the declared plane frame into `case.frame`, with a right-handed rigid transform. For `X_base = R X_plane + t`, the transformed plane is `n_base = R n_plane`, `d_base = d_plane + n_base · t`. The target offset is then `d_base + height_above_plane_m`. If the frames match, omit the transform. Do not assume an arm's Base frame shares World's height or orientation.

`height_sigma_m` and `source` are required plane fields. Choose a measured/configuration-derived uncertainty; zero is permitted only for an intended exact prior. Optional `extrinsic_sigma_m` adds an isotropic calibration floor. Shared plane and target-height errors do not average down across views. Unmodeled plane tilt, calibration correlations, support bounds, occlusion, semantic selection and an incorrect height remain limitations; this mode does not estimate full object pose or infer object dimensions.

Optional per-target `plane_options` (all positive):

| Field | Default | Meaning |
|---|---:|---|
| `pixel_sigma_px` | 1 | Pixel annotation standard deviation |
| `min_ray_plane_angle_deg` | 3 | Reject rays closer to parallel with the plane |
| `max_view_disagreement_m` | 0.01 | Maximum distance between per-view intersections |
| `max_reprojection_px` | 3 | Maximum fused-point reprojection residual |
| `max_position_sigma_m` | 0.01 | Maximum estimated largest-axis standard deviation |

These gates are configurable engineering checks, not measured accuracy claims. `estimate.priors` records the declared plane, transformed target plane and height assumption; `per_view` records each intersection and angle. `position_base_m` and covariance are in `case.frame` (the historical field name is retained even when that frame is explicitly `world`). A rejected estimate must not be used as a coordinate.

## Outputs and evidence

`points3d.json` retains each target's ID, object ID, description and kind. `estimate.accepted=false` means no usable coordinate for that request. The helper emits declared priors and detailed geometry. Artifacts contain refined raw RGB boundaries, local supports and target projections. Derived positions remain in `case.frame`, meters.

This version was developed on10 fixed semantic requests. Its measured success on that set must not be presented as a guaranteed accuracy for arbitrary objects, camera poses or new scenes.
