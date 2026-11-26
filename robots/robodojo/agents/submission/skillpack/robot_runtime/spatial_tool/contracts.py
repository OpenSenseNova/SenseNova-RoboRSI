"""Public request contracts and a deliberately small output vocabulary."""
from copy import deepcopy

STATUSES = ("success", "estimated", "ambiguous", "occluded", "insufficient_views",
            "invalid_calibration", "invalid_geometry", "low_confidence")
VECTOR = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
TEXT = {"type": "string", "minLength": 1, "maxLength": 2000}
FRAME = {"type": "string", "const": "environment_origin"}
NONNEGATIVE = {"type": "number", "minimum": 0}
PLANE = {
    "type": "object", "additionalProperties": False,
    "required": ["normal", "b", "coordinate_frame", "uncertainty_m"],
    "properties": {"normal": VECTOR, "b": {"type": "number"},
                   "coordinate_frame": FRAME, "uncertainty_m": NONNEGATIVE,
                   "source": TEXT},
    "description": "Plane n dot P + b = 0 in metres; normal need not be unit length. Uncertainty is normal-distance sigma in metres.",
}
PIXEL = {
    "name": "spatial_localize_pixel",
    "description": "Map an ORIGINAL RGB pixel to environment XYZ by intersecting its snapshot camera ray with a declared plane or absolute known Z height. Uses no grounding model. Copy observation_id/image_id from observe. Geometry is conditional on the point actually belonging to the specified plane; give plane_membership=violated to reject a known mismatch.",
    "input_schema": {"type": "object", "additionalProperties": False,
        "required": ["camera_id", "pixel", "coordinate_frame", "reference_point"],
        "anyOf": [{"required": ["observation_id"]}, {"required": ["image_id"]}],
        "oneOf": [{"required": ["target_plane"], "not": {"required": ["known_height_m"]}},
                  {"required": ["known_height_m"], "not": {"required": ["target_plane"]}}],
        "properties": {"camera_id": TEXT, "observation_id": TEXT, "image_id": TEXT,
            "pixel": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
            "coordinate_frame": FRAME, "reference_point": TEXT, "target": TEXT,
            "target_plane": PLANE, "known_height_m": {"type": "number"},
            "height_uncertainty_m": NONNEGATIVE,
            "pixel_uncertainty_px": {"type": "number", "exclusiveMinimum": 0},
            "plane_membership": {"type": "string", "enum": ["assumed", "confirmed", "violated"]}}},
}


def tool_spec(mode="pixel"):
    if mode != "pixel":
        raise ValueError("Only pixel localization is supported")
    return deepcopy(PIXEL)


class LocalizationError(Exception):
    def __init__(self, status, reason):
        if status not in STATUSES:
            raise ValueError("Unknown localization status")
        self.status, self.reason = status, reason
        super().__init__(reason)


def failure(request, status, reason):
    return dict(status=status, target=request.get("target", "pixel"),
                reference_point=request.get("reference_point", "unspecified"),
                coordinate_frame=request.get("coordinate_frame", "environment_origin"),
                units="meters", reason=reason)


def public_result(value, request):
    """Allowlist only: internal traces, annotations, calibration and paths never escape."""
    if not isinstance(value, dict) or value.get("status") not in STATUSES:
        return failure(request, "low_confidence", "invalid_worker_result")
    allowed = {"status", "target", "reference_point", "position_m", "coordinate_frame", "units",
               "method", "uncertainty_m", "reason", "assumptions"}
    result = {key: item for key, item in value.items() if key in allowed}
    if result["status"] not in ("success", "estimated"):
        return failure(request, result["status"], result.get("reason", "localization_rejected"))
    # A successful process exit alone is never treated as a successful localization.
    from math import isfinite
    point, sigma = result.get("position_m"), result.get("uncertainty_m")
    if (not isinstance(point, list) or len(point) != 3
            or any(type(x) not in (float, int) or not isfinite(x) for x in point)
            or type(sigma) not in (float, int) or not isfinite(sigma) or sigma < 0
            or result.get("coordinate_frame") != "environment_origin"
            or result.get("units") != "meters" or not result.get("reference_point")):
        return failure(request, "low_confidence", "invalid_worker_result")
    return result
