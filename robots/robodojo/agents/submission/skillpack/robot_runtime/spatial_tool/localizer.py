"""Pixel-to-plane geometry with compact metric output."""
import numpy as np
from .geometry import intersect_plane
from .validation import pixel, require


def _result(request, point, sigma, method, reference, **extra):
    return dict(status="success",
        target=request.get("target", "pixel"), reference_point=reference,
        position_m=np.asarray(point).tolist(), coordinate_frame="environment_origin", units="meters",
        method=method, uncertainty_m=float(sigma), **extra)


def pixel_localize(request, snapshot, limits):
    name = request["camera_id"]
    require(name in snapshot.cameras, "invalid_calibration", "unknown_camera")
    require("observation_id" not in request or request["observation_id"] == snapshot.identity,
            "invalid_calibration", "requested_observation_mismatch")
    require("image_id" not in request or request["image_id"] == snapshot.images[name]["image_id"],
            "invalid_calibration", "requested_image_mismatch")
    require(request.get("plane_membership") != "violated", "invalid_geometry", "point_not_on_target_plane")
    plane = request.get("target_plane")
    if plane is None:
        plane = dict(normal=[0, 0, 1], b=-request["known_height_m"], coordinate_frame=snapshot.frame,
                     uncertainty_m=request.get("height_uncertainty_m", 0.))
    p, sigma = intersect_plane(pixel(request["pixel"], snapshot.cameras[name]), snapshot.cameras[name],
        plane, snapshot.frame, limits, pixel_sigma=request.get("pixel_uncertainty_px"))
    snapshot.check_current()
    return _result(request, p, sigma, "support_plane_intersection", request["reference_point"],
                   assumptions=["The specified physical point lies on the declared target plane."])
