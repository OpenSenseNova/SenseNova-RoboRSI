"""Deterministic geometry. Reuses the existing calibrated RGB ray code.

The public plane convention here is n.P + b = 0. The old support_plane helper
uses n.P = offset; callers never need to know that internal convention.
"""
import importlib.util
from pathlib import Path
import numpy as np
from .validation import finite_vector, point_result, require

_legacy_path = Path(__file__).resolve().parents[2]/"common/skills/robodojo-spatial-localize/scripts/rgb_geometry.py"
_spec = importlib.util.spec_from_file_location("_spatial_legacy_geometry", _legacy_path)
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)


def plane_parameters(plane, frame):
    require(isinstance(plane, dict), "invalid_geometry", "target_plane_required")
    require(plane.get("coordinate_frame") == frame, "invalid_calibration", "plane_coordinate_frame_mismatch")
    n = finite_vector(plane.get("normal"), 3)
    length = float(np.linalg.norm(n))
    require(length > 1e-12, "invalid_geometry", "zero_plane_normal")
    b, sigma = plane.get("b"), plane.get("uncertainty_m")
    require(type(b) in (int, float) and np.isfinite(b), "invalid_geometry", "invalid_plane_offset")
    require(type(sigma) in (int, float) and np.isfinite(sigma) and sigma >= 0,
            "invalid_geometry", "invalid_plane_uncertainty")
    return n/length, b/length, float(sigma)


def _intersection(pixel, camera, n, b):
    c, d = G.ray(pixel, camera)
    denominator = float(n @ d)
    require(abs(denominator) > 1e-10, "invalid_geometry", "ray_parallel_to_plane")
    distance = -float(n @ c + b)/denominator
    require(distance > 0, "invalid_geometry", "intersection_behind_camera")
    return c+distance*d, abs(denominator)


def intersect_plane(pixel, camera, plane, frame, limits, *, pixel_sigma=None):
    n, b, plane_sigma = plane_parameters(plane, frame)
    p, sine = _intersection(pixel, camera, n, b)
    require(np.degrees(np.arcsin(np.clip(sine, 0, 1))) >= limits.min_ray_plane_angle_deg,
            "invalid_geometry", "ray_nearly_parallel_to_plane")
    projected, depth = G.project(p, camera)
    require(depth > 0 and np.isfinite(projected).all()
            and np.linalg.norm(projected-np.asarray(pixel)) <= limits.max_reprojection_px,
            "invalid_calibration", "unstable_distortion_inversion")
    eps = 1e-3
    pixel = np.asarray(pixel, float)
    j = np.column_stack([(_intersection(pixel+np.eye(2)[i]*eps, camera, n, b)[0]
                         - _intersection(pixel-np.eye(2)[i]*eps, camera, n, b)[0])/(2*eps) for i in range(2)])
    dh = (_intersection(pixel, camera, n, b+eps)[0]-_intersection(pixel, camera, n, b-eps)[0])/(2*eps)
    sigma_px = max(limits.pixel_sigma_px, pixel_sigma or limits.pixel_sigma_px)
    covariance = j @ j.T*sigma_px**2 + np.outer(dh, dh)*plane_sigma**2
    covariance += np.eye(3)*limits.calibration_sigma_m**2
    sigma = float(np.sqrt(max(0., np.linalg.eigvalsh(covariance).max())))
    return point_result(p, sigma, limits), sigma
