"""Independent input, calibration and workspace checks."""
from dataclasses import asdict, dataclass
import math
import numpy as np
from .contracts import LocalizationError


@dataclass(frozen=True)
class Limits:
    min_ray_plane_angle_deg: float = 3.
    max_reprojection_px: float = 3.
    max_uncertainty_m: float = .02
    pixel_sigma_px: float = 2.
    calibration_sigma_m: float = .001
    max_view_time_skew_s: float = .01
    workspace_min: tuple = (-2., -2., -.1)
    workspace_max: tuple = (2., 2., 2.5)

    @classmethod
    def parse(cls, value=None):
        result = cls(**(value or {}))
        for key, val in asdict(result).items():
            if key.startswith("workspace_"):
                if len(val) != 3 or not np.isfinite(val).all():
                    raise ValueError("Invalid workspace bounds")
            elif isinstance(val, bool) or not math.isfinite(val) or val <= 0:
                raise ValueError("Invalid spatial threshold: " + key)
        if (not 0 < result.min_ray_plane_angle_deg < 90
                or np.any(np.asarray(result.workspace_min) >= result.workspace_max)):
            raise ValueError("Invalid spatial thresholds or workspace")
        return result


def require(condition, status, reason):
    if not condition:
        raise LocalizationError(status, reason)


def finite_vector(value, size, status="invalid_geometry", reason="nonfinite_or_invalid_vector"):
    require(isinstance(value, (list, tuple, np.ndarray)), status, reason)
    try:
        a = np.asarray(value, dtype=float)
    except (ValueError, TypeError):
        raise LocalizationError(status, reason) from None
    require(a.shape == (size,) and np.isfinite(a).all(), status, reason)
    require(not any(isinstance(v, (bool, str)) for v in value), status, reason)
    return a


def pixel(value, camera):
    a = finite_vector(value, 2)
    h, w = camera["image_size"]
    require(0 <= a[0] < w and 0 <= a[1] < h, "invalid_geometry", "pixel_outside_original_image")
    return a


def camera(value):
    try:
        k = np.asarray(value["K"], float)
        t = np.asarray(value["T_base_camera_cv"], float)
        size = value["image_size"]
        distortion = np.asarray(value.get("distortion", []), float)
        valid = (k.shape == (3, 3) and t.shape == (4, 4)
            and np.isfinite(k).all() and np.isfinite(t).all()
            and np.allclose(k[2], [0, 0, 1]) and min(k[0, 0], k[1, 1]) > 0
            and np.isclose(k[0, 1], 0) and np.isclose(k[1, 0], 0)
            and abs(np.linalg.det(k)) > 1e-12 and len(size) == 2
            and all(type(n) is int and n > 0 for n in size)
            and np.allclose(t[3], [0, 0, 0, 1])
            and np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-6)
            and np.isclose(np.linalg.det(t[:3, :3]), 1., atol=1e-6)
            and distortion.ndim == 1 and distortion.size in (0, 4, 5, 8, 12, 14)
            and np.isfinite(distortion).all())
    except (KeyError, ValueError, TypeError, IndexError):
        valid = False
    require(valid, "invalid_calibration", "invalid_camera_matrices_or_image_size")


def point_result(position, sigma, limits):
    p = finite_vector(position, 3)
    require(np.all(p >= limits.workspace_min) and np.all(p <= limits.workspace_max),
            "invalid_geometry", "point_outside_workspace")
    require(math.isfinite(sigma) and 0 <= sigma <= limits.max_uncertainty_m,
            "low_confidence", "position_uncertainty_too_large")
    return p
