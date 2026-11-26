"""Load immutable host-captured RGB/calibration pairs, including wrist poses."""
import hashlib
import io
import json
from pathlib import Path
from PIL import Image
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from runtime_records import read as read_host_state
from .validation import camera, require
from .contracts import LocalizationError


def observation_id(directory, case_name):
    return directory + ":" + Path(case_name).stem


def public_snapshot(spatial):
    path = Path(spatial["case_path"])
    identity = observation_id(path.parent.name, path.name)
    return dict(observation_id=identity, control_step=spatial["control_step"],
        coordinate_frame="environment_origin", units="meters", pixel_order="u,v",
        images={name: {"image_id": identity+":"+name, **size}
                for name, size in spatial.get("image_sizes", {}).items()})


class Snapshot:
    def __init__(self, case, receipt, images, identity, latest_path, latest_digest):
        self.case, self.receipt, self.images = case, receipt, images
        self.identity, self.latest_path, self.latest_digest = identity, latest_path, latest_digest
        self.cameras = case["cameras"]
        self.step = receipt["control_step"]
        self.frame = case["frame"]

    @classmethod
    def load(cls, descriptor, image_root, limits):
        try:
            root = Path(image_root).resolve(strict=True)
            ref = descriptor["case_path"]
            require(isinstance(ref, str) and ref.startswith("/observations/"),
                    "invalid_calibration", "invalid_snapshot_reference")
            path = (root/ref.removeprefix("/observations/")).resolve(strict=True)
            require(path.is_relative_to(root), "invalid_calibration", "snapshot_outside_capture_root")
            raw = path.read_bytes();case = json.loads(raw)
            receipt_path = path.with_suffix(".meta.json")
            receipt = json.loads(receipt_path.read_bytes())
            latest_path = path.parent/"latest_spatial.json"
            latest = read_host_state(latest_path)
            require(receipt["case_sha256"] == hashlib.sha256(raw).hexdigest(),
                    "invalid_calibration", "snapshot_checksum_mismatch")
            require(type(receipt["control_step"]) is int and receipt["control_step"] >= 0
                    and descriptor["control_step"] == receipt["control_step"]
                    and latest["control_step"] == receipt["control_step"] and latest["case_file"] == path.name,
                    "invalid_calibration", "stale_snapshot")
            require(case.get("frame") == "environment_origin" and case.get("units") == "meters",
                    "invalid_calibration", "snapshot_coordinate_frame_mismatch")
            require(isinstance(case.get("cameras"), dict) and bool(case["cameras"]),
                    "invalid_calibration", "missing_camera_calibration")
            identity = observation_id(path.parent.name, path.name)
            require(receipt.get("observation_id") == identity,
                    "invalid_calibration", "missing_or_mismatched_snapshot_attestation")
            require(receipt.get("capture_source") == "render_projection_and_actual_optical_pose",
                    "invalid_calibration", "unverified_camera_pose_source")
            bindings = receipt.get("camera_bindings", {})
            images, times = {}, []
            for name, cam in case["cameras"].items():
                # Original geometry files omit dimensions; the host descriptor binds
                # them to this capture, then the decoded image verifies them below.
                size = descriptor["image_sizes"][name]
                cam.setdefault("image_size", [size["height"], size["width"]])
                camera(cam)
                binding = bindings.get(name, {})
                require(binding.get("image_control_step") == receipt["control_step"]
                        and binding.get("pose_control_step") == receipt["control_step"]
                        and binding.get("calibration_control_step") == receipt["control_step"]
                        and binding.get("observation_id") == identity,
                        "invalid_calibration", "image_calibration_or_wrist_pose_mismatch")
                image_path = (path.parent/cam["image"]).resolve(strict=True)
                require(image_path.is_relative_to(path.parent), "invalid_calibration", "image_outside_snapshot")
                data = image_path.read_bytes()
                require(hashlib.sha256(data).hexdigest() == receipt["images"][name],
                        "invalid_calibration", "image_checksum_mismatch")
                # Inspect the stored RGB container and dimensions; no color conversion.
                with Image.open(io.BytesIO(data)) as im:
                    require(im.format in ("JPEG", "PNG") and im.mode == "RGB"
                            and [im.height, im.width] == cam["image_size"],
                            "invalid_calibration", "image_dimensions_or_format_mismatch")
                    mime = "image/jpeg" if im.format == "JPEG" else "image/png"
                    im.verify()
                images[name] = dict(data=data, mime=mime, image_id=identity+":"+name)
                # All cameras captured while physics is frozen are simultaneous in
                # simulation time. If capture timestamps exist, also verify them.
                if "capture_time_s" in binding:
                    import math
                    value = binding["capture_time_s"]
                    require(type(value) in (int, float) and math.isfinite(value),
                            "invalid_calibration", "invalid_capture_timestamp")
                    times.append(value)
            require(not times or (len(times) == len(images) and max(times)-min(times) <= limits.max_view_time_skew_s),
                    "invalid_calibration", "unsynchronized_camera_snapshots")
            return cls(case, receipt, images, identity, latest_path, latest)
        except LocalizationError:
            raise
        except (OSError, ValueError, KeyError, TypeError):
            raise LocalizationError("invalid_calibration", "snapshot_or_calibration_unavailable") from None

    def check_current(self):
        try:
            same = read_host_state(self.latest_path) == self.latest_digest
        except (OSError, ValueError, KeyError, TypeError):
            same = False
        require(same, "invalid_calibration", "snapshot_changed_during_localization")
