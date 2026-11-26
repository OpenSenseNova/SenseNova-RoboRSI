"""Lossless native RGB over private IPC; JPEG files are archival artifacts only.

The byte payload is consumed before returning a JSON-safe transition to the
runner. It must never be logged, archived or exposed to the model as text.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
TRANSPORT = "native_rgb_bytes_v1"
MAX_CAMERA_BYTES = 12 * 1024 * 1024


def encode_rgb(vision):
    """Serialize exact HWC RGB bytes, including noncontiguous/read-only arrays."""
    if set(vision) != set(CAMERAS):
        raise ValueError("Native RGB requires the three configured cameras")
    result = {}
    for camera in CAMERAS:
        values = np.asarray(vision[camera]["color"])
        if (values.dtype != np.uint8 or values.ndim != 3 or values.shape[2] != 3
                or min(values.shape) < 1 or values.nbytes > MAX_CAMERA_BYTES):
            raise ValueError("Native camera must be bounded uint8 HWC RGB")
        raw = values.tobytes(order="C")
        result[camera] = dict(shape=list(values.shape), dtype="uint8",
                              sha256=hashlib.sha256(raw).hexdigest(),
                              data=base64.b64encode(raw).decode("ascii"))
    return result


def decode_rgb(payload):
    """Decode bytes, not JPEG; reject wrong sizes/hashes before Model sees them."""
    if not isinstance(payload, dict) or set(payload) != set(CAMERAS):
        raise ValueError("Native RGB camera set differs")
    result = {}
    for camera in CAMERAS:
        item = payload[camera]
        shape = item.get("shape")
        if (item.get("dtype") != "uint8" or not isinstance(shape, list) or len(shape) != 3
                or any(type(n) is not int or n < 1 for n in shape) or shape[2] != 3):
            raise ValueError("Invalid native RGB shape/dtype")
        expected = shape[0] * shape[1] * shape[2]
        data = item.get("data")
        if (expected > MAX_CAMERA_BYTES or not isinstance(data, str)
                or len(data) != 4 * ((expected + 2) // 3)):
            raise ValueError("Native RGB payload size differs")
        try:
            raw = base64.b64decode(data, validate=True)
        except ValueError as error:
            raise ValueError("Invalid native RGB byte encoding") from error
        if len(raw) != expected or hashlib.sha256(raw).hexdigest() != item.get("sha256"):
            raise ValueError("Native RGB byte checksum differs")
        # A raw-byte ndarray view, not an image-bit decoder. Immutable bytes own
        # its storage; JPEG codecs and RGB/BGR conversion never enter this path.
        color = np.ndarray(tuple(shape), dtype=np.uint8, buffer=raw)
        result[camera] = {"color": color}
    return result


@lru_cache(maxsize=1)
def _shared_encoder():
    # The private simulator worker drops PYTHONPATH. Load the shared codec from
    # this frozen checkout, also when an isolated worktree has another name.
    import importlib.util
    import sys
    checkout = Path(__file__).resolve().parents[2]
    if 'XPolicyLab' not in sys.modules:
        spec = importlib.util.spec_from_file_location('XPolicyLab', checkout/'__init__.py',
                                                      submodule_search_locations=[str(checkout)])
        package = importlib.util.module_from_spec(spec)
        sys.modules['XPolicyLab'] = package
        spec.loader.exec_module(package)
    # A simulator environment can already import another XPolicyLab checkout.
    # Load this checkout's shared helper by path without replacing that package.
    spec = importlib.util.spec_from_file_location('_roborsi_archive_codec', checkout/'utils/process_data.py')
    codec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codec)
    return codec.encode_image_bit


def archive_rgb(vision, directory, frame):
    """Encode JPEG once for archives; raw RGB remains the live input contract."""
    encode_image_bit = _shared_encoder()

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cameras, artifacts = {}, {}
    for camera in CAMERAS:
        path = directory/f"{frame:05d}_{camera}.jpg"
        if path.exists() or path.is_symlink():
            raise ValueError("Refusing to overwrite a native RGB archive")
        path.write_bytes(encode_image_bit(vision[camera]["color"], quality=95, subsampling='444'))
        cameras[camera] = "obs://"+path.name
        artifacts[camera] = dict(file=path.name, bytes=path.stat().st_size,
                                 sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return cameras, artifacts


class NativeRgbBackend:
    """Consume a private raw-frame response and expose only metadata to logs."""
    def __init__(self, backend):
        self.backend = backend
        self._reply = self._vision = None

    def request(self, **request):
        previous = self._reply
        transition = request.get("op") in ("reset", "step", "finish")
        if transition:
            # A failed/uncertain request must never make an old frame look current.
            self._reply = self._vision = None
        reply = self.backend.request(**request)
        if transition:
            payload = reply.pop("_native_rgb", None)
            meta = reply.get("native_rgb")
            if (not isinstance(meta, dict) or meta.get("transport") != TRANSPORT
                    or meta.get("control_step") != reply.get("observation", {}).get("control_step")):
                raise ValueError("Native worker did not return the required lossless RGB frame")
            expected_frame = 0 if request["op"] == "reset" else (
                previous["native_rgb"]["frame"] + 1 if previous is not None else None)
            if type(meta.get("frame")) is not int or meta["frame"] != expected_frame:
                raise ValueError("Native RGB frame sequence differs")
            vision = decode_rgb(payload)
            for camera in CAMERAS:
                if payload[camera]["sha256"] != meta.get("rgb_sha256", {}).get(camera):
                    raise ValueError("Native RGB metadata checksum differs")
            self._reply, self._vision = reply, vision
        return reply

    def model_observation(self, reply):
        if reply is not self._reply or self._vision is None:
            raise ValueError("Only the current acknowledged native frame can reach Model")
        obs = reply["observation"]
        value = dict(instruction=obs["instruction"], state=deepcopy(obs["state"]),
                    vision={name: dict(color=self._vision[name]["color"]) for name in CAMERAS},
                    additional_info={"frequency": obs["control_frequency_hz"],
                                     **({"spatial_calibration": deepcopy(obs["spatial_calibration"])}
                                        if obs.get("spatial_calibration") is not None else {})})
        if "extra_info" in obs:
            value["extra_info"] = deepcopy(obs["extra_info"])
        if "roborsi_execution" in reply:
            value["roborsi_execution"] = deepcopy(reply["roborsi_execution"])
        return value

    def close(self):
        try:
            self.backend.close()
        finally:
            self._reply = self._vision = None


def create_backend(cfg, run, *, backend_factory=None):
    """Select this transport explicitly; do not rewrite custom/replay adapters."""
    root = Path(__file__).resolve().parent
    source = root/"vendor/robodojo_adaptive/source"
    original = source/"code/configurations/robodojo/20261005_standard_1x_10x/adapters/adapter.py"
    adapter, name = cfg["adapter"].rsplit(":", 1)
    if Path(adapter).resolve() != original.resolve() or name != "Adapter":
        raise ValueError("Raw RGB wrapper requires the original native joint/EEF adapter")
    configured = deepcopy(cfg)
    configured["adapter"] = str(root/"native_rgb_adapter.py")+":Adapter"
    if backend_factory is None:
        from robots.robodojo.agents.icl_pi05.benchmark import original_backend
        backend_factory, _ = original_backend(source)
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    path = run/"config.json"
    if path.exists() and json.loads(path.read_text()) != configured:
        raise ValueError("Native RGB config already exists with different settings")
    path.write_text(json.dumps(configured, indent=2)+"\n")
    return NativeRgbBackend(backend_factory(configured, run))


def verify_archive(records, images):
    """Verify a cloud/M JPEG copy against its original per-frame file hashes."""
    images = Path(images).resolve(strict=True)
    files = {}
    for line in Path(records).read_text().splitlines():
        row = json.loads(line)
        if row.get("transport") != TRANSPORT or set(row.get("archive", {})) != set(CAMERAS):
            raise ValueError("Invalid native RGB archive manifest")
        for item in row["archive"].values():
            name = item["file"]
            if Path(name).name != name or Path(name).suffix != ".jpg" or name in files:
                raise ValueError("Unsafe or duplicate native JPEG archive path")
            path = images/name
            if path.is_symlink() or not path.is_file():
                raise ValueError("Missing native JPEG archive")
            raw = path.read_bytes()
            if len(raw) != item["bytes"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
                raise ValueError("Native JPEG archive differs from original bytes")
            files[name] = len(raw)
    if not files:
        raise ValueError("Native JPEG archive manifest is empty")
    return dict(files=len(files), bytes=sum(files.values()), byte_identical=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verify original JPEG archive bytes after cloud/M transfer")
    parser.add_argument("--records", type=Path, required=True, help="Original rgb_frames.jsonl")
    parser.add_argument("--images", type=Path, required=True, help="Copied JPEG directory")
    options = parser.parse_args()
    print(json.dumps(verify_archive(options.records, options.images)))
