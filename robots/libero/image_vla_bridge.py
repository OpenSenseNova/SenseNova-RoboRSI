# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Run image-owned actions and VLA skills; the adapter owns every physical step."""

from __future__ import annotations

import base64
import contextlib
import json
import os
import selectors
import subprocess
import sys
from pathlib import Path


def encode(value):
    import numpy as np

    if isinstance(value, np.ndarray):
        return {
            "__array__": base64.b64encode(value.tobytes()).decode(),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def decode(value):
    if "__array__" in value:
        import numpy as np

        return np.frombuffer(
            base64.b64decode(value["__array__"], validate=True), dtype=value["dtype"]
        ).reshape(value["shape"])
    return value


DIRECT_ACTIONS = frozenset(
    {"move_to", "move_pose", "rotate_pitch", "rotate_wrist", "release", "set_gripper"}
)


def image_arguments(data):
    """Translate the public rotation schema without changing image defaults."""
    result = dict(data)
    name = result.pop("primitive")
    if name in {"rotate_pitch", "rotate_wrist"}:
        axis = "pitch" if name == "rotate_pitch" else "yaw"
        for key in ("target", "delta"):
            if key in result:
                result[key + "_" + axis] = result.pop(key)
    return name, result


def observation(adapter):
    result = {
        k: adapter._latest.get(k)
        for k in (
            "main_images",
            "wrist_images",
            "extra_view_images",
            "states",
            "task_descriptions",
        )
    }
    facade = getattr(adapter._env, "_facade", adapter._env)
    if hasattr(facade, "raw_obs"):
        result["robot0_eef_quat"] = facade.raw_obs()["robot0_eef_quat"]
    else:
        # Test environments may expose only the equivalent rotation vector.
        from scipy.spatial.transform import Rotation

        result["robot0_eef_quat"] = Rotation.from_rotvec(
            result["states"][3:6]
        ).as_quat()
    return result


class ImageSkillSession:
    def __init__(self, adapter, data):
        from robots.libero.image_tools import image_manifest

        self.adapter = adapter
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}
        }
        env["PYTHONNOUSERSITE"] = "1"
        self.log = (adapter._native_dir / "image-action-worker.log").open("a")
        self.process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            env=env,
            cwd="/tmp",
        )
        self.pending = False
        self.done = False
        self.send(
            {
                "manifest": image_manifest(),
                "data": data,
                "observation": observation(adapter),
            }
        )

    def send(self, data):
        self.process.stdin.write((json.dumps(data, default=encode) + "\n").encode())
        self.process.stdin.flush()

    def receive(self):
        # Unbuffered line protocol with a bounded wait, including worker crashes.
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(150):
                raise TimeoutError("Image action worker did not respond within 150s")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(
                "Image action worker exited; inspect image-action-worker.log"
            )
        return json.loads(line, object_hook=decode)

    def reply_step(self):
        a = self.adapter
        self.send(
            {
                "observation": observation(a),
                "terminated": a._primary_success,
                "truncated": a._ended and not a._primary_success,
            }
        )
        self.pending = False

    def events(self):
        while not self.done:
            event = self.receive()
            kind = event["kind"]
            if kind == "predict":
                import numpy as np

                obs = event["observation"]
                wire = {
                    k: None
                    if obs.get(k) is None
                    else np.asarray(obs[k], dtype=np.uint8)[None]
                    for k in ("main_images", "wrist_images", "extra_view_images")
                }
                wire.update(
                    states=np.asarray(obs["states"], dtype=np.float32)[None],
                    task_descriptions=[obs["task_descriptions"]],
                )
                actions = np.asarray(
                    self.adapter._tool_clients["vla"].call(
                        "vla.predict", args=(wire, {"mode": "eval"}), timeout_s=120
                    )
                )
                if (
                    actions.ndim != 3
                    or actions.shape[0] != 1
                    or actions.shape[2] != 7
                    or not np.isfinite(actions).all()
                ):
                    raise ValueError("Invalid VLA action chunk")
                self.send({"actions": actions[0]})
            elif kind == "step":
                if self.adapter._ended:
                    raise RuntimeError(
                        "Image skill attempted an action after episode termination"
                    )
                self.pending = True
                yield event["action"]
                self.reply_step()
            elif kind == "result":
                self.adapter._tool_feedback = event["result"]
                self.done = True
            else:
                raise RuntimeError("Unknown image VLA worker event")

    def finish(self):
        # The common adapter breaks immediately on success/budget. Acknowledge
        # that last step so the original skill can finish its diagnostics.
        if self.pending:
            self.reply_step()
            for _ in self.events():
                raise RuntimeError("Unexpected action while finalizing image skill")

    def close(self):
        try:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        finally:
            self.process.stdin.close()
            self.process.stdout.close()
            self.log.close()


def worker():
    channel_in, channel_out = sys.stdin, sys.stdout
    request = json.loads(channel_in.readline(), object_hook=decode)

    def ask(message):
        channel_out.write(json.dumps(message, default=encode) + "\n")
        channel_out.flush()
        reply = channel_in.readline()
        if not reply:
            raise RuntimeError("Image skill parent disconnected")
        return json.loads(reply, object_hook=decode)

    import hashlib

    manifest = request["manifest"]
    source = Path(manifest["root"]) / manifest["source_file"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Original image skill changed")
    sys.path.insert(0, manifest["root"])
    with contextlib.redirect_stdout(sys.stderr):
        from robots.libero.tools import LiberoPrimitives

        class Model:
            def predict(self, obs, options):
                assert options == {"mode": "eval"}
                return ask({"kind": "predict", "observation": obs})["actions"]

        class Environment:
            terminated = False
            truncated = False
            return_all_frames = False

            def __init__(self):
                self.obs = request["observation"]

            def raw_obs(self):
                return {"robot0_eef_quat": self.obs["robot0_eef_quat"]}

            def step(self, action):
                if self.terminated or self.truncated:
                    raise RuntimeError(
                        "Image action attempted a step after termination"
                    )
                reply = ask({"kind": "step", "action": action})
                self.terminated = reply["terminated"]
                self.truncated = reply["truncated"]
                self.obs = reply["observation"]
                return self.obs, 0.0, self.terminated, self.truncated, {}

            def chunk_step(self, actions, **kwargs):
                last = None
                for action in actions:
                    if self.terminated or self.truncated:
                        break
                    last, *_ = self.step(action)
                if last is None:
                    raise ValueError("Image skill executed an empty action chunk")
                return last, 0.0, self.terminated, self.truncated, {}

        primitives = LiberoPrimitives(Environment(), Model(), None, lambda: None)
        primitives.set_obs(request["observation"])
        name, data = image_arguments(request["data"])
        if name not in DIRECT_ACTIONS | {"pi0_pick", "pi0_doubled"}:
            raise ValueError("Unsupported image skill")
        result = getattr(primitives, name)(**data)
    channel_out.write(
        json.dumps({"kind": "result", "result": result}, default=encode) + "\n"
    )
    channel_out.flush()


if __name__ == "__main__":
    worker()
