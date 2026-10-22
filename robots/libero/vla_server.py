# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Isolated OpenPI HTTP inference server; never executes simulator actions."""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

PROTOCOL = "roborsi-libero-openpi-v1"
MAX_REQUEST = 48 * 1024 * 1024


def decode_observation(payload: dict) -> dict:
    """Convert the bounded JSON transport into OpenPI's LIBERO observation."""
    if not isinstance(payload, dict) or payload.get("protocol") != PROTOCOL:
        raise ValueError("VLA protocol mismatch")
    instruction, images = payload.get("instruction"), payload.get("images")
    if (
        not isinstance(instruction, str)
        or not 0 < len(instruction.strip()) <= 2048
        or not isinstance(images, list)
        or len(images) != 2
        or type(payload.get("rotate_images")) is not bool
    ):
        raise ValueError("VLA needs an instruction and two camera images")
    state = np.asarray(payload.get("state"), dtype=np.float32)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("VLA state must be a finite eight-vector")
    decoded = []
    for value in images:
        if not isinstance(value, str) or len(value) > 24 * 1024 * 1024:
            raise ValueError("VLA camera exceeds size limit")
        binary = base64.b64decode(value, validate=True)
        with Image.open(io.BytesIO(binary)) as image:
            if image.width * image.height > 4096 * 4096:
                raise ValueError("VLA camera dimensions exceed limit")
            pixels = np.asarray(image.convert("RGB"))
        if payload["rotate_images"]:
            pixels = pixels[::-1, ::-1]
        decoded.append(np.ascontiguousarray(pixels))
    return {
        "observation/image": decoded[0],
        "observation/wrist_image": decoded[1],
        "observation/state": state,
        "prompt": instruction,
    }


def make_server(policy, *, host="127.0.0.1", port=8000, policy_config="pi05_libero"):
    """Create a testable server around an already loaded and warmed policy."""
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            logging.info(format, *args)

        def respond(self, status, document):
            body = json.dumps(document, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/v1/health":
                self.respond(
                    200,
                    {
                        "protocol": PROTOCOL,
                        "policy_config": policy_config,
                        "ready": True,
                    },
                )
            else:
                self.respond(404, {"error": "unknown route"})

        def do_POST(self):
            if self.path != "/v1/infer":
                self.respond(404, {"error": "unknown route"})
                return
            try:
                self.connection.settimeout(30)
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_REQUEST:
                    raise ValueError("invalid request size")
                request = json.loads(self.rfile.read(length))
                observation = decode_observation(request)
            except (ValueError, TypeError, OSError):
                self.respond(400, {"error": "invalid VLA observation"})
                return
            try:
                with lock:
                    actions = np.asarray(policy.infer(observation)["actions"])
                if (
                    actions.ndim != 2
                    or actions.shape[1] != 7
                    or not 1 <= len(actions) <= 1000
                    or not np.isfinite(actions).all()
                ):
                    raise ValueError("invalid model actions")
                self.respond(200, {"protocol": PROTOCOL, "actions": actions.tolist()})
            except Exception:
                logging.exception("OpenPI inference failed")
                self.respond(500, {"error": "OpenPI inference failed"})

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--policy-config", default="pi05_libero")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    from scripts.evaluation.downloads import prepared_openpi_checkpoint

    checkpoint = prepared_openpi_checkpoint(args.checkpoint)
    # Heavy imports exist only in the isolated OpenPI environment.
    from openpi.policies import policy_config
    from openpi.training import config
    from openpi_client import image_tools

    native_policy = policy_config.create_trained_policy(
        config.get_config(args.policy_config), checkpoint
    )

    class Policy:
        def infer(self, observation):
            for name in ("observation/image", "observation/wrist_image"):
                observation[name] = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(observation[name], 224, 224)
                )
            return native_policy.infer(observation)

    policy = Policy()
    # Publish readiness only after parameters load and first JIT/inference complete.
    policy.infer(
        {
            "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/state": np.zeros(8, dtype=np.float32),
            "prompt": "pick up the object",
        }
    )
    with make_server(
        policy, host=args.host, port=args.port, policy_config=args.policy_config
    ) as server:
        server.serve_forever()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
