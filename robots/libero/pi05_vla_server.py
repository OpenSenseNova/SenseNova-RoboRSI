# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""OpenPI HTTP entry point backed by the pinned image's RLinf loader."""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    root = Path(os.environ.get("ROBORSI_IMAGE_TOOLS_ROOT", "/opt/roborsi-image-tools"))
    if not root.exists() and "ROBORSI_IMAGE_TOOLS_ROOT" not in os.environ:
        root = Path("/opt/roborsi")
    lock = json.loads((here / "image_tools.lock.json").read_text())
    name = "roborsi/robots/components/pi05_vla_server.py"
    if (
        hashlib.sha256((root / name).read_bytes()).hexdigest()
        != lock["service_sources"][name]
    ):
        raise ValueError("Image VLA loader differs from pinned version")
    sys.path.insert(0, str(root))
    import numpy as np

    from roborsi.robots.components import pi05_vla_server as image

    # Keep image loading arguments; the only added mode is OpenPI HTTP.
    parser = argparse.ArgumentParser()
    parser.add_argument("--embodiment", default="libero")
    parser.add_argument("--transport", choices=["http"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--model-path")
    parser.add_argument("--model-backend", default="openpi_pytorch")
    parser.add_argument("--repo-id")
    parser.add_argument("--norm-stats-path")
    parser.add_argument("--api", choices=["rpc", "openpi"], default="openpi")
    args = parser.parse_args()
    if args.api == "rpc":
        sys.argv = [a for a in sys.argv if a not in ("--api", "rpc")]
        return image.main()
    if args.embodiment != "libero":
        raise ValueError("OpenPI bridge requires LIBERO")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    facade = image.Pi05VLAFacade(
        model_path=args.model_path or os.environ["PI05_CHECKPOINT_PATH"],
        embodiment=args.embodiment,
        model_backend=args.model_backend,
        repo_id=args.repo_id,
        norm_stats_path=args.norm_stats_path,
    )
    spec = importlib.util.spec_from_file_location(
        "_openpi_http", here / "vla_server.py"
    )
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)

    class Policy:
        def infer(self, obs):
            wire = {
                "main_images": np.asarray(obs["observation/image"])[None],
                "wrist_images": np.asarray(obs["observation/wrist_image"])[None],
                "extra_view_images": None,
                "states": np.asarray(obs["observation/state"], dtype=np.float32)[None],
                "task_descriptions": [obs["prompt"]],
            }
            return {"actions": facade.predict(wire, {"mode": "eval"})[0]}

    policy = Policy()
    policy.infer(
        {
            "observation/image": np.zeros((256, 256, 3), dtype=np.uint8),
            "observation/wrist_image": np.zeros((256, 256, 3), dtype=np.uint8),
            "observation/state": np.zeros(8, dtype=np.float32),
            "prompt": "pick up the object",
        }
    )
    with transport.make_server(
        policy, host=args.host, port=args.port, policy_config="pi05_libero"
    ) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
