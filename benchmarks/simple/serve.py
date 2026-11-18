# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Launch the standard Benchmark HTTP service with explicit GPU and asset mounts."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import uuid
from pathlib import Path

from roborsi.evaluation.docker_common import validate_image_reference


def main() -> None:
    """Run one non-root service; model credentials never enter the container."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", required=True, help="one physical GPU index or UUID")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dds-domain", type=int, default=21)
    parser.add_argument("--accept-nvidia-eula", action="store_true", required=True)
    args = parser.parse_args()
    validate_image_reference(args.image)
    if not re.fullmatch(r"(?:[0-9]+|GPU-[0-9a-fA-F-]+)", args.gpu):
        parser.error("select exactly one GPU index or UUID")
    if not 1 <= args.port <= 65535 or not 0 <= args.dds_domain <= 232:
        parser.error("invalid port or DDS domain")
    catalog, assets = args.catalog.resolve(), args.assets.resolve()
    if not (catalog / "catalog.json").is_file() or not assets.is_dir():
        parser.error("materialized catalog and assets directories are required")
    output = args.output.resolve()
    for path in (catalog, assets, output):
        if any(c in str(path) for c in (",", "\n", "\r")):
            parser.error("mount paths cannot contain commas or newlines")
    output.mkdir(parents=True, exist_ok=False)
    (output / "home").mkdir()
    (output / "tmp").mkdir()
    name = "simple-service-" + uuid.uuid4().hex
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--read-only",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=4g",
        "--runtime=nvidia",
        "--gpus",
        "device=" + args.gpu,
        "--shm-size=4g",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-p",
        f"127.0.0.1:{args.port}:8000",
    ]
    for key, value in {
        "OMNI_KIT_ACCEPT_EULA": "Y",
        "HOME": "/output/home",
        "XDG_CACHE_HOME": "/output/home/.cache",
        "TMPDIR": "/output/tmp",
        "ROBORSI_SIMPLE_GPU_INDEX": "0",
        "MUJOCO_EGL_DEVICE_ID": "0",
        "ROBORSI_SIMPLE_DDS_DOMAIN": str(args.dds_domain),
    }.items():
        command += ["-e", key + "=" + value]
    if os.environ.get("ROBORSI_BENCHMARK_TOKEN"):
        command += ["-e", "ROBORSI_BENCHMARK_TOKEN"]
    for source, target, readonly in (
        (catalog, "/roborsi/assets/simple-catalog", True),
        (assets, "/roborsi/assets/simple-assets", True),
        (output, "/output", False),
    ):
        command += [
            "--mount",
            f"type=bind,src={source},dst={target}" + (",readonly" if readonly else ""),
        ]
    command.append(args.image)
    try:
        result = subprocess.run(command, check=False)
        if result.returncode:
            raise SystemExit(result.returncode)
    finally:
        subprocess.run(
            ["docker", "stop", "--time", "30", name],
            capture_output=True,
            timeout=60,
            check=False,
        )


if __name__ == "__main__":
    main()
