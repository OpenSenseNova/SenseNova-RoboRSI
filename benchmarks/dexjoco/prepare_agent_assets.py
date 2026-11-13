# Copyright 2026 The HEART Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Verify frozen knowledge and construct the original public robot references."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from robots.dexjoco.contracts import COMMIT, LIMITS
from robots.dexjoco.native import verify_upstream
from robots.dexjoco.reference import load_reference


def prepare(upstream: Path, knowledge: Path, archive: Path, output: Path) -> dict:
    """Prepare a new portable Agent asset directory without raw demonstrations."""
    verify_upstream(upstream)
    ref = load_reference(archive)
    cfg = ref.load_config(
        ref.ROOT / "configs/dexjoco_icl_5ep_rand_full_robot_v2_two_session.json"
    )
    catalog = json.loads((ref.ROOT / "provenance/knowledge_packages.json").read_text())[
        "packages"
    ]
    for task in LIMITS:
        ref.verify_package(
            knowledge / "rand_full" / task, task, cfg, catalog[task]["manifest_sha256"]
        )
    output.mkdir(parents=True, exist_ok=False)
    try:
        for task in LIMITS:
            destination = output / "rand_full" / task
            shutil.copytree(knowledge / "rand_full" / task, destination)
            ref.verify_package(destination, task, cfg, catalog[task]["manifest_sha256"])
        for robot in ("single_arm", "dual_arm"):
            ref.copy_robot_reference(
                upstream, output / "robot_reference" / robot, robot, COMMIT
            )
        inventory = ref.tree_hashes(output)
        ref.write_json(
            output / "assets.json", {"schema_version": 1, "files": inventory}
        )
        return {
            "files": len(inventory),
            "manifest_sha256": ref.file_hash(output / "assets.json"),
        }
    except BaseException:
        shutil.rmtree(output)
        raise


def main() -> None:
    """Build assets from an offline copy of the public frozen knowledge release."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument(
        "--knowledge", type=Path, required=True, help="Directory containing rand_full/"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("robots/dexjoco/agents/baseline/resources/reference-code.zip"),
    )
    args = parser.parse_args()
    prepare(
        args.upstream.resolve(),
        args.knowledge.resolve(),
        args.reference.resolve(),
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()
