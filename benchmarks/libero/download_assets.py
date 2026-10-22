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

"""Download a fixed LIBERO asset revision and publish a Registry snapshot."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from roborsi.benchmarks.registry import snapshot_directory
from roborsi.evaluation.result import write_json_atomic

ASSETS_REPO = "RLinf/LIBERO-assets"
ASSETS_REVISION = "3ba78404b48e8c70fa4ac9782d5aac8e5b46d55f"


def main() -> None:
    """Write asset.json and a read-only directory under the requested output."""
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default=ASSETS_REPO)
    parser.add_argument("--revision", default=ASSETS_REVISION)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="download-", dir=root) as temporary:
        source = Path(temporary)
        snapshot_download(
            args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            local_dir=source,
            token=False,
        )
        # Hub bookkeeping is not part of the simulator assets.
        shutil.rmtree(source / ".cache", ignore_errors=True)
        snapshot = snapshot_directory(source, root / "snapshots")
    asset = {
        "name": "libero",
        "kind": "directory",
        "version": args.revision,
        "path": snapshot.relative_to(root).as_posix(),
        "sha256": snapshot.name,
    }
    write_json_atomic(root / "asset.json", asset)
    print(json.dumps(asset))


if __name__ == "__main__":
    main()
