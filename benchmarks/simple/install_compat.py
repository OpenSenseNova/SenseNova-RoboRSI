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

"""Fetch unmodified, hash-pinned upstream task files and their licenses."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path


def install(sources: Path, output: Path) -> None:
    """Install original sources outside the native checkout; never execute them."""
    specs = json.loads(sources.read_text())["compatibility"]
    output.mkdir(parents=True, exist_ok=False)
    try:
        for alias, spec in specs.items():
            for path_key, hash_key, filename in (
                ("path", "sha256", Path(spec["path"]).name),
                ("license_path", "license_sha256", alias + "-LICENSE"),
            ):
                url = (
                    "https://raw.githubusercontent.com/physical-superintelligence-lab/"
                    f"SIMPLE/{spec['revision']}/{spec[path_key]}"
                )
                with urllib.request.urlopen(url, timeout=60) as response:
                    data = response.read(1024 * 1024)
                if hashlib.sha256(data).hexdigest() != spec[hash_key]:
                    raise ValueError(f"upstream compatibility hash differs: {alias}")
                (output / filename).write_bytes(data)
        (output / "provenance.json").write_text(json.dumps(specs, indent=2) + "\n")
    except BaseException:
        shutil.rmtree(output)
        raise


def main() -> None:
    """Install only the declared public compatibility sources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    install(args.sources, args.output)


if __name__ == "__main__":
    main()
