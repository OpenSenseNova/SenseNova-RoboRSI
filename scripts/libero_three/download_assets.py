# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Download and verify every non-model asset for the three LIBERO evaluations."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

_REPOSITORY = Path(__file__).resolve().parents[2]
if str(_REPOSITORY) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY))

from benchmarks.libero_pro.assets import (  # noqa: E402
    DATASET_REPOSITORY,
    DATASET_REVISION,
    REQUIRED_SUITES,
    prepare_asset,
    stage_image_asset,
)
from roborsi.evaluation.result import write_json_atomic  # noqa: E402
from scripts.libero_three.prepare_agent_assets import (  # noqa: E402
    prepare as prepare_agent_assets,
)

ICL_REPOSITORY = "sensenova/SenseNova-RoboRSI-Packages"
ICL_REVISION = "e977b5ad667b4ce753ee947cbb0b95d4fc0fe98c"
ICL_PATHS = {
    "longpp": "libero-long-plus-plus/v1.0.0/packages",
    "ultra": "libero-ultra-long/v1.0.0/packages",
    "pro": "libero-pro/v1.0.0/packages",
}
TOKENIZER_REPOSITORY = "Qwen/Qwen3.8-Flash-Next"
TOKENIZER_REVISION = "34567a4712bc9766c4449e2e98e4468bfa24d915"
TOKENIZER_FILE = "tokenizer.json"


def download(
    destination: Path,
    *,
    repository: Path,
    pro_build_context: Path | None = None,
    local_files_only: bool = False,
) -> Path:
    """Fetch pinned public inputs and emit verified image-build directories."""
    destination = destination.expanduser().resolve()
    repository = repository.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    try:
        icl_snapshot = Path(
            snapshot_download(
                repo_id=ICL_REPOSITORY,
                repo_type="dataset",
                revision=ICL_REVISION,
                allow_patterns=[f"{path}/**" for path in ICL_PATHS.values()],
                token=False,
                local_files_only=local_files_only,
            )
        )
        pro_snapshot = Path(
            snapshot_download(
                repo_id=DATASET_REPOSITORY,
                repo_type="dataset",
                revision=DATASET_REVISION,
                allow_patterns=[
                    f"{kind}/{suite}/**"
                    for kind in ("bddl_files", "init_files")
                    for suite in REQUIRED_SUITES
                ],
                token=False,
                local_files_only=local_files_only,
            )
        )
        tokenizer = Path(
            hf_hub_download(
                repo_id=TOKENIZER_REPOSITORY,
                filename=TOKENIZER_FILE,
                revision=TOKENIZER_REVISION,
                token=False,
                local_files_only=local_files_only,
            )
        )
        # Hub cache snapshots intentionally use symlinks. Context packages
        # reject links, so materialize regular files before verification and
        # before copying the sealed payload into the Docker input directory.
        with tempfile.TemporaryDirectory(prefix="roborsi-public-assets-") as raw:
            materialized = Path(raw)
            icl_roots = {}
            for name, path in ICL_PATHS.items():
                target = materialized / name / "packages"
                shutil.copytree(icl_snapshot / path, target)
                icl_roots[name] = target
            tokenizer_file = materialized / TOKENIZER_FILE
            shutil.copy2(tokenizer, tokenizer_file)
            agent_assets = prepare_agent_assets(
                icl_roots,
                tokenizer_file,
                destination / "agent-assets",
                repository=repository,
            )
        pro_descriptor = prepare_asset(pro_snapshot, destination / "pro-assets")
        if pro_build_context is not None:
            stage_image_asset(
                destination / "pro-assets" / pro_descriptor["path"],
                pro_build_context,
            )
        write_json_atomic(
            destination / "sources.json",
            {
                "schema_version": 1,
                "icl": {
                    "repository": ICL_REPOSITORY,
                    "revision": ICL_REVISION,
                    "paths": ICL_PATHS,
                },
                "libero_pro": {
                    "repository": DATASET_REPOSITORY,
                    "revision": DATASET_REVISION,
                    "asset": pro_descriptor,
                },
                "tokenizer": {
                    "repository": TOKENIZER_REPOSITORY,
                    "revision": TOKENIZER_REVISION,
                    "path": TOKENIZER_FILE,
                },
                "agent_assets": agent_assets.relative_to(destination).as_posix(),
            },
        )
        return destination
    except BaseException:
        shutil.rmtree(destination)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pro-build-context", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    try:
        output = download(
            args.output,
            repository=_REPOSITORY,
            pro_build_context=args.pro_build_context,
            local_files_only=args.local_files_only,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(output)


if __name__ == "__main__":
    main()
