# Copyright 2026 The RoboRSI Contributors.
# Licensed under the Apache License, Version 2.0.

"""Create a relocatable Agent source bundle from a verified frozen ICL archive."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import yaml

from roborsi.agents.config import AgentConfig
from robots.bench2dex.contracts import (
    ROOT,
    TASKS,
    task_for,
)
from robots.bench2dex.icl import extract_verified, package_entry, verify_package


def prepare(
    task: str,
    output: Path,
    *,
    source: Path | None = None,
    repo_id: str | None = None,
    revision: str = "main",
) -> Path:
    """Stage code and immutable resources under the same portable source root."""
    row = package_entry(task)
    if source is not None:
        archive = source / row["archive"]
    else:
        if not repo_id:
            raise ValueError("supply downloaded archives or a published ICL dataset ID")
        from huggingface_hub import hf_hub_download

        archive = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                filename=row["archive"],
            )
        )
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("choose a new output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    preset = AgentConfig.from_file(
        ROOT.parents[1] / "configs/agents/bench2dex/compat.yaml"
    )
    with tempfile.TemporaryDirectory(
        prefix=".bench2dex-agent-", dir=output.parent
    ) as temporary:
        staging = Path(temporary) / "source"
        staging.mkdir()
        for path in preset.input_files:
            target = staging / path.relative_to(preset.source_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        package = Path(temporary) / "verified-package"
        extract_verified(archive, package, row)
        verify_package(package, task_for(task))
        resource = staging / "resources/icl_package.zip"
        resource.parent.mkdir()
        shutil.copyfile(archive, resource)
        config = staging / preset.path.relative_to(preset.source_root)
        # Materialize inherited settings: editing only the thin alias loses its
        # parent's resource map, and relative paths must follow the leaf preset.
        document = preset.document
        document["resources"]["icl_package"] = os.path.relpath(resource, config.parent)
        config.write_text(yaml.safe_dump(document, sort_keys=False))
        AgentConfig.from_file(config).verify()
        shutil.copytree(staging, output)
    return output / preset.path.relative_to(preset.source_root)


def main() -> None:
    """Download only the chosen task; the checked-in catalog pins its bytes."""
    parser = argparse.ArgumentParser(description=__doc__)
    origin = parser.add_mutually_exclusive_group(required=True)
    origin.add_argument("--source", type=Path)
    origin.add_argument("--repo-id")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = prepare(
        args.task,
        args.output,
        source=args.source,
        repo_id=args.repo_id,
        revision=args.revision,
    )
    print(
        json.dumps(
            {
                "config": str(config),
                "archive_sha256": package_entry(args.task)["sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
