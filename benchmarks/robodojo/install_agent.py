# Copyright 2026 The RoboRSI Contributors.
# Licensed under the Apache License, Version 2.0.

"""Install authenticated submission runtimes without contacting a model API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen

from benchmarks.robodojo.provenance import verify_submission

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "robots/robodojo/agents/submission"


def _verify_cli(root: Path, pins: dict) -> None:
    expected = {"bin/codex": pins["sha256"], **pins["companion_sha256"]}
    for name, digest in expected.items():
        path = root / name
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise ValueError("Codex artifact differs: " + name)
    result = subprocess.run(
        [str(root / "bin/codex"), "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.stdout.strip() != pins["expected_version"]:
        raise ValueError("Installed Codex version differs from submission")


def install_cli(archive: Path | None = None) -> dict:
    """Authenticate the native CLI plus both fixed companion binaries."""
    from robots.robodojo.agents.submission.cli_artifact import unpack_cli_archive
    from robots.robodojo.agents.submission.custom_cli import selected_cli

    metadata = json.loads((POLICY / "submission.json").read_text())
    cli = selected_cli(POLICY, metadata["cli"])
    runtime = POLICY / "runtime"
    runtime.mkdir(exist_ok=True)
    destination = runtime / "codex"
    if destination.exists():
        _verify_cli(destination, cli)
    elif cli["selection"] == "operator_custom_build":
        source = Path(cli["executable"]).parent.parent
        _verify_cli(source, cli)
        destination.symlink_to(source, target_is_directory=True)
    else:
        with tempfile.TemporaryDirectory(prefix="cli-", dir=runtime) as temporary:
            staging = Path(temporary)
            package = archive
            if package is None:
                package = staging / "release.whl"
                with (
                    urlopen(cli["download"]["url"], timeout=60) as response,
                    package.open("xb") as output,
                ):
                    shutil.copyfileobj(response, output)
            extracted = unpack_cli_archive(metadata, package, staging / "unpacked")
            _verify_cli(extracted, cli)
            extracted.rename(destination)
    return {"version": cli["expected_version"], "sha256": cli["sha256"]}


def resolve_agent_source(reference: str | Path) -> tuple[Path, Path]:
    """Locate a source or installed bundle's own declared submission payload."""
    from roborsi.agents.config import AgentConfig

    config = AgentConfig.load(reference)
    if config.factory != "robots.robodojo.agents.factory:make_agent":
        raise ValueError("Select a submitted RoboDojo Agent preset")
    return config.source_root, config.resources["submission_manifest"].parent


def install_simate(checkpoint_dir: Path | None) -> dict:
    """Use the same selected checkpoint for installation and verification."""
    from robots.robodojo.agents.submission.simate_runtime import check_simate_runtime

    environment = dict(os.environ)
    checkpoint = checkpoint_dir.expanduser().resolve() if checkpoint_dir else None
    if checkpoint is not None:
        environment["ROBORSI_SIM_CHECKPOINT"] = str(checkpoint)
    subprocess.run(
        ["bash", str(POLICY / "install_simate.sh")], check=True, env=environment
    )
    return check_simate_runtime(POLICY, checkpoint)


def main() -> None:
    global POLICY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent-config",
        default=str(ROOT / "robots/robodojo/agent.yaml"),
        help="source preset or installed bundle:robodojo_submission",
    )
    parser.add_argument(
        "--vla-backend",
        choices=("pi05", "simate_beta", "disabled"),
        default="simate_beta",
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument(
        "--cli-archive", type=Path, help="existing authenticated native release wheel"
    )
    parser.add_argument(
        "--check-input",
        action="store_true",
        help="verify source only; no downloads or model calls",
    )
    args = parser.parse_args()
    selected_root, POLICY = resolve_agent_source(args.agent_config)
    report = verify_submission(selected_root)
    from robots.robodojo.agents.submission.tool_install import (
        check_input,
        install_tools,
    )

    check_input(POLICY)
    if args.check_input:
        print(json.dumps(report, indent=2))
        return
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "amd64"}:
        parser.error("the submitted Codex/tool runtime requires Linux x86-64")
    if sys.version_info[:3] != (3, 10, 12):
        parser.error("use the submitted controller's Python 3.10.12 interpreter")
    report["cli"] = install_cli(args.cli_archive)
    report["tools"] = install_tools(POLICY)
    if args.vla_backend == "pi05":
        from robots.robodojo.agents.submission.pi05_runtime import install_pi05_runtime

        report["vla"] = install_pi05_runtime(
            policy_dir=POLICY, checkpoint_dir=args.checkpoint_dir
        )
    elif args.vla_backend == "simate_beta":
        report["vla"] = install_simate(args.checkpoint_dir)
    else:
        report["vla"] = {"backend": "disabled"}
    report["vla_backend"] = args.vla_backend
    (POLICY / "runtime/release-install.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
