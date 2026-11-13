# Copyright 2026 The RoboRSI Contributors.
# Licensed under the Apache License, Version 2.0.

"""Check the original policy namespace without running a model or simulator."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path

from codex_cli_bin import bundled_codex_path

from robots.bench2dex.compat_agent import CLI_SHA256
from robots.bench2dex.contracts import ROOT, file_sha256
from robots.bench2dex.frozen_reference import load_reference
from robots.bench2dex.icl import PROTOCOL


def check() -> dict:
    """Fail explicitly on undeclared policy runtimes, without reading credentials."""
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("historical policy runtime requires Linux x86_64")
    binary = Path(os.environ.get("BENCH2DEX_CODEX_BIN") or bundled_codex_path())
    digest = file_sha256(binary)
    version = subprocess.check_output([str(binary), "--version"], text=True).strip()
    if digest != CLI_SHA256 or version != PROTOCOL["codex_version"]:
        raise ValueError("historical CLI executable differs")
    ref = load_reference(ROOT.parents[1] / "agents/bench2dex_compat/reference-code.zip")
    expected = {
        "numpy": "1.21.5",
        "scipy": "1.8.0",
        "Pillow": "9.0.1",
        "PyYAML": "5.4.1",
    }
    with tempfile.TemporaryDirectory(prefix="bench2dex-policy-check-") as tmp:
        root = Path(tmp)
        for name in (
            "policy/input",
            "private/codex-home",
            "private/interface",
            "private/action-transport",
        ):
            (root / name).mkdir(parents=True, exist_ok=True)
        command = ref.bridge.isolation.mounts(
            root / "policy",
            root / "private/codex-home",
            root / "private/interface",
            binary,
            numerical_runtime=True,
        )
        command += [
            "--",
            "/usr/bin/python3",
            "-c",
            "import json,numpy,scipy,PIL,yaml; import importlib.metadata as m; print(json.dumps({n:m.version(n) for n in ['numpy','scipy','Pillow','PyYAML']}))",
        ]
        packages = json.loads(subprocess.check_output(command, text=True))
        if packages != expected:
            raise ValueError(
                f"policy system packages differ: {packages}; expected {expected}"
            )
    return {
        "status": "passed",
        "cli_version": version,
        "cli_sha256": digest,
        "system_policy_packages": packages,
    }


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
