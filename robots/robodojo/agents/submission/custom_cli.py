"""Explicit, recorded native CLI selection for operator-owned experiments."""

import json
import os
import re
from pathlib import Path


def selected_cli(policy_dir, release):
    manifest = os.environ.get("ROBORSI_CUSTOM_CODEX_MANIFEST")
    if not manifest:
        return {**release, "executable": str(Path(policy_dir) / "runtime/codex/bin/codex"),
                "selection": "release"}
    path = Path(manifest).expanduser()
    if not path.is_absolute():
        raise ValueError("ROBORSI_CUSTOM_CODEX_MANIFEST must be absolute")
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1 or data.get("selection") != "operator_custom_build":
        raise ValueError("Unsupported custom CLI identity manifest")
    executable = Path(data.get("executable", ""))
    if not executable.is_absolute() or not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("Custom CLI must select an executable absolute native binary")
    with executable.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            raise ValueError("Custom CLI must select ELF, not a shell wrapper")
    if not re.fullmatch(r"codex-cli \d+\.\d+\.\d+", data.get("expected_version", "")):
        raise ValueError("Custom CLI requires an exact version")
    if data.get("expected_version") != "codex-cli " + str(data.get("version", "")):
        raise ValueError("Custom CLI version fields disagree")
    if not re.fullmatch(r"[a-f0-9]{64}", data.get("sha256", "")):
        raise ValueError("Custom CLI requires a SHA-256 pin")
    companions = data.get("companion_sha256", {})
    if set(companions) != {"bin/codex-code-mode-host", "codex-resources/bwrap"} or any(
        not re.fullmatch(r"[a-f0-9]{64}", str(value)) for value in companions.values()
    ):
        raise ValueError("Custom CLI requires companion SHA-256 pins")
    return {**data, "manifest": str(path), "executable": str(executable.resolve())}
