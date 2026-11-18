# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0.

"""Record native dependency identities and reject model SDKs/credential files."""

import importlib.metadata
import json
import re
import subprocess
from pathlib import Path

forbidden = {
    "openai",
    "anthropic",
    "pydantic-ai",
    "pydantic-ai-slim",
    "roborsi",
    "heart-runtime",
    "mcp",
    "claude-agent-sdk",
    "openai-codex",
}
installed = {
    dist.metadata["Name"].lower().replace("_", "-"): dist.version
    for dist in importlib.metadata.distributions()
}
overlap = forbidden.intersection(installed)
if overlap:
    raise RuntimeError(f"Agent dependencies entered the native base: {sorted(overlap)}")
for name in (".ssh", ".aws", ".config/gh", ".docker/config.json", ".netrc"):
    if (Path("/root") / name).exists():
        raise RuntimeError(f"Unexpected credential location in native base: {name}")
patterns = (
    rb"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}",
    rb"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})",
    rb"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----",
)
text_suffixes = {".py", ".md", ".json", ".yaml", ".yml", ".txt", ".toml", ".sh"}
scanned = 0
for root in (Path("/opt/simple"), Path("/opt/simple-build")):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in text_suffixes and ".git" not in path.parts:
            data = path.read_bytes()
            scanned += 1
            if any(re.search(pattern, data) for pattern in patterns):
                raise RuntimeError(
                    f"Credential pattern found; inspect privately: {path}"
                )
sources = {}
for root in [
    Path("/opt/simple"),
    *Path("/opt/simple/third_party").iterdir(),
    Path("/opt/cyclonedds-src"),
]:
    if (root / ".git").is_dir():
        sources[str(root)] = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
Path("/opt/simple-build/native-manifest.json").write_text(
    json.dumps(
        {"sources": sources, "distributions": installed, "text_files_scanned": scanned},
        indent=2,
    )
    + "\n"
)
