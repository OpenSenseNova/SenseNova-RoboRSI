# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Fail closed unless an installed three-LIBERO Agent image is self-contained."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from roborsi.agents.config import AgentConfig
from roborsi.agents.context_package import load_package_map
from roborsi.benchmarks.registry import directory_sha256
from roborsi.utils.config import require_project_env

_SPECS = {
    "libero_long_plus_plus": (
        "bundle:libero_long_plus_plus_v13",
        "ROBORSI_LIBERO_LONGPP_ICL_ROOT",
        12,
        12,
    ),
    "libero_ultra_long": (
        "bundle:libero_ultra_long_v13",
        "ROBORSI_LIBERO_ULTRA_ICL_ROOT",
        9,
        9,
    ),
    "libero_pro": ("bundle:libero_pro", "ROBORSI_LIBERO_PRO_ICL_ROOT", 80, 40),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(lock_path: Path) -> dict:
    """Verify installed bundles, every package payload and the local tokenizer."""
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    tokenizer = Path(require_project_env("ROBORSI_CONTEXT_TOKENIZER_PATH"))
    tokenizer_digest = _sha256(tokenizer)
    expected_tokenizer = lock["assets"]["context_tokenizer"]["sha256"]
    if tokenizer_digest != expected_tokenizer:
        raise ValueError("Agent image tokenizer differs from assets.lock.json")
    result = {}
    for benchmark, (bundle, environment, cases, packages) in _SPECS.items():
        config = AgentConfig.load(bundle)
        if config.parameters.get("enable_vla") or config.parameters.get("enable_sam3"):
            raise ValueError(f"{bundle} unexpectedly enables a model tool")
        root = Path(os.environ[environment])
        loaded = load_package_map(config.resources["context_map"], root)
        unique = {package.root for package in loaded.values()}
        if len(loaded) != cases or len(unique) != packages:
            raise ValueError(f"{bundle} has incomplete ICL mappings")
        if {path.name for path in root.iterdir()} != {path.name for path in unique}:
            raise ValueError(f"{bundle} contains undeclared ICL packages")
        digest = directory_sha256(root)
        expected = lock["assets"][f"{benchmark}_icl"]["snapshot_sha256"]
        if digest != expected:
            raise ValueError(f"{bundle} ICL snapshot differs from assets.lock.json")
        context = config.runtime().context
        if (
            context is None
            or context.tokenizer_file != str(tokenizer)
            or context.tokenizer_sha256 != tokenizer_digest
        ):
            raise ValueError(f"{bundle} does not lock the embedded tokenizer")
        result[benchmark] = {
            "cases": len(loaded),
            "packages": len(unique),
            "sha256": digest,
        }
    return {"packages": result, "tokenizer_sha256": tokenizer_digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.lock), sort_keys=True))


if __name__ == "__main__":
    main()
