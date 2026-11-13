# Copyright 2026 The RoboRSI Contributors.
# Licensed under the Apache License, Version 2.0.

"""Load the unchanged evaluator under a private module namespace."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import stat
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

REFERENCE_SHA256 = "cfe38f4cdf051df401dc85f4eb33d79c77ce87bcd618e9ed2121f8bb679b7371"
_temporary = None
_reference = None


def load_reference(archive: Path) -> SimpleNamespace:
    """Verify the frozen bytes before importing code or reading prompts."""
    global _temporary, _reference
    if hashlib.sha256(archive.read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise ValueError("historical evaluator archive hash mismatch")
    if _reference is not None:
        return _reference
    _temporary = tempfile.TemporaryDirectory(prefix="bench2dex-reference-")
    root = Path(_temporary.name)
    with zipfile.ZipFile(archive) as bundle:
        seen = set()
        for member in bundle.infolist():
            name = member.filename.rstrip("/")
            parts = PurePosixPath(name).parts
            if (
                not parts
                or parts[0] != "bench2dex-eval"
                or any(p in {"", ".", ".."} for p in name.split("/"))
                or "\\" in name
                or name in seen
                or stat.S_IFMT(member.external_attr >> 16)
                not in {0, stat.S_IFREG, stat.S_IFDIR}
            ):
                raise ValueError("unsafe reference archive member")
            seen.add(name)
        bundle.extractall(root)
    source = root / "bench2dex-eval"
    name = "_bench2dex_frozen_eval"
    if name in sys.modules:
        raise RuntimeError("historical evaluator namespace is already in use")
    spec = importlib.util.spec_from_file_location(
        name,
        source / "src/gpt6_eval/__init__.py",
        submodule_search_locations=[str(source / "src/gpt6_eval")],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    _reference = SimpleNamespace(
        root=source,
        bridge=importlib.import_module(name + ".bridge"),
        core=importlib.import_module(name + ".core"),
        knowledge=importlib.import_module(name + ".knowledge"),
        throughput=importlib.import_module(name + ".throughput"),
        calibration=importlib.import_module(name + ".calibration"),
    )
    return _reference
