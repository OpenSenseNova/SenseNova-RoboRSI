# Copyright 2026 The HEART Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Load the immutable, separately licensed historical evaluator snapshot."""

from __future__ import annotations

import hashlib
import importlib
import sys
import tempfile
import zipfile
from pathlib import Path

REFERENCE_SHA256 = "e883be2bf8f22674b24e56758721f754cce6353a45044f32025e26be7e870674"
_temporary = None
_module = None


def load_reference(archive: Path):
    """Verify before importing original prompts, validators and policy transport."""
    global _temporary, _module
    if hashlib.sha256(archive.read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise ValueError("Historical evaluator archive hash mismatch")
    if _module is not None:
        return _module
    if "gpt6_eval" in sys.modules:
        raise RuntimeError("Use a fresh Agent process without another gpt6_eval import")
    _temporary = tempfile.TemporaryDirectory(prefix="dexjoco-reference-")
    root = Path(_temporary.name)
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            if not (root / name).resolve().is_relative_to(root):
                raise ValueError("Reference archive path escapes its root")
        bundle.extractall(root)
    source = root / "dexjoco-eval"
    sys.path.insert(0, str(source / "src"))
    _module = importlib.import_module("gpt6_eval.dexjoco_eval")
    return _module
