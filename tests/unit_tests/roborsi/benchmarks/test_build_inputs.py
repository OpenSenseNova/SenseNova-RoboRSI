# Copyright 2026 The RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Declared build materials are copied, pinned and kept visible to Docker."""

import hashlib
import json
import shutil

import pytest

from benchmarks.package import prepare_context
from roborsi.benchmarks.runtime_source import benchmark_source_digest
from tests.unit_tests.roborsi.benchmarks.test_runtime_package import ROOT


def checkout(tmp_path):
    context = prepare_context("libero", tmp_path / "initial", source_root=ROOT)
    root = context / "benchmark-runtime"
    shutil.copytree(
        ROOT / "benchmarks/libero", root / "benchmarks/libero", dirs_exist_ok=True
    )
    return root, root / "benchmarks/libero"


def test_build_inputs_copy_directories_globs_modes_and_track_changes(tmp_path):
    root, directory = checkout(tmp_path)
    (directory / "patches/nested").mkdir(parents=True)
    (directory / "patches/nested/native.patch").write_text("patch")
    script = directory / "prepare.sh"
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(0o755)
    (directory / "unused.secret").write_text("do not copy")
    (directory / "build-inputs.json").write_text(
        json.dumps(
            {"schema_version": 1, "inputs": ["requirements.txt", "patches", "*.sh"]}
        )
    )
    version = benchmark_source_digest(root, "libero")
    exported = prepare_context("libero", tmp_path / "export", source_root=root)
    assert not (exported / "unused.secret").exists()
    assert (exported / "prepare.sh").stat().st_mode & 0o111
    assert (exported / "patches/nested/native.patch").read_text() == "patch"
    assert (
        "!patches/nested/native.patch"
        in (exported / "Dockerfile.dockerignore").read_text()
    )
    record = json.loads((exported / "build.json").read_text())
    for name, info in record["files"].items():
        assert (
            info["sha256"] == hashlib.sha256((exported / name).read_bytes()).hexdigest()
        )
    script.write_text("#!/bin/sh\nexit 1\n")
    other = prepare_context("libero", tmp_path / "other", source_root=root)
    assert (
        json.loads((other / "build.json").read_text())["context_sha256"]
        != record["context_sha256"]
    )
    assert benchmark_source_digest(root, "libero") == version


@pytest.mark.parametrize(
    "pattern", ["../outside", "/absolute", "missing*.patch", "source.json"]
)
def test_invalid_inputs_fail_without_partial_export(tmp_path, pattern):
    root, directory = checkout(tmp_path)
    (directory / "source.json").write_text("{}")
    (directory / "build-inputs.json").write_text(
        json.dumps({"schema_version": 1, "inputs": [pattern]})
    )
    with pytest.raises(ValueError):
        prepare_context("libero", tmp_path / "export", source_root=root)
    assert not (tmp_path / "export").exists()


def test_symlink_and_existing_destination_are_rejected(tmp_path):
    root, directory = checkout(tmp_path)
    (directory / "linked").symlink_to(directory / "requirements.txt")
    (directory / "build-inputs.json").write_text(
        json.dumps({"schema_version": 1, "inputs": ["linked"]})
    )
    with pytest.raises(ValueError, match="symlink"):
        prepare_context("libero", tmp_path / "export", source_root=root)
    (directory / "build-inputs.json").write_text(
        json.dumps({"schema_version": 1, "inputs": []})
    )
    target = tmp_path / "existing"
    target.mkdir()
    (target / "keep").write_text("keep")
    with pytest.raises(FileExistsError):
        prepare_context("libero", target, source_root=root)
    assert (target / "keep").read_text() == "keep"
