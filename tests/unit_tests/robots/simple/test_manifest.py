# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Registry interoperability and materialized-asset integrity checks."""

import json

import pytest

from benchmarks.simple.manifest import build_manifest
from roborsi.benchmarks.registry import BenchmarkRegistry
from robots.simple.benchmark_service import SimpleBenchmark
from tests.unit_tests.robots.simple.test_simple import catalog


def test_manifest_roundtrip_preserves_case_and_rejects_asset_drift(
    tmp_path, monkeypatch
):
    root = tmp_path / "simple"
    path, _ = catalog(root)
    data = json.loads(path.read_text())
    data["cases"][0]["task"]["capabilities"].pop("state_dim")
    path.write_text(json.dumps(data))
    assets = root / "assets"
    assets.mkdir()
    (assets / "fixture").write_text("offline fixture")
    monkeypatch.setattr(
        "robots.simple.benchmark_service.benchmark_source_digest",
        lambda *_: "a" * 64,
    )
    provider = SimpleBenchmark(path)
    image = "example.invalid/simple@sha256:" + "b" * 64
    output = root / "manifest.json"
    document = build_manifest(
        image, path.parent, assets, output, provider.runtime_info().as_dict()
    )
    output.write_text(json.dumps(document))
    lock = BenchmarkRegistry([tmp_path]).lock("simple")
    original = provider.tasks()[0]
    assert lock.task_message(original.task_id, original.seed) == original
    assert lock.image_reference == image
    (assets / "fixture").write_text("modified")
    with pytest.raises(ValueError, match="digest mismatch"):
        lock.verify()
    with pytest.raises(ValueError, match="pinned"):
        build_manifest(
            "example.invalid/simple:latest",
            path.parent,
            assets,
            output,
            provider.runtime_info().as_dict(),
        )
