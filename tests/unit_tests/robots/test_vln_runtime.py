# Copyright 2026 The RoboRSI Contributors.
# Licensed under the Apache License, Version 2.0.

"""Exported navigation services retain the manifest's adapter identity."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.package import prepare_context
from roborsi.benchmarks.runtime_source import (
    benchmark_source_digest,
    benchmark_source_files,
)
from robots.vln_shared import benchmark, campaign
from robots.vln_shared.assets import manifest
from robots.vln_shared.storage import write_json
from tests.unit_tests.roborsi.benchmarks.test_runtime_package import _IMPORT_GUARD

ROOT = Path(__file__).resolve().parents[3]


def catalog(dataset):
    """Use a public task catalog without loading simulator assets or a model."""
    return {
        "spec": {
            "dataset": dataset,
            "python": "/opt/native/bin/python",
            "native_commit": "a" * 40,
            "seed": 42,
            "split": "test",
            "inputs": {},
        },
        "episodes": [
            {
                "id": "episode-7",
                "episode_id": "7",
                "episode_index": 3,
                "instruction": "Navigate to the door.",
            }
        ],
    }


@pytest.mark.parametrize("dataset", ["r2r", "rxr", "lhpr_vln"])
def test_exported_service_matches_manifest_without_agent_imports(tmp_path, dataset):
    name = "lhpr_vln" if dataset == "lhpr_vln" else "vlnce"
    context = prepare_context(name, tmp_path / "context", source_root=ROOT)
    runtime = context / "benchmark-runtime"
    assert {
        path.relative_to(runtime).as_posix()
        for path in runtime.rglob("*")
        if path.is_file()
    } == set(benchmark_source_files(name))
    assets = tmp_path / "assets"
    write_json(assets / "catalog.json", catalog(dataset))
    document = manifest(
        assets, tmp_path / "manifest.json", f"example/{name}@sha256:" + "b" * 64
    )
    identity = document["adapter_version"]
    assert identity == benchmark_source_digest(ROOT, name)
    assert json.loads((context / "source.json").read_text())["sha256"] == identity
    code = (
        _IMPORT_GUARD
        + r"""
import json, os
from pathlib import Path
blocked += ('habitat', 'habitat_sim', 'torch',
            'robots.vln_shared.cli', 'robots.vln_shared.campaign',
            'robots.vln_shared.execution', 'robots.vln_shared.exploration',
            'robots.vln_shared.agents')
name, assets, expected = sys.argv[2:]
os.environ['ROBORSI_VLN_CATALOG'] = str(Path(assets) / 'catalog.json')
os.environ['ROBORSI_VLN_BUDGET'] = '500'
module = importlib.import_module('robots.' + name + '.benchmark_service')
importlib.import_module('robots.' + name + '.native')
importlib.import_module('robots.vln_shared.native_worker')
provider = module.make_benchmark()
assert provider.runtime_info().adapter_version == expected
assert provider.tasks()[0].max_steps == 500
assert provider.tasks()[0].seed == 42
assert provider.assets()[0]['name'] == name
assert not any(key in sys.modules for key in blocked)
for key, loaded in tuple(sys.modules.items()):
    if key.split('.')[0] in ('roborsi', 'robots') and getattr(loaded, '__file__', None):
        assert Path(loaded.__file__).is_relative_to(sys.argv[1]), (key, loaded.__file__)
from roborsi.evaluation.service_host import main
main(['--kind', 'benchmark', '--factory',
      'robots.' + name + '.benchmark_service:make_benchmark', '--help'])
"""
    )
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            code,
            str(runtime),
            name,
            str(assets),
            identity,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--factory" in result.stdout and "--kind" in result.stdout


@pytest.mark.parametrize("dataset", ["r2r", "rxr", "lhpr_vln"])
def test_service_and_campaign_identities_track_their_own_sources(
    tmp_path, monkeypatch, dataset
):
    name = "lhpr_vln" if dataset == "lhpr_vln" else "vlnce"
    # Use real exported code plus the complete navigation implementation, without
    # changing the user's source checkout or relying on simulator dependencies.
    root = (
        prepare_context(name, tmp_path / "context", source_root=ROOT)
        / "benchmark-runtime"
    )
    for directory in ("vln_shared", "vlnce", "lhpr_vln"):
        shutil.copytree(
            ROOT / "robots" / directory,
            root / "robots" / directory,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    monkeypatch.setattr(
        benchmark, "__file__", str(root / "robots/vln_shared/benchmark.py")
    )
    monkeypatch.setattr(
        campaign, "__file__", str(root / "robots/vln_shared/campaign.py")
    )
    section = catalog(dataset)
    provider = benchmark.NavigationBenchmark(section["spec"], section["episodes"], 500)
    service_before = provider.runtime_info().adapter_version
    campaign_before = campaign.campaign_source_identity()
    for module in ("cli", "campaign", "execution", "exploration"):
        path = root / f"robots/vln_shared/{module}.py"
        original = path.read_text()
        path.write_text(original + "\n# Agent orchestration changed.\n")
        assert provider.runtime_info().adapter_version == service_before
        assert campaign.campaign_source_identity() != campaign_before
        path.write_text(original)

    # The other Benchmark must not change this service's version.
    other = "vlnce" if name == "lhpr_vln" else "lhpr_vln"
    path = root / f"robots/{other}/native.py"
    path.write_text(path.read_text() + "\n# Other native adapter changed.\n")
    assert provider.runtime_info().adapter_version == service_before
    assert campaign.campaign_source_identity() != campaign_before

    for module in (
        f"robots/{name}/native.py",
        "robots/vln_shared/native_worker.py",
        "roborsi/evaluation/benchmark_http.py",
    ):
        path = root / module
        original = path.read_text()
        path.write_text(original + "\n# Service source changed.\n")
        assert provider.runtime_info().adapter_version != service_before
        path.write_text(original)
