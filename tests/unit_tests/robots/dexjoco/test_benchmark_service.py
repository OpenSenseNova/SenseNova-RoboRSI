# Copyright 2026 The SenseNova-RoboRSI Authors.
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

"""Offline discovery and manifest checks using task configuration fixtures."""

import json
import sys

import pytest
import yaml
from jsonschema import Draft202012Validator

from benchmarks.dexjoco.manifest import build_manifest, main
from roborsi.benchmarks.registry import benchmark_manifest_schema
from robots.dexjoco.benchmark_service import DexJocoBenchmark
from robots.dexjoco.contracts import CAPABILITIES, COMMIT, EVALUATION, LIMITS, SEEDS


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    configs = tmp_path / "upstream" / "configs" / "rand_full"
    configs.mkdir(parents=True)
    for name in LIMITS:
        (configs / f"{name}.yaml").write_text(
            yaml.safe_dump(
                {
                    "env_name": name,
                    "robot_type": (
                        "dual_arm" if name.startswith("bimanual") else "single_arm"
                    ),
                    "camera_mapping": {"base": "ego"},
                    "prompt": "Complete " + name,
                }
            )
        )
    monkeypatch.setattr(
        "robots.dexjoco.benchmark_service.verify_upstream", lambda root: None
    )
    return configs.parents[1]


@pytest.mark.parametrize("multiplier", [1, 10])
def test_task_discovery_retains_cohort_and_physical_budgets(catalog, multiplier):
    provider = DexJocoBenchmark(catalog, multiplier=multiplier)
    tasks = provider.tasks()
    assert len(tasks) == 55
    assert {(task.task_id, task.seed) for task in tasks} == {
        (name, seed) for name in LIMITS for seed in SEEDS
    }
    for task in tasks:
        assert task.benchmark == "dexjoco"
        assert task.instruction == "Complete " + task.task_id
        assert task.max_steps == LIMITS[task.task_id] * multiplier
        assert task.native["regime"] == "rand_full"
        assert task.native["multiplier"] == multiplier
        assert task.capabilities == CAPABILITIES
    assert provider.supports_episode_stop
    assert provider.assets() == []


def test_manifest_describes_actual_discovery_and_image(catalog):
    provider = DexJocoBenchmark(catalog)
    repository = "registry.example.invalid/benchmarks/dexjoco"
    digest = "sha256:" + "a" * 64
    manifest = build_manifest(provider, repository + "@" + digest)
    Draft202012Validator(benchmark_manifest_schema()).validate(manifest)
    assert manifest["benchmark_commit"] == COMMIT
    assert manifest["image_repository"] == repository
    assert manifest["image_digest"] == digest
    assert manifest["evaluation"] == EVALUATION
    assert manifest["assets"] == []
    assert manifest["resources"]["gpus"] == 0
    assert len(manifest["tasks"]) == 11
    assert sum(len(task["seeds"]) for task in manifest["tasks"]) == 55
    discovery = {task.task_id: task for task in provider.tasks()}
    for task in manifest["tasks"]:
        observed = discovery[task["task_id"]]
        assert task["seeds"] == list(SEEDS)
        assert task["max_steps"] == observed.max_steps
        assert task["instruction"] == observed.instruction
        assert task["native"] == observed.native


def test_manifest_requires_immutable_image_reference(catalog):
    with pytest.raises(ValueError, match="actual repository@sha256:digest"):
        build_manifest(DexJocoBenchmark(catalog), "registry.example.invalid/dexjoco:v1")


@pytest.mark.parametrize("existing_parent", [False, True])
def test_manifest_cli_creates_output_parents(
    catalog, tmp_path, monkeypatch, existing_parent
):
    output = tmp_path / "new registry" / "dexjoco" / "manifest.json"
    if existing_parent:
        output.parent.mkdir(parents=True)
    image = "registry.example.invalid/dexjoco@sha256:" + "a" * 64
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "manifest",
            "--upstream",
            str(catalog),
            "--image",
            image,
            "--output",
            str(output),
        ],
    )

    main()

    document = json.loads(output.read_text())
    Draft202012Validator(benchmark_manifest_schema()).validate(document)
    assert len(document["tasks"]) == 11
    assert sum(len(task["seeds"]) for task in document["tasks"]) == 55
    assert not list(output.parent.glob(".*.tmp"))
