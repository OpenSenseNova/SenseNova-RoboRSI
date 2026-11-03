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

"""Asset relocation and usage isolation regressions for navigation delivery."""

import threading

import pytest

from roborsi.evaluation.messages import EvaluationResult, Provenance
from robots.vln_shared.assets import stage
from robots.vln_shared.benchmark import make_task
from robots.vln_shared.execution import BenchExecutor
from robots.vln_shared.storage import file_digest, read_json, write_json


def spec():
    return {
        "dataset": "r2r",
        "python": "unused",
        "seed": 42,
        "split": "test",
        "inputs": {},
        "native_commit": "a" * 40,
    }


def episode():
    return {
        "id": "a",
        "dataset": "r2r",
        "episode_id": "7",
        "episode_index": 3,
        "instruction": "Navigate",
        "scene": "test",
    }


def test_stage_copies_inputs_without_writing_source(tmp_path):
    source = tmp_path / "original"
    source.mkdir()
    data = source / "annotation.json"
    data.write_text('{"instruction":"navigate"}')
    expected = file_digest(data)
    (source / "annotation-link.json").symlink_to(data)
    section = {
        "spec": {**spec(), "data_root": str(source), "inputs": {str(data): expected}},
        "episodes": [episode()],
    }
    destination = tmp_path / "delivery/assets"
    stage(section, {str(source): "data"}, destination)
    document = read_json(destination / "catalog.json")
    assert document["spec"]["data_root"] == "data"
    assert document["spec"]["inputs"] == {"data/annotation.json": expected}
    assert file_digest(data) == expected
    assert not (destination / "data/annotation-link.json").is_symlink()
    with pytest.raises(ValueError, match="outside"):
        stage(section, {str(source): "data"}, source / "bad")


def test_token_accounting_excludes_copied_training_logs(tmp_path):
    root = tmp_path / "bench/evaluation/case-a"
    evidence = root / "native/result.json"
    write_json(
        evidence, {"success": True, "steps": 1, "terminated": True, "truncated": False}
    )
    write_json(root / "result.json", {})
    write_json(
        root / "agent/codex-cli/call-0001/result.json",
        {
            "stats": {
                "usage_status": "complete",
                "total_input_tokens": 10,
                "total_output_tokens": 3,
            }
        },
    )
    write_json(
        root / "agent/workspace/old/result.json",
        {
            "stats": {
                "usage_status": "complete",
                "total_input_tokens": 9999,
                "total_output_tokens": 9999,
            }
        },
    )
    task = make_task(episode(), spec(), 500)
    result = EvaluationResult(
        task=task,
        episode_id="episode",
        status="success",
        success=True,
        score=1.0,
        provenance=Provenance(),
        native_score_source="native/result.json",
        native_score_sha256=file_digest(evidence),
    )
    backend = BenchExecutor({"r2r": {"test": {"spec": spec()}}}, {}, threading.Event())
    actual = backend._result(episode(), 500, tmp_path, result)
    assert actual["usage"] == {
        "status": "complete",
        "input_tokens": 10,
        "output_tokens": 3,
    }
