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

"""Read existing navigation inputs without changing source code or historical runs."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

from robots.vln_shared.storage import file_digest, read_json, write_json

VLNCE_COMMIT = "729d141b2ee10628061ada74dd3a5b9f70faeba5"
LHPR_COMMIT = "6c0a17d3830be5fa9c64e8dec35be29808581f2d"
DATASETS = ("lhpr_vln", "r2r", "rxr")


def annotations(spec: dict) -> tuple[list[dict], Path]:
    """Read annotation order exactly as consumed by the native loader."""
    root = Path(spec["data_root"])
    split = spec["split"]
    if spec["dataset"] == "lhpr_vln":
        path = root / "data" / f"{split}.jsonl"
        return [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ], path
    suffix = "_guide" if spec["dataset"] == "rxr" else ""
    path = root / split / f"{split}{suffix}.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)["episodes"], path


def catalog(spec: dict, selected: list[str]) -> dict:
    """Pin sources and annotations and retain each selected original row index."""
    rows, annotation = annotations(spec)
    lhpr = spec["dataset"] == "lhpr_vln"
    indexed = {
        str(row["id" if lhpr else "episode_id"]): (i, row) for i, row in enumerate(rows)
    }
    if len(indexed) != len(rows) or len(set(selected)) != len(selected):
        raise ValueError("duplicate annotation or selection ID")
    files = [annotation]
    source = Path(spec["runtime_root"] if lhpr else spec["repo_root"])
    if lhpr:
        files += [
            source / "_runtime.py",
            source / "_metrics.py",
            *sorted((source / "vendor").glob("*.py")),
        ]
        files += [Path(spec["scene_dataset"])]
    else:
        files += sorted((source / "habitat_extensions").rglob("*.py"))
        files += sorted((source / "habitat_extensions/config").glob("*.yaml"))
        files += [
            annotation.with_name(annotation.name.replace(".json.gz", "_gt.json.gz"))
        ]
    episodes = []
    for selected_id in selected:
        index, row = indexed[selected_id]
        episodes.append(
            {
                "id": f"{spec['dataset']}-{selected_id}",
                "dataset": spec["dataset"],
                "episode_id": selected_id,
                "episode_index": index,
                "instruction": row["instruction"]
                if lhpr
                else row["instruction"]["instruction_text"],
                "scene": row["scene"] if lhpr else row["scene_id"],
            }
        )
    return {
        "spec": {**spec, "inputs": {str(p.absolute()): file_digest(p) for p in files}},
        "episodes": episodes,
    }


def prepare(
    source: Path,
    history: Path,
    destination: Path,
    *,
    reuse_exploration: bool = False,
    cli: str = "codex",
    cli_version: str = "codex-cli 0.161.0",
    selected_datasets: list[str] | None = None,
) -> Path:
    """Import task identities only; old success/failure never controls new scheduling."""
    names = DATASETS if selected_datasets is None else selected_datasets
    if not names or len(set(names)) != len(names) or set(names) - set(DATASETS):
        raise ValueError("select unique supported navigation datasets")
    source, history = source.resolve(), history.resolve()
    destination = destination.absolute()
    if destination.is_relative_to(source) or destination.is_relative_to(history):
        raise ValueError("new inputs must be outside the source project and history")
    destination.mkdir(parents=True, exist_ok=False)
    index = history / "episode_rollouts_500_then_5000/source_episode_index.csv"
    with index.open(encoding="utf-8-sig", newline="") as stream:
        selected_rows = list(csv.DictReader(stream))
    datasets = {}
    for legacy in ("lhvln", "r2r", "rxr"):
        name = "lhpr_vln" if legacy == "lhvln" else legacy
        if name not in names:
            continue
        old = read_json(history / "configs" / f"{legacy}.json")
        options = old["benchmark_options"]
        filename = {"lhvln": "lhvln.csv", "r2r": "r2rce.csv", "rxr": "rxrce.csv"}[
            legacy
        ]
        ids = [r["episode_id"] for r in selected_rows if r["csv_file"] == filename]
        if not ids:
            raise ValueError(
                f"historical selection contains no episodes for {filename}"
            )
        train_ids = [str(e["episode_id"]) for e in old["train_episodes"]]
        common = {
            "dataset": name,
            "python": options["simulator_python"],
            "seed": options["seed"],
            "gpu_device_id": options.get("gpu_device_id", -1),
        }
        if legacy == "lhvln":
            native = source.parent / "AgentCanvas/workspace/nodesets/env/env_lhvln"
            data = source.parent / "EASI-private/datasets/oscarqjh_LHPR-VLN_easi"
            training_data = Path(options["data_root"])
            test_spec = {
                **common,
                "native_commit": LHPR_COMMIT,
                "runtime_root": str(native),
                "data_root": str(data),
                "split": "unseen_test_filtered_0708",
                "scene_dataset": str(
                    data / "hm3d/hm3d_annotated_basis.scene_dataset_config.json"
                ),
                "regions_root": str(
                    source.parent / "SenseNova-Nav/vln_data_gen/outputs/scenes"
                ),
            }
            train_spec = {
                **common,
                "native_commit": LHPR_COMMIT,
                "split": "train",
                "runtime_root": str(
                    source / "third_party/benchmarks/sn-lhvln/sn_lhvln"
                ),
                "data_root": str(training_data),
                "external_scenes": True,
                "scenes_root": str(training_data.parent / "hm3d"),
                "scene_dataset": str(
                    training_data / "hm3d_annotated_basis.scene_dataset_config.json"
                ),
                "regions_root": str(training_data / "regions"),
            }
        else:
            base = {
                **common,
                "native_commit": VLNCE_COMMIT,
                "repo_root": str(source / "third_party/benchmarks/vln-ce"),
                "scenes_root": options["scenes_root"],
                "role": "guide",
            }
            test_spec = {
                **base,
                "data_root": str(
                    source
                    / "data"
                    / ("r2r-rand100" if name == "r2r" else "RxR_VLNCE_v0")
                ),
                "split": "rand100" if name == "r2r" else "val_unseen",
            }
            train_spec = {**base, "data_root": options["data_root"], "split": "train"}
        training, testing = catalog(train_spec, train_ids), catalog(test_spec, ids)
        train_scenes = {e["scene"] for e in training["episodes"]}
        test_scenes = {e["scene"] for e in testing["episodes"]}
        if train_scenes & test_scenes:
            raise ValueError(f"training and evaluation scenes overlap: {name}")
        datasets[name] = {"train": training, "test": testing}
        if reuse_exploration:
            datasets[name]["import_snapshot"] = str(history / "shared" / legacy)
    path = destination / "campaign.json"
    write_json(
        path,
        {
            "schema_version": 1,
            "datasets": datasets,
            "agent": {"executable": cli, "expected_version": cli_version},
            "provenance": {
                "source_project": str(source),
                "history": str(history),
                "selection_sha256": file_digest(index),
                "selection_policy": "IDs only; all cases rerun at 500",
            },
        },
    )
    return path
