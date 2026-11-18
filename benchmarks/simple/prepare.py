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

"""Export released evaluation resets into a private, hash-checked catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from roborsi.evaluation.messages import Task
from robots.simple.contracts import (
    ACTION_SCHEMA,
    EVALUATION_RULES,
    SIMPLE_REVISION,
    SOURCES,
)


def export_catalog(
    archives: Path,
    output: Path,
    aliases: list[str],
    *,
    multiplier: int = 1,
    seed_base: int = 20261002,
) -> Path:
    """Verify archive bytes, retain all ten cases at every level and export resets."""
    if multiplier not in (1, 10) or type(seed_base) is not int or seed_base < 0:
        raise ValueError(
            "explicit 1x/10x multiplier and nonnegative seed base required"
        )
    if (
        not aliases
        or len(set(aliases)) != len(aliases)
        or any(a not in SOURCES["tasks"] for a in aliases)
    ):
        raise ValueError("choose unique supported SIMPLE tasks")
    output.mkdir(parents=True, exist_ok=False)
    try:
        (output / "resets").mkdir()
        cases = []
        for alias in aliases:
            spec = SOURCES["tasks"][alias]
            path = archives / spec["test_archive"]["path"]
            raw = path.read_bytes()
            if (
                len(raw) != spec["test_archive"]["bytes"]
                or hashlib.sha256(raw).hexdigest() != spec["test_archive"]["sha256"]
            ):
                raise ValueError(f"released SIMPLE test archive differs: {alias}")
            with zipfile.ZipFile(path) as archive:
                for level in range(3):
                    entry = f"{spec['env_id']}/dr-level-{level}/meta/episodes.jsonl"
                    info = archive.getinfo(entry)
                    if info.file_size > 16 * 1024 * 1024:
                        raise ValueError("SIMPLE reset metadata exceeds size bound")
                    rows = [
                        json.loads(line) for line in archive.read(entry).splitlines()
                    ]
                    if [r["episode_index"] for r in rows] != list(range(10)):
                        raise ValueError("expected all ten released cases in order")
                    for row in rows:
                        if (
                            not isinstance(row.get("tasks"), list)
                            or len(row["tasks"]) != 1
                            or not isinstance(row["tasks"][0], str)
                            or not row["tasks"][0].strip()
                        ):
                            raise ValueError(
                                "released case needs exactly one native instruction"
                            )
                        episode = row["episode_index"]
                        state = json.loads(row["environment_config"])
                        scene = state["dr_state_dict"]["scene"]
                        scene["uid"] = scene["uid"].replace("102344280", "scene3")
                        reset = json.dumps(state, sort_keys=True).encode()
                        ref = f"resets/{alias}_L{level}_E{episode:02d}.json"
                        (output / ref).write_bytes(reset)
                        task = Task(
                            benchmark="simple",
                            task_id=f"{alias}/L{level}/E{episode:02d}",
                            seed=seed_base + level * 100 + episode,
                            instruction=row["tasks"][0],
                            max_steps=spec["official_steps"] * multiplier,
                            native={
                                "alias": alias,
                                "level": level,
                                "episode": episode,
                                "official_steps": spec["official_steps"],
                                "controller": spec["controller"],
                                "reset_sha256": hashlib.sha256(reset).hexdigest(),
                                "score_protocol": EVALUATION_RULES["score_protocol"],
                            },
                            capabilities={
                                "action_schema": ACTION_SCHEMA,
                                "observation_channels": ["head_stereo_left"],
                            },
                        )
                        cases.append({"task": task.as_dict(), "reset": ref})
        catalog = output / "catalog.json"
        catalog.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "simple_revision": SIMPLE_REVISION,
                    "dataset_revision": SOURCES["dataset_revision"],
                    "cases": cases,
                },
                indent=2,
            )
            + "\n"
        )
        return catalog
    except BaseException:
        shutil.rmtree(output)
        raise


def main() -> None:
    """Read explicitly downloaded archives; never silently fetch or select trials."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tasks", nargs="+", choices=list(SOURCES["tasks"]), required=True
    )
    parser.add_argument("--budget-multiplier", type=int, choices=[1, 10], default=1)
    parser.add_argument("--seed-base", type=int, default=20261002)
    args = parser.parse_args()
    export_catalog(
        args.archives,
        args.output,
        args.tasks,
        multiplier=args.budget_multiplier,
        seed_base=args.seed_base,
    )


if __name__ == "__main__":
    main()
