# Copyright 2026 The RoboRSI Authors.
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

"""Score RoboCasa365 Target50 runs: the mean of the 50 per-task success rates.

Every expected cell counts. Under the Target50 retry policy a planner that
runs out of its wall clock is a valid task failure, not a re-run. Any other
cell without a valid native result (missing, or an infrastructure error)
scores as a failure and is listed, so it can be re-run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robots.robocasa.benchmark_service import TARGET50

VALID = {"success", "task_failure", "planner_timeout"}


def _status(result: dict) -> str:
    message = (result.get("error") or {}).get("message") or ""
    if result["status"] == "agent_error" and "planner timed out" in message:
        return "planner_timeout"
    return result["status"]


def collect(runs: list[Path]) -> dict[tuple[str, int], dict]:
    cells: dict[tuple[str, int], dict] = {}
    for run in runs:
        for path in sorted(run.glob("case-*/result.json")):
            result = json.loads(path.read_text(encoding="utf-8"))
            task = result["task"]
            if task.get("benchmark") != "robocasa365":
                continue
            key = (task["task_id"], task["seed"])
            # A valid result replaces an earlier invalid attempt of the same cell.
            if key not in cells or cells[key]["status"] not in VALID:
                cells[key] = {"status": _status(result), "path": str(path)}
    return cells


def score(cells: dict[tuple[str, int], dict], seeds: str = "all") -> dict:
    report = {"splits": {}, "invalid": []}
    rates = []
    for name, split in TARGET50["splits"].items():
        expected = [s for s in split["seeds"] if seeds == "all" or s == 1]
        split_rates, solved, total = [], 0, 0
        for task in split["tasks"]:
            ok = 0
            for seed in expected:
                cell = cells.get((task, seed))
                if cell is None or cell["status"] not in VALID:
                    report["invalid"].append(f"{name}/{task}_s{seed}")
                ok += bool(cell and cell["status"] == "success")
            solved, total = solved + ok, total + len(expected)
            split_rates.append(ok / len(expected))
        rates += split_rates
        report["splits"][name] = {
            "solved": solved,
            "cells": total,
            "success_rate": round(100 * sum(split_rates) / len(split_rates), 2),
        }
    report["overall"] = round(100 * sum(rates) / len(rates), 2)
    report["cells_expected"] = sum(s["cells"] for s in report["splits"].values())
    report["cells_valid"] = report["cells_expected"] - len(report["invalid"])
    report["planner_timeouts"] = sorted(
        f"{task}_s{seed}"
        for (task, seed), cell in cells.items()
        if cell["status"] == "planner_timeout"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+", help="bench eval run directories")
    parser.add_argument("--cells", choices=("all", "seed1"), default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = score(collect(args.runs), args.cells)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
