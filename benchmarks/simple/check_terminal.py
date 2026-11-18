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

"""Check one 1x native episode through finish without a model or budget override."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import uuid
from pathlib import Path

from roborsi.evaluation.benchmark_client import BenchmarkClient
from roborsi.evaluation.media import CaseMediaChannel
from roborsi.evaluation.messages import Action
from roborsi.utils.logging import get_logger
from robots.simple.contracts import validate_task

logger = get_logger("simple.terminal_check")


def check(endpoint: str, task_id: str, output: Path, source_sha256: str) -> dict:
    """Hold the initial state until native termination or the unchanged 1x limit.

    This is an interface diagnostic, not a task policy. It checks scoring against
    downloaded native evidence and never fabricates a successful task outcome.
    """
    output.mkdir(parents=True, exist_ok=False)
    client = BenchmarkClient(endpoint, token=os.environ.get("ROBORSI_BENCHMARK_TOKEN"))
    media = CaseMediaChannel(output / "media")
    episode_id = "simple-terminal-" + uuid.uuid4().hex
    report = {
        "scope": "model_free_native_terminal_check",
        "passed": False,
        "model_requests": 0,
        "episode_id": episode_id,
        "task_id": task_id,
        "native_finish_exercised": False,
        "native_rows": 0,
        "task_success": None,
    }
    started = time.monotonic()
    case_started = False
    try:
        runtime = client.runtime_info(timeout_s=10)
        if runtime is None or runtime.adapter_version != (
            "roborsi-source-sha256:" + source_sha256
        ):
            raise ValueError("SIMPLE service source identity differs")
        (output / "runtime.json").write_text(json.dumps(runtime.as_dict(), indent=2))
        task = next(t for t in client.tasks() if t.task_id == task_id)
        validate_task(task)
        if task.max_steps != task.native["official_steps"]:
            raise ValueError("terminal diagnostic requires an unchanged 1x catalog")
        report["step_limit"] = task.max_steps
        (output / "task.json").write_text(json.dumps(task.as_dict(), indent=2))
        case_started = True
        observation = client.start(task, episode_id, timeout_s=900).observation
        hold = None
        terminal = False
        while True:
            state = observation.data["state"]
            if len(state) != 32 or not all(math.isfinite(x) for x in state):
                raise ValueError("invalid public state")
            rows = observation.data["native_env_steps"]
            remaining = observation.data["remaining_native_steps"]
            if not 0 <= rows <= task.max_steps or remaining != task.max_steps - rows:
                raise ValueError("native step accounting differs from task budget")
            if not observation.media:
                raise ValueError("missing RGB feedback")
            client.collect_observation(observation, media, timeout_s=30)
            (output / f"observation_{observation.step_index:04d}.json").write_text(
                json.dumps(observation.as_dict(), indent=2)
            )
            report["native_rows"] = rows
            report["elapsed_s"] = time.monotonic() - started
            (output / "progress.json").write_text(json.dumps(report, indent=2))
            if terminal:
                break
            if not remaining:
                raise ValueError("native budget exhausted without a terminal response")
            if hold is None:
                hold = [*state, 0.0, 0.0, 0.0, 0.0]
            action = Action(
                benchmark="simple",
                episode_id=episode_id,
                step_index=observation.step_index,
                data={"actions": [hold[:] for _ in range(min(20, remaining))]},
            )
            (output / f"action_{observation.step_index:04d}.json").write_text(
                json.dumps(action.as_dict(), indent=2)
            )
            step = client.step(action, timeout_s=180)
            (output / f"step_{observation.step_index:04d}.json").write_text(
                json.dumps(step.as_dict(), indent=2)
            )
            observation = step.observation
            advanced = observation.data["native_env_steps"] - rows
            if not 1 <= advanced <= len(action.data["actions"]):
                raise ValueError("native action progress is inconsistent")
            terminal = step.terminated or step.truncated
            if terminal:
                report.update(
                    terminated=step.terminated,
                    truncated=step.truncated,
                    termination_reason=step.termination_reason,
                )
            logger.info(
                "Native terminal check: %s/%s rows", rows + advanced, task.max_steps
            )

        native = client.finish(timeout_s=30)
        client.collect_native(native, output, timeout_s=30)
        client.collect_artifacts(output, timeout_s=60)
        (output / "native_response.json").write_text(
            json.dumps(native.as_dict(), indent=2)
        )
        source = output / "native/result.json"
        if (
            native.source != "native/result.json"
            or hashlib.sha256(source.read_bytes()).hexdigest() != native.source_sha256
        ):
            raise ValueError("native result source hash differs")
        saved = json.loads(source.read_text())
        records = [
            json.loads(line)
            for line in (output / "native/actions.jsonl").read_text().splitlines()
        ]
        if len(records) != rows or [r["native_step"] for r in records] != list(
            range(1, rows + 1)
        ):
            raise ValueError("downloaded native action trace has missing rows")
        first = next((r["native_step"] for r in records if r["success"]), None)
        success = first is not None
        if (
            native.task != task
            or native.episode_id != episode_id
            or native.success is not success
            or native.score != float(success)
            or saved != {"success": success, **native.details}
            or native.details["native_env_steps"] != rows
            or native.details["first_success_step"] != first
            or native.details["official_1x_steps"] != task.max_steps
            or native.details["success_within_1x"] is not success
        ):
            raise ValueError("native score disagrees with downloaded action evidence")
        if report["truncated"] and (rows != task.max_steps or success):
            raise ValueError("budget truncation is inconsistent with native score")
        report.update(
            native_finish_exercised=True,
            task_success=success,
            score=native.score,
            first_success_step=first,
            downloaded_artifacts=len(client.artifact_records),
            score_matches_action_trace=True,
            state_after_finish=client.status(episode_id)["state"],
        )
        if report["state_after_finish"] != "completed":
            raise ValueError("finished episode is not completed")
        client.close()
        report["state_after_close"] = client.status(episode_id)["state"]
        client.delete(episode_id)
        case_started = False
        report.update(case_deleted=True, passed=True)
    except BaseException as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if case_started:
            try:
                client.collect_artifacts(output, timeout_s=30)
            except Exception as error:
                report["artifact_download_error"] = type(error).__name__
        raise
    finally:
        try:
            if case_started:
                client.close()
        finally:
            report["elapsed_s"] = time.monotonic() - started
            (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """Run one explicit 1x terminal diagnostic against an existing service."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    args = parser.parse_args()
    if len(args.source_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in args.source_sha256
    ):
        parser.error("source-sha256 must be the image's exported source digest")
    logger.info("SIMPLE terminal check: %s", check(**vars(args)))


if __name__ == "__main__":
    main()
