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

"""Check a deployed SIMPLE HTTP service with at most three rows and no model."""

from __future__ import annotations

import argparse
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

logger = get_logger("simple.http_smoke")


def check(endpoint: str, task_id: str, output: Path, source_sha256: str) -> dict:
    """Verify identity, reset, action feedback, downloads and owned-case cleanup.

    This cancels a nonterminal case after three rows. It never changes the native
    budget, supplies a model, or reports an unfinished task as success or failure.
    """
    output.mkdir(parents=True, exist_ok=False)
    client = BenchmarkClient(endpoint, token=os.environ.get("ROBORSI_BENCHMARK_TOKEN"))
    episode_id = "simple-smoke-" + uuid.uuid4().hex
    media = CaseMediaChannel(output / "media")
    report = {
        "scope": "model_free_http_smoke",
        "passed": False,
        "model_requests": 0,
        "task_success": None,
        "native_finish_exercised": False,
        "native_rows": 0,
        "episode_id": episode_id,
        "task_id": task_id,
    }
    started = time.monotonic()
    case_started = False
    try:
        runtime = client.runtime_info(timeout_s=10)
        if runtime is None or runtime.adapter_version != (
            "roborsi-source-sha256:" + source_sha256
        ):
            raise ValueError("SIMPLE service source identity differs from export")
        (output / "runtime.json").write_text(
            json.dumps(runtime.as_dict(), indent=2) + "\n"
        )
        task = next(t for t in client.tasks() if t.task_id == task_id)
        (output / "task.json").write_text(json.dumps(task.as_dict(), indent=2) + "\n")
        case_started = True
        observation = client.start(task, episode_id, timeout_s=900).observation
        terminal = False
        for index in range(4):
            client.collect_observation(observation, media, timeout_s=20)
            media.verify_observation(observation)
            state = observation.data["state"]
            if len(state) != 32 or not all(math.isfinite(x) for x in state):
                raise ValueError("SIMPLE returned invalid proprioception")
            if not observation.media:
                raise ValueError("SIMPLE returned no RGB media")
            (output / f"observation_{index:02d}.json").write_text(
                json.dumps(observation.as_dict(), indent=2) + "\n"
            )
            report["native_rows"] = observation.data["native_env_steps"]
            if terminal or index == 3:
                break
            action = Action(
                benchmark="simple",
                episode_id=episode_id,
                step_index=observation.step_index,
                data={"actions": [[*state, 0.0, 0.0, 0.0, 0.0]]},
            )
            (output / f"action_{index:02d}.json").write_text(
                json.dumps(action.as_dict(), indent=2) + "\n"
            )
            step = client.step(action, timeout_s=120)
            observation = step.observation
            terminal = step.terminated or step.truncated
        if not 1 <= report["native_rows"] <= 3:
            raise ValueError("SIMPLE smoke exceeded its three-row execution limit")
        if terminal:
            native = client.finish(timeout_s=20)
            client.collect_native(native, output, timeout_s=20)
            report["native_finish_exercised"] = True
        client.collect_artifacts(output, timeout_s=20)
        report["downloaded_artifacts"] = len(client.artifact_records)
        client.close()
        report["state_after_close"] = client.status(episode_id)["state"]
        client.delete(episode_id)
        report["case_deleted"] = True
        case_started = False
        report["passed"] = True
    except BaseException as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if case_started:
            try:
                client.collect_artifacts(output, timeout_s=20)
                report["downloaded_artifacts"] = len(client.artifact_records)
            except Exception as artifact_error:
                report["artifact_download_error"] = type(artifact_error).__name__
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
    """Run the explicitly selected short check against an existing service."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    args = parser.parse_args()
    if len(args.source_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in args.source_sha256
    ):
        parser.error("--source-sha256 must be the exported source.json SHA-256")
    report = check(args.endpoint, args.task_id, args.output, args.source_sha256)
    logger.info("SIMPLE HTTP smoke: %s", report)


if __name__ == "__main__":
    main()
