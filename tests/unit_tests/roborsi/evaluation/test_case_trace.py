# Copyright 2026 The SenseNova-RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Saved interaction records reject damaged events, media and task identity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from roborsi.evaluation.case_trace import TRACE_MANIFEST
from roborsi.evaluation.media import CaseMediaChannel
from roborsi.evaluation.messages import (
    Action,
    NativeResult,
    Observation,
    StartResult,
    StepResult,
)
from roborsi.evaluation.store import ResultStore
from tests.unit_tests.roborsi.evaluation.test_store import (
    _provenance,
    _result,
    _task,
)


def _recorded_case(root: Path, *, extra_event: str | None = None):
    store = ResultStore(root)
    task, provenance = _task(), _provenance()
    directory = store.claim("run1", "case1", task, provenance)
    media = CaseMediaChannel(directory / "media")
    reference = media.write("episode-1", 0, b"camera bytes", "image/png")
    observation = Observation(
        benchmark="demo",
        episode_id="episode-1",
        step_index=0,
        media=(reference,),
    )
    trace = store.start_trace("run1", "case1", "episode-1")
    trace.record(StartResult(task=task, observation=observation), stage="adapter.start")
    trace.begin_step(observation)
    trace.record(
        Action(benchmark="demo", episode_id="episode-1", step_index=0, data={}),
        stage="agent.act",
    )
    trace.record(
        StepResult(
            observation=Observation(
                benchmark="demo", episode_id="episode-1", step_index=1
            ),
            terminated=True,
            truncated=False,
            termination_reason="native_success",
        ),
        stage="adapter.step",
    )
    trace.end_step()
    trace.record(
        NativeResult(task=task, episode_id="episode-1", success=True, score=1.0),
        stage="adapter.finish",
    )
    if extra_event is not None:
        trace.emit(extra_event, {"note": "extension payload"})
    trace.finish("success")
    result = _result(task, provenance, trace_manifest=TRACE_MANIFEST)
    store.publish("run1", "case1", result)
    return store, directory, reference


def test_roborsi_trace_roundtrip_preserves_messages_and_extension_events(
    tmp_path: Path,
) -> None:
    store, directory, _ = _recorded_case(tmp_path, extra_event="benchmark_note")
    state = store.trace_state("run1", "case1")
    manifest = json.loads((directory / TRACE_MANIFEST).read_text())
    assert manifest["format"] == "roborsi_case_trace"
    assert state.complete
    assert store.load("run1", "case1").success
    events = [
        json.loads(line)
        for line in (directory / manifest["events"]["ref"]).read_text().splitlines()
    ]
    messages = [event for event in events if event["type"] == "roborsi_message"]
    assert [event["payload"]["stage"] for event in messages] == [
        "adapter.start",
        "agent.act",
        "adapter.step",
        "adapter.finish",
    ]
    assert any(event["type"] == "benchmark_note" for event in events)


def test_saved_trace_rejects_heart_manifest_format(tmp_path: Path) -> None:
    store, directory, _ = _recorded_case(tmp_path)
    manifest_path = directory / TRACE_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest["format"] = "heart_case_trace"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="roborsi_case_trace"):
        store.load("run1", "case1")


def test_saved_trace_rejects_heart_message_event(tmp_path: Path) -> None:
    store, directory, _ = _recorded_case(tmp_path)
    manifest_path = directory / TRACE_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    events_path = directory / manifest["events"]["ref"]
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    event = next(item for item in events if item["payload"].get("stage") == "agent.act")
    event["type"] = "heart_message"
    data = ("\n".join(json.dumps(item) for item in events) + "\n").encode()
    events_path.write_bytes(data)
    manifest["events"].update(
        size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest()
    )
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsupported.*heart_message"):
        store.load("run1", "case1")


@pytest.mark.parametrize(
    "damaged",
    [
        "events",
        "media",
        "identity",
        "version",
        "terminal",
        "order",
        "episode",
        "message_task",
        "missing_message",
    ],
)
def test_saved_trace_rejects_changed_files_and_identity(
    tmp_path: Path, damaged: str
) -> None:
    store, directory, image = _recorded_case(tmp_path)
    assert store.trace_state("run1", "case1").complete
    manifest_path = directory / TRACE_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    events_path = directory / manifest["events"]["ref"]
    if damaged == "events":
        events_path.write_bytes(events_path.read_bytes() + b"{}\n")
    elif damaged == "media":
        (directory / image.ref).write_bytes(b"damaged")
    elif damaged in {"identity", "version"}:
        if damaged == "identity":
            manifest["task"]["seed"] += 1
        else:
            manifest["schema_version"] += 1
        manifest_path.write_text(json.dumps(manifest))
    else:
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        if damaged == "terminal":
            events.pop()
            manifest["last_seq"] -= 1
        elif damaged == "order":
            events[1]["seq"] += 1
        elif damaged == "episode":
            event = next(item for item in events if item["type"] == "roborsi_message")
            event["payload"]["message"]["episode_id"] = "another-episode"
        else:
            event = next(item for item in events if item["type"] == "roborsi_message")
            if damaged == "message_task":
                event["payload"]["message"]["payload"]["task"]["seed"] += 1
            else:
                del event["payload"]["message"]
        data = ("\n".join(json.dumps(event) for event in events) + "\n").encode()
        events_path.write_bytes(data)
        manifest["events"].update(
            size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest()
        )
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        store.load("run1", "case1")


def test_trace_writer_cannot_replace_a_claimed_or_published_recording(
    tmp_path: Path,
) -> None:
    store, directory, _ = _recorded_case(tmp_path)
    original = (directory / TRACE_MANIFEST).read_bytes()
    with pytest.raises(FileExistsError):
        store.start_trace("run1", "case1", "another-episode")
    assert (directory / TRACE_MANIFEST).read_bytes() == original
