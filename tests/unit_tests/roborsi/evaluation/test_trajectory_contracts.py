# Copyright 2026 The RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
from pathlib import Path

import pytest


def event(seq: int, kind: str, **fields) -> dict:
    return {"seq": seq, "event_id": str(seq), "run_id": "run", "type": kind, **fields}


def test_usage_counts_attempts_once_and_weights_cache_by_input() -> None:
    from roborsi.evaluation.trajectory import TrajectoryProjection

    projection = TrajectoryProjection()
    events = [
        event(
            1,
            "model_attempt_end",
            request_id="a",
            attempt=1,
            payload={"status": "error", "usage": None},
        ),
        event(
            2,
            "model_attempt_end",
            request_id="a",
            attempt=2,
            agent_id="parent",
            payload={
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "cache_read_tokens": 8,
                }
            },
        ),
        event(
            3,
            "model_attempt_end",
            request_id="b",
            attempt=1,
            agent_id="child",
            payload={
                "usage": {
                    "input_tokens": 90,
                    "output_tokens": 3,
                    "cache_read_tokens": 2,
                }
            },
        ),
        event(
            4,
            "model_request_end",
            request_id="a",
            payload={"usage": {"input_tokens": 999}},
        ),
    ]
    for item in [*events, events[1]]:
        projection.apply(item)
    usage = projection.summary()["usage"]
    assert usage["input_tokens"] == 100
    assert usage["total_tokens"] == 105
    assert usage["cache_ratio"] == 0.1
    assert usage["attempts"] == 3
    assert usage["unknown_usage_attempts"] == 1


def test_zero_or_unreported_cache_is_not_invented() -> None:
    from roborsi.evaluation.trajectory import TrajectoryProjection

    projection = TrajectoryProjection()
    projection.apply(
        event(
            1,
            "model_attempt_end",
            request_id="a",
            attempt=1,
            payload={"usage": {"input_tokens": 4, "output_tokens": 1}},
        )
    )
    assert projection.summary()["usage"]["cache_ratio"] is None
    projection.apply(
        event(
            2,
            "model_attempt_end",
            request_id="a",
            attempt=1,
            payload={
                "usage": {"input_tokens": 0, "output_tokens": 1, "cache_read_tokens": 0}
            },
        )
    )
    assert projection.summary()["usage"]["cache_ratio"] is None
    assert projection.summary()["usage"]["attempts"] == 1


def test_live_cumulative_usage_matches_offline_when_parallel_turn_finishes_late() -> (
    None
):
    from roborsi.evaluation.trajectory import TrajectoryProjection

    projection = TrajectoryProjection()
    for item in [
        event(1, "turn_start", agent_id="first", turn_id="a"),
        event(2, "turn_start", agent_id="second", turn_id="b"),
        event(
            3,
            "model_attempt_end",
            turn_id="b",
            request_id="b",
            attempt=1,
            payload={"usage": {"input_tokens": 20}},
        ),
    ]:
        projection.apply(item)
    initial = projection.index()
    live_turns = {turn["turn_id"]: turn for turn in initial["turns"]}
    projection.apply(
        event(
            4,
            "model_attempt_end",
            turn_id="a",
            request_id="a",
            attempt=1,
            payload={"usage": {"input_tokens": 10}},
        )
    )
    delta = projection.index(after_seq=initial["summary"]["last_seq"])
    live_turns.update({turn["turn_id"]: turn for turn in delta["turns"]})
    assert list(live_turns.values()) == projection.index()["turns"]
    assert [turn["cumulative_tokens"] for turn in live_turns.values()] == [10, 30]
    assert projection.index(after_seq=4)["turns"] == []


def test_unfinished_attempt_is_counted_with_unknown_usage_then_replaced() -> None:
    from roborsi.evaluation.trajectory import TrajectoryProjection

    projection = TrajectoryProjection()
    started = event(
        1,
        "model_attempt_start",
        agent_id="root",
        turn_id="t",
        request_id="a",
        attempt=1,
        payload={"model": "test"},
    )
    projection.apply(started)
    projection.apply(started)
    usage = projection.summary()["usage"]
    assert usage["attempts"] == 1
    assert usage["unknown_usage_attempts"] == 1
    assert usage["reported_usage_attempts"] == 0
    assert usage["cache_ratio"] is None

    projection.apply(
        event(
            2,
            "model_attempt_end",
            agent_id="root",
            turn_id="t",
            request_id="a",
            attempt=1,
            payload={
                "status": "completed",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "cache_read_tokens": 5,
                },
            },
        )
    )
    usage = projection.summary()["usage"]
    assert usage["attempts"] == 1
    assert usage["unknown_usage_attempts"] == 0
    assert usage["total_tokens"] == 12
    assert usage["cache_ratio"] == 0.5


@pytest.mark.parametrize("explicit_reference", [False, True])
@pytest.mark.parametrize("step", [None, 3])
@pytest.mark.parametrize("stale_file", [False, True])
def test_failed_media_keeps_error_and_can_recover(
    tmp_path: Path,
    explicit_reference: bool,
    step: int | None,
    stale_file: bool,
) -> None:
    from roborsi.evaluation.trajectory import TrajectoryReader
    from roborsi.session import EnvState

    name = "episode.mp4" if step is None else "action_move_to.mp4"
    artifact = EnvState(tmp_path).artifact_path(name, step=step)
    reference = artifact.relative_to(tmp_path).as_posix()
    # A failed overwrite must not make an older on-disk artifact look current.
    artifact.parent.mkdir(parents=True, exist_ok=True)
    if stale_file:
        artifact.write_bytes(b"previous video")
    payload = {"error": "encoder unavailable"}
    payload.update(
        {"artifact_ref": reference}
        if explicit_reference
        else {
            "name": name,
            "step_idx": step,
        }
    )
    reader = TrajectoryReader(tmp_path)
    reader.projection.apply(event(1, "artifact_failed", payload=payload))
    media = reader.media()
    assert len(media) == 1
    assert media[0]["artifact_ref"] == reference
    assert media[0]["status"] == "failed"
    assert media[0]["error"] == "encoder unavailable"
    assert media[0]["available"] is False

    artifact.write_bytes(b"recovered video")
    reader.projection.apply(
        event(
            2,
            "artifact_written",
            payload={
                "artifact_ref": reference,
            },
        )
    )
    recovered = reader.media()[0]
    assert recovered["status"] == "ready"
    assert recovered["available"] is True
    assert not recovered.get("error")


def test_failed_media_name_cannot_escape_run(tmp_path: Path) -> None:
    from roborsi.evaluation.trajectory import TrajectoryReader

    reader = TrajectoryReader(tmp_path)
    reader.projection.apply(
        event(
            1,
            "artifact_failed",
            payload={
                "name": "../outside.mp4",
                "step_idx": None,
                "error": "failed",
            },
        )
    )
    with pytest.raises(ValueError, match="outside"):
        reader.media()


def test_reader_buffers_tail_and_rejects_corrupt_middle(tmp_path: Path) -> None:
    from roborsi.evaluation.trajectory import TrajectoryReader

    trace = tmp_path / "trace"
    trace.mkdir()
    path = trace / "events.jsonl"
    path.write_text(json.dumps(event(1, "run_start")) + '\n{"seq":2', encoding="utf-8")
    reader = TrajectoryReader(tmp_path)
    assert reader.update().summary()["last_seq"] == 1
    with path.open("a", encoding="utf-8") as stream:
        stream.write(',"type":"run_end","payload":{"status":"completed"}}\n')
    assert reader.update().summary()["status"] == "completed"
    with path.open("a", encoding="utf-8") as stream:
        stream.write("not json\n")
    with pytest.raises(ValueError, match="invalid trajectory event"):
        reader.update()


def test_request_payload_and_media_paths_stay_inside_run(tmp_path: Path) -> None:
    from roborsi.evaluation.trajectory import TrajectoryReader

    reader = TrajectoryReader(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        reader.artifact_path("../secret")
    with pytest.raises(ValueError, match="outside"):
        reader.artifact_path(str(tmp_path.parent / "secret"))
    reader.projection.apply(event(1, "turn_start", agent_id="root", turn_id="turn"))
    reader.projection.apply(
        event(
            2,
            "model_request_start",
            turn_id="turn",
            request_id="a",
            payload={"request_ref": "../secret"},
        )
    )
    with pytest.raises(ValueError, match="outside"):
        reader.turn("turn")


def test_video_segments_keep_each_tool_mapping(tmp_path: Path) -> None:
    from roborsi.evaluation.trajectory import TrajectoryReader

    (tmp_path / "episode.mp4").write_bytes(b"video")
    (tmp_path / "states.json").write_text(
        json.dumps(
            {
                "run_artifacts": ["episode.mp4"],
                "steps": [
                    {
                        "step_idx": step,
                        "artifacts": [],
                        "extras": {
                            "tool_call_id": f"tool-{step}",
                            "videos": [
                                {
                                    "video_ref": "episode.mp4",
                                    "frame_start": step * 4,
                                    "frame_end": (step + 1) * 4,
                                    "fps": 20,
                                }
                            ],
                        },
                    }
                    for step in range(2)
                ],
            }
        ),
        encoding="utf-8",
    )
    media = TrajectoryReader(tmp_path).media()
    assert len(media) == 1
    assert [segment["frame_start"] for segment in media[0]["segments"]] == [0, 4]
    assert media[0]["segments"][1]["tool_call_id"] == "tool-1"


def test_custom_tool_media_event_keeps_exact_frame_mapping(tmp_path: Path) -> None:
    from roborsi.evaluation.trajectory import TrajectoryReader

    (tmp_path / "synthetic-action.mp4").write_bytes(b"video")
    reader = TrajectoryReader(tmp_path)
    reader.projection.apply(
        event(
            1,
            "artifact_written",
            turn_id="turn-1",
            tool_call_id="tool-1",
            payload={
                "artifact_ref": "synthetic-action.mp4",
                "video_ref": "synthetic-action.mp4",
                "media_type": "video/mp4",
                "frame_start": 0,
                "frame_end": 12,
                "fps": 12,
                "step_idx": 1,
            },
        )
    )
    reader.projection.apply(
        event(
            2,
            "artifact_written",
            turn_id="turn-2",
            tool_call_id="tool-2",
            payload={"artifact_ref": "synthetic-action.mp4"},
        )
    )
    media = reader.media()
    assert len(media) == 1
    assert media[0]["available"] is True
    # A later event without frame positions must not invent another segment.
    assert media[0]["segments"] == [
        {
            "video_ref": "synthetic-action.mp4",
            "frame_start": 0,
            "frame_end": 12,
            "fps": 12,
            "step_idx": 1,
            "tool_call_id": "tool-1",
            "turn_id": "turn-1",
        }
    ]


def test_export_embeds_safe_payloads_and_copies_media(tmp_path: Path) -> None:
    from roborsi.cli.trajectory import export_trajectory

    run = tmp_path / "run"
    (run / "trace/requests").mkdir(parents=True)
    (run / "trace/requests/a.json").write_text(
        json.dumps({"messages": ["</script><img onerror=alert(1)>"]}), encoding="utf-8"
    )
    (run / "episode.mp4").write_bytes(b"video fixture")
    (run / "states.json").write_text(
        json.dumps({"run_artifacts": ["episode.mp4"], "steps": []}), encoding="utf-8"
    )
    events = [
        event(1, "turn_start", turn_id="t", agent_id="root"),
        event(
            2,
            "model_request_start",
            turn_id="t",
            request_id="a",
            payload={"request_ref": "trace/requests/a.json"},
        ),
    ]
    (run / "trace/events.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8"
    )
    output = tmp_path / "report"
    index = export_trajectory(run, output)
    html = index.read_text(encoding="utf-8")
    assert "</script><img onerror" not in html
    assert "\\u003c/script" in html
    assert (output / "media/episode.mp4").read_bytes() == b"video fixture"
    assert "partial" in html
    assert 'src="http' not in html
    with pytest.raises(ValueError, match="empty"):
        export_trajectory(run, output)
