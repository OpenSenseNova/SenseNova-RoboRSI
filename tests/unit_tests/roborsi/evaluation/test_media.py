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

"""Media references stay within their case, episode and observation step."""

from __future__ import annotations

from pathlib import Path

import pytest

from roborsi.evaluation.media import MAX_MEDIA_BYTES, CaseMediaChannel
from roborsi.evaluation.messages import Observation


def _observation(*media) -> Observation:
    return Observation(benchmark="demo", episode_id="case-1", step_index=0, media=media)


def test_media_channel_round_trips_two_camera_images(tmp_path: Path) -> None:
    writer = CaseMediaChannel(tmp_path / "media")
    front = writer.write("case-1", 0, b"front-image", "image/png")
    wrist = writer.write("case-1", 0, b"wrist-image", "image/jpeg")
    reader = CaseMediaChannel(tmp_path / "media")

    reader.verify_observation(_observation(front, wrist))
    assert reader.read(front, "case-1", 0) == b"front-image"
    assert reader.read(wrist, "case-1", 0) == b"wrist-image"
    assert front.ref.startswith("media/") and wrist.ref.startswith("media/")
    assert not list((tmp_path / "media").rglob("*.tmp"))


def test_media_channel_rejects_other_steps_cases_and_damaged_files(
    tmp_path: Path,
) -> None:
    channel = CaseMediaChannel(tmp_path / "media")
    reference = channel.write("case-1", 0, b"camera", "image/png")
    with pytest.raises(ValueError, match="different observation"):
        channel.read(reference, "case-2", 0)
    with pytest.raises(ValueError, match="different observation"):
        channel.read(reference, "case-1", 1)
    with pytest.raises(ValueError, match="repeats"):
        channel.verify_observation(_observation(reference, reference))

    file = tmp_path / reference.ref
    file.write_bytes(b"damage")
    with pytest.raises(ValueError, match="digest mismatch"):
        channel.verify_observation(_observation(reference))
    file.unlink()
    with pytest.raises(ValueError, match="unavailable"):
        channel.read(reference, "case-1", 0)


def test_media_channel_rejects_symlinks_and_oversized_references(
    tmp_path: Path,
) -> None:
    channel = CaseMediaChannel(tmp_path / "media")
    reference = channel.write("case-1", 0, b"camera", "image/png")
    path = tmp_path / reference.ref
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"camera")
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        channel.read(reference, "case-1", 0)

    oversized = reference.model_copy(update={"size_bytes": MAX_MEDIA_BYTES + 1})
    with pytest.raises(ValueError, match="size limit"):
        channel.read(oversized, "case-1", 0)
