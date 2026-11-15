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

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from roborsi.dashboard.events import NullDashboardEventSink
from roborsi.llm import LLMConfig
from roborsi.planner.base import build_planner
from roborsi.runtime import RuntimeConfig
from roborsi.runtime.trace import TraceConfig
from roborsi.session import EnvState


def test_runtime_model_config_reaches_actual_planner_request(tmp_path, monkeypatch):
    called = []

    def build(config):
        called.append(config.model)
        return FunctionModel(
            lambda messages, info: ModelResponse(parts=[TextPart("configured reply")])
        )

    monkeypatch.setattr(LLMConfig, "build_model", build)
    planner = build_planner(
        "api",
        output_dir=tmp_path,
        recipe_tag="test",
        robot_name="test",
        dashboard_events=NullDashboardEventSink(),
        include_image_reader=False,
        runtime=RuntimeConfig(
            llm=LLMConfig(provider="openai", model="selected"),
            trace=TraceConfig(mode="off"),
        ),
    )
    toolkit = SimpleNamespace(state=EnvState(tmp_path), get_tools_spec=lambda: [])
    result = planner.solve(
        system_prompt="rules", user_message="task", toolkit=toolkit, max_turns=2
    )
    assert called == ["selected"]
    assert not result.error
    assert any("configured reply" in str(item) for item in result.messages)


def test_conflicting_model_config_rejected_before_model_build(tmp_path):
    with pytest.raises(ValueError, match="runtime.*llm|llm.*runtime"):
        build_planner(
            "api",
            output_dir=tmp_path,
            recipe_tag="test",
            robot_name="test",
            model="openai:other",
            dashboard_events=NullDashboardEventSink(),
            runtime=RuntimeConfig(llm=LLMConfig(provider="openai", model="selected")),
        )


def test_trace_ownership_lasts_until_toolkit_closes(tmp_path: Path):
    from roborsi.evaluation.trajectory import TrajectoryReader
    from roborsi.runtime.lifecycle import create_trace, finish_trace

    toolkit = SimpleNamespace(state=EnvState(tmp_path), capture_video=False)
    recorder = create_trace(
        tmp_path, RuntimeConfig(), NullDashboardEventSink(), toolkit=toolkit
    )
    with recorder.activate():
        recorder.emit("agent_end", {"status": "completed"})
    toolkit.state.save("episode.mp4", b"video", step=None)
    finish_trace(recorder)
    data = TrajectoryReader(tmp_path)
    events = data.update().events
    assert events[-1]["type"] == "run_end"
    assert any(item["type"] == "artifact_written" for item in events[:-1])
    assert toolkit.capture_video


def test_trace_supports_tool_catalog_without_artifact_state(tmp_path: Path):
    from roborsi.evaluation.trajectory import TrajectoryReader
    from roborsi.runtime.lifecycle import close_toolkit_trace, create_trace

    closed = []
    toolkit = SimpleNamespace(close=lambda: closed.append(True))
    recorder = create_trace(
        tmp_path, RuntimeConfig(), NullDashboardEventSink(), toolkit=toolkit
    )
    close_toolkit_trace(toolkit, recorder)
    assert closed == [True]
    assert TrajectoryReader(tmp_path).index()["summary"]["status"] == "completed"


@pytest.mark.parametrize("failed", [False, True])
def test_finalization_preserves_published_media_state(tmp_path: Path, failed: bool):
    from roborsi.evaluation.trajectory import TrajectoryReader
    from roborsi.runtime.lifecycle import create_trace, finish_trace

    state = EnvState(tmp_path)
    state.save("episode.mp4", b"previous bytes", step=None)
    recorder = create_trace(tmp_path, RuntimeConfig(), NullDashboardEventSink())
    if failed:
        recorder.emit(
            "artifact_failed",
            {"artifact_ref": "episode.mp4", "error": "encoding failed"},
        )
    else:
        recorder.record_artifact(tmp_path / "episode.mp4")
    finish_trace(recorder)
    reader = TrajectoryReader(tmp_path)
    data = reader.index()
    media_events = [
        event
        for event in reader.projection.events
        if event["type"] in {"artifact_written", "artifact_failed"}
    ]
    assert len(media_events) == 1
    assert data["media"][0]["status"] == ("failed" if failed else "ready")
    assert data["media"][0]["available"] is not failed
