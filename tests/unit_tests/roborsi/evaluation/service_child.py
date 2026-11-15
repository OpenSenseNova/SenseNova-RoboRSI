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

"""Subprocess fixtures for the RoboRSI HTTP service integration test."""

from __future__ import annotations

import json
import os
from pathlib import Path

from roborsi.evaluation.media import CaseMediaChannel
from roborsi.evaluation.messages import (
    Action,
    BenchmarkRuntimeInfo,
    NativeResult,
    Observation,
    StartResult,
    StepResult,
    Task,
)


class DemoAgent:
    def __init__(self) -> None:
        self.task: Task | None = None

    def reset(self, task: Task) -> None:
        self.task = task

    def act(self, observation: Observation) -> Action:
        assert self.task is not None
        return Action(
            benchmark=observation.benchmark,
            episode_id=observation.episode_id,
            step_index=observation.step_index,
            data={"native": {"__ndarray__": "literal-field"}},
        )

    def close(self) -> None:
        self.task = None


class DemoAdapter:
    def __init__(self) -> None:
        self.task: Task | None = None
        self.episode_id: str | None = None

    def start(self, task: Task, episode_id: str) -> StartResult:
        self.task = task
        self.episode_id = episode_id
        return StartResult(
            task=task,
            observation=Observation(
                benchmark=task.benchmark,
                episode_id=self.episode_id,
                step_index=0,
                data={"state": [0.0]},
            ),
            capabilities=task.capabilities,
        )

    def step(self, action: Action) -> StepResult:
        assert action.data == {"native": {"__ndarray__": "literal-field"}}
        return StepResult(
            observation=Observation(
                benchmark=action.benchmark,
                episode_id=action.episode_id,
                step_index=action.step_index + 1,
                data={"state": [1.0]},
            ),
            terminated=True,
            truncated=False,
            termination_reason="native_success",
        )

    def finish(self) -> NativeResult:
        assert self.task is not None and self.episode_id is not None
        return NativeResult(
            task=self.task,
            episode_id=self.episode_id,
            success=True,
            score=0.75,
            details={"native_metric": 0.75},
        )

    def close(self) -> None:
        self.task = None
        self.episode_id = None


def make_demo_agent(case_dir: Path) -> DemoAgent:
    assert case_dir.is_dir()
    return DemoAgent()


class FeedbackDemoAgent(DemoAgent):
    def observe_end(self, step: StepResult) -> None:
        Path(os.environ["ROBORSI_TEST_FEEDBACK"]).write_text(json.dumps(step.as_dict()))


def make_feedback_agent(case_dir: Path) -> FeedbackDemoAgent:
    return FeedbackDemoAgent()


class FailingAgent(DemoAgent):
    def act(self, observation: Observation) -> Action:
        raise RuntimeError("demo agent action failed")


def make_failing_agent(case_dir: Path) -> FailingAgent:
    assert case_dir.is_dir()
    return FailingAgent()


def make_demo_adapter() -> DemoAdapter:
    return DemoAdapter()


class VersionedDemoAdapter(DemoAdapter):
    def runtime_info(self) -> BenchmarkRuntimeInfo:
        return BenchmarkRuntimeInfo(
            benchmark="demo",
            benchmark_version="demo-1",
            benchmark_commit="a" * 40,
            adapter_version="adapter-1",
            evaluation={"score_metric": "native"},
            platform={"machine": "test-cpu"},
        )


def make_versioned_demo_adapter() -> VersionedDemoAdapter:
    return VersionedDemoAdapter()


class PromptScoreAdapter(DemoAdapter):
    """Score the pinned Agent prompt through the normal Adapter boundary."""

    def step(self, action: Action) -> StepResult:
        self.prompt = action.data["prompt"]
        return StepResult(
            observation=Observation(
                benchmark=action.benchmark,
                episode_id=action.episode_id,
                step_index=action.step_index + 1,
            ),
            terminated=True,
            truncated=False,
            termination_reason="episode_complete",
        )

    def finish(self) -> NativeResult:
        assert self.task is not None and self.episode_id is not None
        success = self.prompt == "version two"
        return NativeResult(
            task=self.task,
            episode_id=self.episode_id,
            success=success,
            score=float(success),
        )


def make_prompt_score_adapter() -> PromptScoreAdapter:
    return PromptScoreAdapter()


class FailingAdapter(DemoAdapter):
    def start(self, task: Task, episode_id: str) -> StartResult:
        raise RuntimeError("demo adapter start failed")


def make_failing_adapter() -> FailingAdapter:
    return FailingAdapter()


class MediaDemoAgent(DemoAgent):
    def __init__(self, channel: CaseMediaChannel) -> None:
        super().__init__()
        self.channel = channel

    def act(self, observation: Observation) -> Action:
        assert len(observation.media) == 2
        assert (
            self.channel.read(observation.media[0], observation.episode_id, 0)
            == b"front"
        )
        assert (
            self.channel.read(observation.media[1], observation.episode_id, 0)
            == b"wrist"
        )
        return super().act(observation)


def make_media_demo_agent(case_dir: Path) -> MediaDemoAgent:
    assert case_dir.is_dir()
    return MediaDemoAgent(CaseMediaChannel(os.environ["ROBORSI_MEDIA_DIR"]))


class MediaDemoAdapter(DemoAdapter):
    def __init__(self, channel: CaseMediaChannel) -> None:
        super().__init__()
        self.channel = channel

    def start(self, task: Task, episode_id: str) -> StartResult:
        result = super().start(task, episode_id)
        assert self.episode_id is not None
        front = self.channel.write(self.episode_id, 0, b"front", "image/png")
        wrist = self.channel.write(self.episode_id, 0, b"wrist", "image/jpeg")
        return StartResult(
            task=result.task,
            observation=result.observation.model_copy(update={"media": (front, wrist)}),
            capabilities=result.capabilities,
        )


def make_media_demo_adapter() -> MediaDemoAdapter:
    return MediaDemoAdapter(CaseMediaChannel(os.environ["ROBORSI_MEDIA_DIR"]))


class MediaDemoBenchmark:
    name = "demo"

    def runtime_info(self):
        return VersionedDemoAdapter().runtime_info()

    def tasks(self):
        return []

    def assets(self):
        return []

    def create_adapter(self, directory, artifacts):
        return MediaDemoAdapter(CaseMediaChannel(directory / "media"))


def make_media_demo_benchmark():
    return MediaDemoBenchmark()


class PromptBenchmark:
    """A second provider built only from the published Benchmark interface."""

    name = "demo"

    def tasks(self):
        return [
            Task(
                benchmark=self.name,
                task_id="place",
                seed=0,
                instruction="Place the object",
                max_steps=1,
            )
        ]

    def runtime_info(self):
        return (
            VersionedDemoAdapter()
            .runtime_info()
            .model_copy(
                update={
                    "evaluation": {
                        "score_metric": "native_score",
                        "success_metric": "native_success",
                        "aggregation": "case_mean",
                        "error_denominator": "expected",
                    }
                }
            )
        )

    def assets(self):
        return []

    def create_adapter(self, directory, artifacts):
        return PromptScoreAdapter()


def make_prompt_benchmark():
    return PromptBenchmark()


class SdkBenchmark(PromptBenchmark):
    """Two model actions with initial, intermediate and terminal camera feedback."""

    def tasks(self):
        return [
            super()
            .tasks()[0]
            .model_copy(
                update={
                    "max_steps": 2,
                    "capabilities": {
                        "action_schema": {
                            "type": "object",
                            "properties": {"prompt": {"type": "string"}},
                            "required": ["prompt"],
                            "additionalProperties": False,
                        }
                    },
                }
            )
        ]

    def create_adapter(self, directory, artifacts):
        return SdkAdapter(CaseMediaChannel(directory / "media"))


class SdkAdapter(PromptScoreAdapter):
    def __init__(self, channel):
        super().__init__()
        self.channel = channel

    def _media(self, observation):
        import io

        from PIL import Image

        data = io.BytesIO()
        Image.new("RGB", (2, 2), (observation.step_index, 0, 0)).save(
            data, format="PNG"
        )
        reference = self.channel.write(
            observation.episode_id, observation.step_index, data.getvalue(), "image/png"
        )
        return observation.model_copy(update={"media": (reference,)})

    def start(self, task, episode_id):
        result = super().start(task, episode_id)
        return result.model_copy(
            update={"observation": self._media(result.observation)}
        )

    def step(self, action):
        result = super().step(action)
        ended = action.step_index == 1
        return result.model_copy(
            update={
                "observation": self._media(result.observation),
                "terminated": ended,
                "termination_reason": "episode_complete" if ended else None,
            }
        )


def make_sdk_benchmark():
    return SdkBenchmark()
