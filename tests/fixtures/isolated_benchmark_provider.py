# Copyright 2026 The RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Source-only provider copied into an isolated Benchmark runtime test."""

import hashlib
import threading

from roborsi.evaluation.media import CaseMediaChannel
from roborsi.evaluation.messages import (
    BenchmarkRuntimeInfo,
    NativeResult,
    Observation,
    StartResult,
    StepResult,
    Task,
)


def task():
    return Task(
        benchmark="demo", task_id="place", seed=7, instruction="Place", max_steps=2
    )


class Provider:
    name = "demo"

    def __init__(self):
        self.steps = 0
        self.closed = 0
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.fail = False

    def tasks(self):
        return [task()]

    def assets(self):
        return []

    def runtime_info(self):
        return BenchmarkRuntimeInfo(
            benchmark="demo",
            benchmark_version="demo-1",
            benchmark_commit="a" * 40,
            adapter_version="adapter-1",
            evaluation={
                "aggregation": "case_mean",
                "error_denominator": "expected",
                "score_metric": "native_score",
                "success_metric": "native_success",
            },
            platform={"machine": "test"},
        )

    def create_adapter(self, directory, artifacts):
        return Adapter(self, directory, artifacts)


class Adapter:
    def __init__(self, provider, directory, artifacts):
        self.provider, self.directory, self.artifacts = provider, directory, artifacts
        self.media = CaseMediaChannel(directory / "media")
        self.index = 0

    def observation(self):
        return Observation(
            benchmark="demo",
            episode_id=self.episode_id,
            step_index=self.index,
            media=(
                self.media.write(
                    self.episode_id, self.index, b"camera image", "image/png"
                ),
            ),
        )

    def start(self, task, episode_id):
        self.task, self.episode_id = task, episode_id
        return StartResult(task=task, observation=self.observation())

    def step(self, action):
        self.provider.entered.set()
        assert self.provider.release.wait(5)
        if self.provider.fail:
            raise RuntimeError("do not expose /private/path or backend traceback")
        self.provider.steps += 1
        self.index += 1
        return StepResult(
            observation=self.observation(),
            terminated=True,
            truncated=False,
            termination_reason="native_success",
        )

    def finish(self):
        source = self.directory / "native" / "score.json"
        source.parent.mkdir(exist_ok=True)
        source.write_bytes(b'{"score":0.75}')
        return NativeResult(
            task=self.task,
            episode_id=self.episode_id,
            success=True,
            score=0.75,
            source="native/score.json",
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        )

    def close(self):
        self.provider.closed += 1
