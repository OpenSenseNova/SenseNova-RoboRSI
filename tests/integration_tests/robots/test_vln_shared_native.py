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

"""Opt-in native Habitat parity checks; no model calls or historical writes."""

import base64
import os
from pathlib import Path

import pytest

from roborsi.evaluation.benchmark_client import BenchmarkClient
from roborsi.evaluation.media import CaseMediaChannel
from roborsi.evaluation.messages import Action
from robots.vln_shared.benchmark import NavigationBenchmark, serve
from robots.vln_shared.native_process import NativeProcess
from robots.vln_shared.storage import read_json


@pytest.fixture(scope="module")
def campaign_config():
    path = os.environ.get("ROBORSI_VLN_CONFIG")
    if not path:
        pytest.skip("set ROBORSI_VLN_CONFIG to a prepared campaign with native assets")
    return read_json(Path(path))


@pytest.mark.timeout(180)
@pytest.mark.parametrize("dataset", ["lhpr_vln", "r2r", "rxr"])
@pytest.mark.parametrize("split", ["train", "test"])
def test_native_http_matches_direct_actions(campaign_config, tmp_path, dataset, split):
    section = campaign_config["datasets"][dataset][split]
    episode = section["episodes"][2]
    steps = []
    provider = NavigationBenchmark(
        section["spec"], [episode], 5, on_progress=steps.append
    )
    provider.runtime_info()
    task = provider.tasks()[0]
    channel = CaseMediaChannel(tmp_path / "client/media")
    with serve(provider, tmp_path / "service") as endpoint:
        client = BenchmarkClient(endpoint)
        try:
            observation = client.start(task, "parity", timeout_s=120).observation
            assert steps == [observation.data["physical_steps"]] == [0]
            client.collect_observation(observation, channel, timeout_s=30)
            initial = [channel.read(ref, "parity", 0) for ref in observation.media]
            requests = 0
            while True:
                action = "turn_left" if requests == 0 else "stop"
                result = client.step(
                    Action(
                        benchmark=task.benchmark,
                        episode_id="parity",
                        step_index=observation.step_index,
                        data={"actions": [action]},
                    ),
                    timeout_s=120,
                )
                requests += 1
                observation = result.observation
                assert steps[-1] == observation.data["physical_steps"]
                if result.terminated or result.truncated:
                    break
            native = client.finish(timeout_s=30)
            assert steps[-1] == native.details["steps"]
            client.collect_native(native, tmp_path / "client", timeout_s=30)
        finally:
            client.close()
    direct = tmp_path / "direct"
    direct.mkdir()
    worker = NativeProcess(section["spec"]["python"], direct, timeout=120)
    try:
        first = worker.call(
            "start",
            spec={
                **section["spec"],
                **task.native,
                "seed": task.seed,
                "max_steps": task.max_steps,
            },
        )
        assert initial == [base64.b64decode(v) for v in first["images"].values()]
        for i in range(requests):
            worker.call("step", action="turn_left" if i == 0 else "stop")
        raw = worker.call("finish")
        assert raw["metrics"] == native.details["metrics"]
        assert raw["steps"] == native.details["steps"]
        assert raw["provenance"] == native.details["provenance"]
        assert raw["success"] is native.success
    finally:
        worker.close()
