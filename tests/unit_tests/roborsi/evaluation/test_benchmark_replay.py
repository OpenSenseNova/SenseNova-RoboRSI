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

"""A lost HTTP reply can be retried without executing a native action twice."""

import pytest

from roborsi.evaluation.benchmark_client import BenchmarkClient
from roborsi.evaluation.messages import Action
from roborsi.utils.rpc import RpcTransportError
from tests.unit_tests.roborsi.evaluation.http_fault_proxy import lost_responses
from tests.unit_tests.roborsi.evaluation.test_benchmark_http import server, task


@pytest.mark.parametrize("mode", ["truncate", "disconnect", "bad_status"])
def test_typed_client_retries_lost_start_step_and_finish_replies(tmp_path, mode):
    with server(tmp_path) as (provider, _, direct):
        with lost_responses(direct.transport.endpoint, mode=mode) as (
            endpoint,
            requests,
        ):
            client = BenchmarkClient(endpoint)
            assert client.start(task(), "one", timeout_s=3).task == task()
            step = client.step(
                Action(benchmark="demo", episode_id="one", step_index=0, data={}),
                timeout_s=3,
            )
            assert step.observation.step_index == 1
            assert client.finish(timeout_s=3).score == 0.75
            calls = [r for r in requests if r["method"] == "POST"]
            assert len(calls) == 6
            for first, retry in zip(calls[::2], calls[1::2], strict=True):
                assert first["dropped"] and not retry["dropped"]
                assert first["body"] == retry["body"]
                assert first["body"]["request_id"]
            assert provider.steps == 1 and provider.closed == 1
            assert len(client.artifacts(kind="log")) == 3
            client.close()


def test_client_stops_after_one_retry_when_both_responses_are_lost(tmp_path):
    with server(tmp_path) as (provider, _, direct):
        with lost_responses(
            direct.transport.endpoint, paths=("/actions",), failures_per_request=2
        ) as (endpoint, requests):
            client = BenchmarkClient(endpoint)
            client.start(task(), "one", timeout_s=3)
            with pytest.raises(RpcTransportError):
                client.step(
                    Action(benchmark="demo", episode_id="one", step_index=0, data={}),
                    timeout_s=3,
                )
            actions = [r for r in requests if r["path"].endswith("/actions")]
            assert len(actions) == 2 and actions[0]["body"] == actions[1]["body"]
            assert provider.steps == 1
            assert client.status("one")["step_index"] == 1
            client.close()


def test_truncated_artifact_is_not_saved_as_a_complete_file(tmp_path):
    with server(tmp_path) as (_, _, direct):
        direct.start(task(), "one", timeout_s=3)
        info = direct.artifacts(kind="image")[0]
        with lost_responses(
            direct.transport.endpoint, paths=(f"/artifacts/{info.artifact_id}",)
        ) as (endpoint, requests):
            client = BenchmarkClient(endpoint)
            with pytest.raises(RpcTransportError):
                client.download(info, tmp_path / "download")
            assert len(requests) == 1
            assert not (tmp_path / "download").exists()
        direct.close()
