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

"""Queries resume one decision without advancing its observation sequence."""

import pytest

from roborsi.evaluation.adapter_service import AdapterFacade
from roborsi.evaluation.messages import (
    Action,
    BenchmarkQuery,
    Observation,
    QueryResult,
)
from roborsi.evaluation.service_protocol import (
    RoboRSIBackendError,
    RoboRSIRequestError,
    _decode,
)
from roborsi.evaluation.services import AdapterClient, AgentFacade
from tests.unit_tests.roborsi.evaluation.test_episode_stop import (
    Adapter,
    task,
    wire,
)
from tests.unit_tests.roborsi.evaluation.test_services import _service


def query(step=0):
    return BenchmarkQuery(
        benchmark="demo",
        episode_id="episode",
        step_index=step,
        query_id="plan-1",
        method="plan_eef",
        arguments={"arm": "left"},
    )


def reply(**updates):
    return QueryResult(
        benchmark="demo",
        episode_id="episode",
        step_index=0,
        query_id="plan-1",
        data={"status": "Fail"},
    ).model_copy(update=updates)


class QueryAdapter(Adapter):
    def __init__(self, invalid=False):
        super().__init__()
        self.queries = 0
        self.invalid = invalid

    def query(self, request):
        self.queries += 1
        return reply(step_index=int(self.invalid))


class QueryAgent:
    def __init__(self):
        self.calls = self.resumes = self.closes = 0

    def reset(self, task):
        pass

    def act(self, observation):
        self.calls += 1
        return query(observation.step_index)

    def resume(self, result):
        self.resumes += 1
        return Action(
            benchmark=result.benchmark,
            episode_id=result.episode_id,
            step_index=result.step_index,
            data={},
        )

    def close(self):
        self.closes += 1


def test_query_replay_keeps_current_step_and_rejects_stale_identity():
    adapter = QueryAdapter()
    facade = AdapterFacade(adapter)
    facade.start(wire(task(), "start"))
    for request in (query(1), query().model_copy(update={"episode_id": "wrong"})):
        with pytest.raises(RoboRSIRequestError):
            # Encode the other identity consistently, so the facade checks it.
            from roborsi.evaluation.service_protocol import _encode

            facade.query(
                _encode(request, request.benchmark, request.episode_id, "invalid")
            )
    assert adapter.queries == 0
    answer = facade.query(wire(query(), "query"))
    assert _decode(answer, QueryResult) == reply()
    assert facade.query(wire(query(), "query")) == answer
    with pytest.raises(RoboRSIRequestError, match="reused"):
        facade.query(wire(query().model_copy(update={"method": "other"}), "query"))
    assert adapter.queries == 1
    facade.close()


def test_invalid_query_result_is_closed_and_error_is_replayed():
    adapter = QueryAdapter(invalid=True)
    facade = AdapterFacade(adapter)
    facade.start(wire(task(), "start"))
    for _ in range(2):
        with pytest.raises(RoboRSIBackendError, match="different step"):
            facade.query(wire(query(), "query"))
    assert adapter.queries == adapter.closes == 1


def test_agent_resumes_once_and_only_an_action_advances_its_sequence():
    agent = QueryAgent()
    facade = AgentFacade(agent)
    facade.reset(wire(task(), "reset"))
    observation = Observation(benchmark="demo", episode_id="episode", step_index=0)
    first = facade.act(wire(observation, "act"))
    assert _decode(first, BenchmarkQuery) == query()
    assert facade.act(wire(observation, "act")) == first
    with pytest.raises(RoboRSIRequestError, match="pending query"):
        facade.act(wire(observation, "extra-act"))
    for updates in ({"step_index": 1}, {"query_id": "other"}):
        with pytest.raises(RoboRSIRequestError, match="different step or query ID"):
            facade.resume(wire(reply(**updates), "invalid-reply"))
    decision = facade.resume(wire(reply(), "resume"))
    assert _decode(decision, Action).step_index == 0
    assert facade.resume(wire(reply(), "resume")) == decision
    with pytest.raises(RoboRSIRequestError, match="no pending"):
        facade.resume(wire(reply(), "extra-reply"))
    with pytest.raises(RoboRSIRequestError, match="out of order"):
        facade.act(wire(observation, "stale-act"))
    facade.act(wire(observation.model_copy(update={"step_index": 1}), "next-act"))
    assert agent.calls == 2 and agent.resumes == 1
    facade.close()


def test_resume_error_does_not_repeat_agent_work():
    class FailingAgent(QueryAgent):
        def resume(self, result):
            self.resumes += 1
            raise RuntimeError("provider failed")

    agent = FailingAgent()
    facade = AgentFacade(agent)
    facade.reset(wire(task(), "reset"))
    facade.act(
        wire(Observation(benchmark="demo", episode_id="episode", step_index=0), "act")
    )
    for _ in range(2):
        with pytest.raises(RoboRSIBackendError, match="provider failed"):
            facade.resume(wire(reply(), "resume"))
    assert agent.resumes == agent.closes == 1


def test_legacy_adapter_rejects_query_and_remains_usable():
    with _service("adapter") as rpc:
        client = AdapterClient(rpc)
        client.start(task(), "episode", timeout_s=3)
        with pytest.raises(RoboRSIRequestError, match="does not advertise"):
            client.query(query(), timeout_s=3)
        step = client.step(
            Action(
                benchmark="demo",
                episode_id="episode",
                step_index=0,
                data={"native": {"__ndarray__": "literal-field"}},
            ),
            timeout_s=3,
        )
        assert step.terminated and step.observation.step_index == 1
        client.close()
