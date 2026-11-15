# Copyright 2026 The SenseNova-RoboRSI Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native scoring, identity, information boundary and cleanup regressions."""

import pytest

from roborsi.evaluation.messages import Action, EpisodeStop
from robots.dexjoco.benchmark_service import make_task
from robots.dexjoco.contracts import ALIGNMENT
from robots.dexjoco.roborsi_adapter import DexJocoAdapter

from .conftest import FakeNative


def create(tmp_path, upstream, name="water_plant"):
    adapter = DexJocoAdapter(tmp_path / "case", upstream, native_factory=FakeNative)
    start = adapter.start(make_task(upstream, name, 100000), "episode")
    return adapter, start


def action(index=0, row=None, episode="episode"):
    return Action(
        benchmark="dexjoco",
        episode_id=episode,
        step_index=index,
        data={"vector": row if row is not None else [0.0] * 22},
    )


@pytest.mark.parametrize("success", [False, True])
def test_native_termination_is_authoritative(tmp_path, upstream, success):
    adapter, start = create(tmp_path, upstream)
    env = adapter.env
    env.succeed = success
    assert len(start.observation.data["state"]) == 23
    assert "seed" not in start.observation.data
    for index in range(3):
        result = adapter.step(action(index))
        assert result.reward is None
        assert result.native == {}
    assert result.terminated
    native = adapter.finish()
    assert native.success is success
    assert native.details["success_by_1x"] is success
    assert native.details["success_by_10x"] is success
    assert adapter.finish() == native
    with pytest.raises(ValueError):
        adapter.step(action(3))
    adapter.close()
    adapter.close()
    assert env.closed == 1


def test_invalid_action_does_not_advance(tmp_path, upstream):
    adapter, _ = create(tmp_path, upstream)
    for message in (action(episode="foreign"), action(1), action(row=[0.0] * 21)):
        with pytest.raises(ValueError):
            adapter.step(message)
    assert adapter.env.steps == 0
    adapter.stop(
        EpisodeStop(
            benchmark="dexjoco",
            episode_id="episode",
            step_index=0,
            reason="invalid_action",
        )
    )
    result = adapter.finish()
    assert not result.success
    assert result.details["outcome"] == "policy_failure"
    adapter.close()


def test_click_mouse_alignment_cannot_be_skipped(tmp_path, upstream):
    adapter, _ = create(tmp_path, upstream, "click_mouse")
    with pytest.raises(ValueError, match="alignment"):
        adapter.step(action())
    adapter.step(action(row=ALIGNMENT))
    assert adapter.steps == 1
    adapter.close()


def test_mutated_task_rejected_before_environment(tmp_path, upstream):
    task = make_task(upstream, "water_plant", 100000)
    adapter = DexJocoAdapter(tmp_path / "case", upstream, native_factory=FakeNative)
    with pytest.raises(ValueError, match="catalog"):
        adapter.start(task.model_copy(update={"instruction": "changed"}), "episode")
    assert adapter.env is None


def test_startup_failure_closes_environment(tmp_path, upstream):
    created = []

    class Broken(FakeNative):
        def reset(self):
            created.append(self)
            raise RuntimeError("reset failed")

    adapter = DexJocoAdapter(tmp_path / "case", upstream, native_factory=Broken)
    with pytest.raises(RuntimeError, match="reset failed"):
        adapter.start(make_task(upstream, "water_plant", 100000), "episode")
    assert created[0].closed == 1


def test_continuation_checkpoint_stays_private_and_is_published(
    tmp_path, upstream, monkeypatch
):
    import json

    from roborsi.evaluation.http_artifacts import ArtifactPublisher
    from robots.dexjoco.contracts import LIMITS

    monkeypatch.setitem(LIMITS, "water_plant", 2)
    adapter, _ = create(tmp_path, upstream)
    env = adapter.env
    env.checkpoint = lambda: b"private simulator checkpoint"
    adapter.artifacts = ArtifactPublisher(adapter.directory, "episode")
    adapter.step(action(0))
    result = adapter.step(action(1))
    assert not result.terminated and not result.truncated
    assert adapter.env is env and env.steps == 2
    assert result.observation.data["remaining_steps"] == 18
    checkpoint = adapter.native_dir / "checkpoint_simulator_1x.npz"
    assert checkpoint.read_bytes() == b"private simulator checkpoint"
    assert json.loads((adapter.native_dir / "checkpoint_1x.json").read_text()) == {
        "step": 2,
        "success_by_1x": False,
        "native_terminal": False,
    }
    assert all("checkpoint" not in key for key in result.observation.data)
    env.succeed = True
    adapter.step(action(2))
    native = adapter.finish()
    assert not native.details["success_by_1x"]
    assert native.details["success_by_10x"]
    refs = {artifact.ref for artifact in adapter.artifacts.list()}
    assert refs == {
        "native/trajectory.jsonl",
        "native/horizon.json",
        "native/checkpoint_1x.json",
        "native/checkpoint_simulator_1x.npz",
    }
    adapter.close()


def test_failed_cleanup_can_retry_without_reopening_episode(tmp_path, upstream):
    from roborsi.evaluation.adapter_service import AdapterFacade

    class RetryCloseNative(FakeNative):
        pending_close_error = True

        def close(self):
            if self.pending_close_error:
                self.pending_close_error = False
                raise RuntimeError("native cleanup interrupted")
            super().close()

    adapter = DexJocoAdapter(
        tmp_path / "case", upstream, native_factory=RetryCloseNative
    )
    adapter.start(make_task(upstream, "water_plant", 100000), "episode")
    env = adapter.env
    facade = AdapterFacade(adapter)
    with pytest.raises(RuntimeError, match="cleanup interrupted"):
        facade.close_episode()
    assert env.closed == 0
    with pytest.raises(ValueError, match="not active"):
        adapter.step(action())
    facade.close_episode()
    facade.close_episode()
    assert env.closed == 1
    assert adapter.env is None
