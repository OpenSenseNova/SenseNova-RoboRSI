"""Short public handles retain native proposal validation and recovery lineage."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

POLICY = Path(__import__('robots.robodojo.agents.submission', fromlist=['__file__']).__file__).parent


@pytest.fixture
def runtime():
    return import_module("robots.robodojo.agents.submission.execution_profile").runtime_modules(POLICY)


class Bridge:
    def __init__(self, error):
        self.error = error
        self.pending = None
        self.sequence = self.steps = 0
        self.calls = []

    def __call__(self, request):
        self.calls.append(deepcopy(request))
        if request["op"] == "vla_propose":
            self.sequence += 1
            arm = dict(position=[.1, .2, .3], quaternion_wxyz=[1, 0, 0, 0], gripper_opening=.25)
            self.pending = dict(proposal_id=f"{self.sequence:032x}",
                                targets=[deepcopy(dict(left=arm, right=arm)) for _ in range(50)])
            return deepcopy(self.pending)
        if request["op"] == "vla_review":
            if self.pending is None or request["proposal_id"] != self.pending["proposal_id"]:
                raise self.error("Current proposal_id required")
            if request["decision"] == "accept":
                self.steps += request["approved_steps"]
            self.pending = None
        return dict(observation=dict(control_step=self.steps), pending_vla_proposal=deepcopy(self.pending))


def act(proposal_id):
    return dict(proposal_id=proposal_id, steps=5, reason="Use the reviewed prefix.",
                assessment=dict(current_subgoal="Approach the object."))


def test_preview_observe_and_native_execution_use_the_same_proposal(runtime):
    bridge = Bridge(runtime.robot_tools.ToolInputError)
    toolkit = runtime.robot_tools.RobotTools(bridge, setting="simate_vla")
    preview = toolkit.dispatch("vla_propose", {"subgoal": "Approach the object."})
    stored = deepcopy(bridge.pending)
    assert preview["proposal_id"] == "p1" and len(preview["preview"]) == 10
    observed = toolkit.dispatch("observe", {})
    assert observed["pending_vla_proposal"]["proposal_id"] == "p1"
    assert stored["proposal_id"] not in json.dumps(observed)
    assert bridge.pending == stored and bridge.steps == 0
    toolkit.dispatch("act", act("p1"))
    assert bridge.calls[-1]["proposal_id"] == stored["proposal_id"]
    assert bridge.steps == 5
    with pytest.raises(runtime.robot_tools.ToolInputError, match="Current proposal_id"):
        toolkit.dispatch("act", act("p1"))
    assert bridge.steps == 5


def test_replacement_rejects_old_handle_and_accepts_new_one(runtime):
    bridge = Bridge(runtime.robot_tools.ToolInputError)
    toolkit = runtime.robot_tools.RobotTools(bridge, setting="simate_vla")
    toolkit.dispatch("vla_propose", {"subgoal": "Approach the object."})
    second = toolkit.dispatch("vla_propose", {"subgoal": "Move above the object."})
    assert second["proposal_id"] == "p2" and bridge.steps == 0
    assert [c["proposal_id"] for c in bridge.calls if c.get("decision") == "reject"] == [f"{1:032x}"]
    with pytest.raises(runtime.robot_tools.ToolInputError, match="Current proposal_id"):
        toolkit.dispatch("act", act("p1"))
    assert bridge.steps == 0
    toolkit.dispatch("act", act("p2"))
    assert bridge.steps == 5 and bridge.calls[-1]["proposal_id"] == f"{2:032x}"


@pytest.mark.parametrize('setting', ['simate_vla', 'pi05_vla'])
@pytest.mark.parametrize('steps,budget', [(5,5),(20,20),(20,80),(50,160),(None,80)])
def test_redundant_tcp_budget_warns_without_changing_vla_acceptance(runtime, setting, steps, budget):
    bridge = Bridge(runtime.robot_tools.ToolInputError)
    toolkit = runtime.robot_tools.RobotTools(bridge, setting=setting)
    toolkit.dispatch('vla_propose', {'subgoal':'Approach the object.'})
    arguments=act('p1')
    if steps is None:arguments.pop('steps')
    else:arguments['steps']=steps
    arguments['max_native_steps']=budget
    before=deepcopy(arguments)
    result=toolkit.dispatch('act',arguments)
    assert bridge.steps==bridge.calls[-1]['approved_steps']==(steps or 5)
    assert 'max_native_steps' not in bridge.calls[-1]
    assert result['warnings'][0]['code']=='parameter_ignored'
    assert result['warnings'][0]['parameter']=='max_native_steps'
    assert result['warnings'][0]['supplied_value']==budget
    assert arguments==before
    # A warning must not bypass consumed-proposal checks.
    with pytest.raises(runtime.robot_tools.ToolInputError,match='Current proposal_id'):
        toolkit.dispatch('act',arguments)
    assert bridge.steps==(steps or 5)


def test_conflicting_tcp_budget_rejects_without_consuming_vla_proposal(runtime):
    bridge = Bridge(runtime.robot_tools.ToolInputError)
    toolkit = runtime.robot_tools.RobotTools(bridge, setting='simate_vla')
    toolkit.dispatch('vla_propose', {'subgoal':'Approach the object.'})
    calls=deepcopy(bridge.calls);pending=deepcopy(bridge.pending)
    with pytest.raises(runtime.robot_tools.ToolInputError,match='smaller than'):
        toolkit.dispatch('act',{**act('p1'),'steps':20,'max_native_steps':10})
    assert bridge.calls==calls and bridge.pending==pending and bridge.steps==0
    toolkit.dispatch('act',act('p1'))
    assert bridge.steps==5


def test_reconnect_and_resume_keep_aliases_without_recycling(runtime, tmp_path):
    bridge = Bridge(runtime.robot_tools.ToolInputError)
    path = tmp_path/"proposal_ids.json"
    cls = runtime.robot_tools.RobotTools
    first = cls(bridge, setting="simate_vla", proposal_ids_path=path)
    first.dispatch("vla_propose", {"subgoal": "Approach the object."})
    second = cls(bridge, setting="simate_vla", proposal_ids_path=path)
    assert second.dispatch("observe", {})["pending_vla_proposal"]["proposal_id"] == "p1"
    second.dispatch("act", act("p1"))
    resumed_path = tmp_path/"resumed_ids.json"
    records = import_module('robots.robodojo.agents.submission.runtime_records')
    resumed = cls(bridge, setting="simate_vla", proposal_ids_path=resumed_path,
                  proposal_ids_seed=records.read(path))
    assert resumed_path.is_file()  # Seed persists even before the first resumed call.
    new = resumed.dispatch("vla_propose", {"subgoal": "Move above the object."})
    assert new["proposal_id"] == "p2"
    with pytest.raises(runtime.robot_tools.ToolInputError, match="Current proposal_id"):
        resumed.dispatch("act", act("p1"))
    assert bridge.steps == 5
    # Conversations created before short IDs can still use the actual native UUID.
    resumed.dispatch("act", act(bridge.pending["proposal_id"]))
    assert bridge.steps == 10


@pytest.mark.parametrize('budget', [None, 80])
def test_short_handles_preserve_real_native_rejection_and_step_acks(runtime, experiment, backend, reference, vla, budget):
    fixtures = import_module("tests.unit_tests.robots.robodojo.upstream.test_hybrid_backend")
    # Apply the same owned adapter/host extensions as the production profile.
    backend._adapter.__class__ = runtime.execution.adapter_class(type(backend._adapter))
    experiment.request = MethodType(runtime.native_request.make_request(
        reference.hybrid_host.request, reference.hybrid_host.ORIGINAL,
        invalid_action_type=reference.InvalidAction, allow_direct_targets=True), experiment)
    toolkit = runtime.robot_tools.RobotTools(experiment.request, setting="simate_vla")
    first = toolkit.dispatch("vla_propose", {"subgoal": "Approach the object."})
    second = toolkit.dispatch("vla_propose", {"subgoal": "Move above the object."})
    assert (first["proposal_id"], second["proposal_id"]) == ("p1", "p2")
    with pytest.raises(reference.InvalidAction, match="Current proposal_id"):
        toolkit.dispatch("act", act("p1"))
    assert backend.control_step == 0
    if budget is not None:
        with pytest.raises(reference.InvalidAction, match='prefix'):
            toolkit.dispatch('act',{**act('p2'),'steps':1,'max_native_steps':budget})
        assert backend.control_step==0
    with ThreadPoolExecutor(max_workers=1) as pool:
        arguments=act('p2')
        if budget is not None:arguments['max_native_steps']=budget
        future = pool.submit(toolkit.dispatch, "act", arguments)
        try:
            for step in range(5):
                pending = backend.next_action(timeout=2)
                assert backend.control_step == step and not future.done()
                backend.acknowledge(pending.request_id, fixtures.observation())
            result = future.result(timeout=2)
        finally:
            if not future.done():
                backend.cancel()
    assert result["executed_steps"] == backend.control_step == 5
    assert bool(result.get('warnings')) == (budget is not None)


@pytest.mark.parametrize("seed", [{"a": "p1", "b": "p1"}, {"a": "p0"}, {"a": "1"}])
def test_invalid_private_aliases_fail_before_any_robot_request(runtime, seed):
    bridge = Bridge(runtime.robot_tools.ToolInputError)
    with pytest.raises(ValueError, match="Invalid private proposal ID"):
        runtime.robot_tools.RobotTools(bridge, setting="simate_vla", proposal_ids_seed=seed)
    assert bridge.calls == []


def test_resume_binding_seeds_the_latest_attempt_mapping(monkeypatch, tmp_path):
    module = import_module("robots.robodojo.agents.submission.skillpack_agent")
    profile = import_module("robots.robodojo.agents.submission.execution_profile")
    old, latest = tmp_path/"old", tmp_path/"latest"
    for root, seed in [(old, {"a": "p1"}), (latest, {"a": "p1", "b": "p2"})]:
        ledger = root/"agent/controller/proposal_ids.json"
        ledger.parent.mkdir(parents=True);ledger.write_text(json.dumps(seed))
    seen = {}
    def start(*args, **kwargs):
        seen.update(kwargs)
        return object()
    monkeypatch.setattr(profile, "start_toolkit", start)
    context = SimpleNamespace(resume={"source": str(old), "workspace_source": str(latest)},
                              episode_dir=tmp_path, model_cfg={})
    agent, planner = SimpleNamespace(), SimpleNamespace()
    module.SkillpackEmbodiedAgent.bind_experiment(agent, context, planner, object(), tmp_path)
    assert seen["proposal_ids_seed"] == {"a": "p1", "b": "p2"}
    assert planner.toolkit is agent._skillpack_toolkit
