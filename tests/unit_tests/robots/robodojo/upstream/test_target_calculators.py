"""Read-only target calculations use current state; only a later act can move."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from decimal import Decimal
from importlib import import_module
from pathlib import Path
from types import MethodType

import pytest

profile = import_module("robots.robodojo.agents.submission.execution_profile")
POLICY = Path(__import__('robots.robodojo.agents.submission', fromlist=['__file__']).__file__).parent
runtime = profile.runtime_modules(POLICY)
tools = runtime.robot_tools
tcp = import_module("tcp_pose")
CALCULATORS = ("hold_current", "close_in_place", "move_relative")


class Bridge:
    def __init__(self):
        self.calls = []
        state = {}
        for arm, y, opening in (("left", -.15, .123456789), ("right", .15, .987654321)):
            pose = tcp.tcp_to_link6_pose([.10049, y, .8, *tcp.angles_to_quat(12, 7, 30)])
            state[arm+"_ee_pose"] = pose
            state[arm+"_ee_joint_state"] = [opening]
        self.value = dict(observation=dict(control_step=17, state=state, review_due=True),
                          pending_vla_proposal=dict(proposal_id="pending-native-id"), done=False)

    def __call__(self, request):
        self.calls.append(deepcopy(request))
        assert request == {"op": "observe"}, "A calculator attempted an operation"
        return deepcopy(self.value)


def make():
    bridge = Bridge()
    return bridge, tools.RobotTools(bridge, setting="simate_vla")


def assert_precision(value):
    if isinstance(value, float):
        assert Decimal(str(value)).as_tuple().exponent >= -3
    elif isinstance(value, list):
        for item in value:
            assert_precision(item)
    elif isinstance(value, dict):
        for item in value.values():
            assert_precision(item)


@pytest.mark.parametrize("name,args", [
    ("hold_current", {}), ("close_in_place", {"arm": "left"}),
    ("move_relative", {"arm": "right", "delta_m": [0, 0, .03]}),
])
def test_calculators_read_once_without_motion_review_or_proposal_changes(name, args):
    bridge, toolkit = make()
    original = deepcopy(bridge.value)
    result = toolkit.execute_tool(name, args)
    assert set(result.result) == {"control_step", "computed_only", "targets"}
    assert result.result["computed_only"] is True and result.result["control_step"] == 17
    assert len(result.result["targets"]) == 1
    assert set(result.result["targets"][0]) == {"left", "right"}
    assert bridge.calls == [{"op": "observe"}]
    assert bridge.value == original
    assert result.content_blocks and all(block["type"] == "text" for block in result.content_blocks)
    assert_precision(result.result)


def test_hold_current_converts_native_link6_to_public_tcp_and_keeps_openings():
    bridge, toolkit = make()
    result = toolkit.dispatch("hold_current", {})["targets"][0]
    for arm in ("left", "right"):
        expected = tcp.link6_to_public_pose(bridge.value["observation"]["state"][arm+"_ee_pose"])
        expected["gripper_opening"] = bridge.value["observation"]["state"][arm+"_ee_joint_state"][0]
        assert result[arm] == tools._round_display(expected)
    assert result["left"]["position"] == [.1, -.15, .8]


@pytest.mark.parametrize("arm,opening", [("left", None), ("right", .333333333)])
def test_close_in_place_changes_only_selected_opening(arm, opening):
    _, toolkit = make()
    before = toolkit.dispatch("hold_current", {})["targets"][0]
    args = dict(arm=arm)
    if opening is not None:
        args["opening"] = opening
    result = toolkit.dispatch("close_in_place", args)["targets"][0]
    expected = deepcopy(before)
    expected[arm]["gripper_opening"] = 0. if opening is None else .333
    assert result == expected


def test_move_relative_uses_environment_axes_and_full_precision_before_rounding():
    bridge, toolkit = make()
    before = toolkit.dispatch("hold_current", {})["targets"][0]
    delta = [.00049, .02, .03]
    result = toolkit.dispatch("move_relative", dict(arm="left", delta_m=delta))["targets"][0]
    # A rounded observe position of .100 plus .00049 would incorrectly remain .100.
    assert result["left"]["position"] == [.101, -.13, .83]
    assert result["right"] == before["right"]
    assert {k:v for k,v in result["left"].items() if k != "position"} == {
        k:v for k,v in before["left"].items() if k != "position"}
    assert bridge.value["observation"]["state"]["left_ee_joint_state"] == [.123456789]


def test_each_call_reads_latest_state_and_binds_its_control_step():
    bridge, toolkit = make()
    first = toolkit.dispatch("hold_current", {})
    state = bridge.value["observation"]["state"]
    state["left_ee_pose"][2] += .1
    bridge.value["observation"]["control_step"] = 18
    second = toolkit.dispatch("hold_current", {})
    assert second["control_step"] == 18 and first["control_step"] == 17
    assert second["targets"][0]["left"]["position"][2] == .9
    assert len(bridge.calls) == 2


@pytest.mark.parametrize("name,args", [
    ("hold_current", {"steps": 5}), ("close_in_place", {}),
    ("close_in_place", {"arm": "both"}),
    ("close_in_place", {"arm": "left", "opening": 1.1}),
    ("close_in_place", {"arm": "left", "opening": True}),
    ("move_relative", {"arm": "left", "delta_m": [0, 0]}),
    ("move_relative", {"arm": "left", "delta_m": [0, 0, float("nan")]}),
    ("move_relative", {"arm": "left", "delta_m": [0, 0, float("inf")]}),
    ("move_relative", {"arm": "left", "delta_m": [0, 0, "0.1"]}),
])
def test_invalid_input_is_rejected_before_any_bridge_call(name, args):
    bridge, toolkit = make()
    with pytest.raises(tools.ToolInputError):
        toolkit.dispatch(name, args)
    assert bridge.calls == []


def test_nonfinite_measured_state_is_rejected_without_executing():
    bridge, toolkit = make()
    bridge.value["observation"]["state"]["left_ee_pose"][0] = float("nan")
    with pytest.raises(tools.ToolInputError, match="Latest measured state"):
        toolkit.dispatch("hold_current", {})
    assert bridge.calls == [{"op": "observe"}]


def test_later_act_is_required_for_native_execution_and_acknowledgements(experiment, backend, reference):
    fixtures = import_module("tests.unit_tests.robots.robodojo.upstream.test_hybrid_backend")
    backend._adapter.__class__ = runtime.execution.adapter_class(type(backend._adapter))
    experiment.request = MethodType(runtime.native_request.make_request(
        reference.hybrid_host.request, reference.hybrid_host.ORIGINAL,
        invalid_action_type=reference.InvalidAction, allow_direct_targets=True), experiment)
    toolkit = tools.RobotTools(experiment.request, setting="simate_vla")
    result = toolkit.dispatch("move_relative", dict(arm="left", delta_m=[.001, 0, 0]))
    assert backend.control_step == 0 and result["computed_only"] is True
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(toolkit.dispatch, "act", dict(targets=result["targets"],
            reason="Execute the calculated target.", assessment=dict(current_subgoal="Move left by 1 mm.")))
        try:
            for step in range(2):
                pending = backend.next_action(timeout=2)
                assert not future.done() and backend.control_step == step
                backend.acknowledge(pending.request_id, fixtures.observation())
            executed = future.result(timeout=2)
        finally:
            if not future.done():
                backend.cancel()
    assert executed["executed_steps"] == backend.control_step == 2


@pytest.mark.parametrize("budget,waypoints,expected_steps,expected_status", [
    (1, 1, 1, "native_step_budget_exhausted"),
    (160, 1, 2, "reached"),
    (160, 3, 6, "reached"),
])
def test_observe_and_hold_read_without_advancing_then_act_advances_real_acks(
        experiment, backend, reference, budget, waypoints, expected_steps, expected_status):
    """A TCP budget is a ceiling, not an instruction to wait that many steps."""
    fixtures = import_module("tests.unit_tests.robots.robodojo.upstream.test_hybrid_backend")
    backend._adapter.__class__ = runtime.execution.adapter_class(type(backend._adapter))
    experiment.request = MethodType(runtime.native_request.make_request(
        reference.hybrid_host.request, reference.hybrid_host.ORIGINAL,
        invalid_action_type=reference.InvalidAction, allow_direct_targets=True), experiment)
    toolkit = tools.RobotTools(experiment.request, setting="simate_vla")
    for _ in range(5):
        observed = toolkit.dispatch("observe", {})
        held = toolkit.dispatch("hold_current", {})
        assert observed["observation"]["control_step"] == held["control_step"] == backend.control_step == 0
        assert held["computed_only"] is True
    # Repeating measured-pose targets is explicit action execution; calculations
    # and observation above have not queued or acknowledged a physics step.
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(toolkit.dispatch, "act", dict(
            targets=deepcopy(held["targets"])*waypoints, max_native_steps=budget,
            reason="Hold the current measured poses while simulation advances.",
            assessment=dict(current_subgoal="Observe the synthetic demonstration.")))
        try:
            for step in range(expected_steps):
                pending = backend.next_action(timeout=2)
                assert not future.done() and backend.control_step == step
                backend.acknowledge(pending.request_id, fixtures.observation())
            executed = future.result(timeout=2)
        finally:
            if not future.done():
                backend.cancel()
    assert executed["status"] == expected_status
    assert executed["executed_steps"] == backend.control_step == expected_steps
