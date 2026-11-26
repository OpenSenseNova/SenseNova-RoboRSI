"""Step ACKs and tool images are synthetic; no physics or provider calls."""
from copy import deepcopy
from importlib import import_module
import base64
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import pytest


POLICY = Path(__import__('robots.robodojo.agents.submission', fromlist=['__file__']).__file__).parent
profile = import_module("robots.robodojo.agents.submission.execution_profile")


@pytest.fixture
def runtime(reference):
    # Load the verified native event writer using the normal offline bootstrap.
    return profile.runtime_modules(POLICY)


def target(opening=0., x=.2):
    return {arm: dict(position=[x, 0., .8], quaternion_wxyz=[1., 0., 0., 0.],
                      gripper_opening=opening) for arm in ("left", "right")}


def owner(feedback):
    observation = dict(control_step=0, state={})
    for arm in ("left", "right"):
        observation["state"][arm + "_ee_pose"] = [0., 0., .8, 1., 0., 0., 0.]
        observation["state"][arm + "_ee_joint_state"] = [.25]
    result = SimpleNamespace(latest=dict(observation=observation, done=False),
                             deadline=time.monotonic() + 60, segment_done=False,
                             fault=None, vla_pending=None, lock=threading.RLock(), calls=[])
    result.public = deepcopy(result.latest)
    result.public["manual_tcp_targets_enabled"] = True

    def native_request(unused_owner, req):
        result.calls.append(deepcopy(req))
        assert req["op"] == "act" and req["repeat"] == 1
        result.latest = deepcopy(result.latest)
        obs = result.latest["observation"]
        obs["control_step"] += 1
        step = obs["control_step"]
        action = req["action"]
        if action["mode"] == "eef":
            for arm in ("left", "right"):
                goal = action["target"][arm]
                obs["state"][arm + "_ee_pose"] = [*goal["position"], *goal["quaternion_wxyz"]]
        for arm, value in feedback(step).items():
            obs["state"][arm + "_ee_joint_state"] = [value]
        obs["cameras"] = {cam: f"/observations/{step}_{cam}.png"
                          for cam in ("cam_head", "cam_left_wrist", "cam_right_wrist")}
        result.public = deepcopy(result.latest)
        result.public["manual_tcp_targets_enabled"] = True
        return result.public

    return result, native_request


def run(runtime, state, native_request, path, tmp_path, *, steps=10, targets=None):
    assessment = dict(reason="Check the carried object.", assessment={"current_subgoal": "Carry."})
    if path == "tcp":
        return runtime.execution.execute_tcp(state, dict(targets=targets or [target()],
            **assessment), native_request, invalid_action_type=ValueError)
    state.active = {"record_dir": tmp_path}
    state.vla_pending = dict(proposal_id="fixture", execution_mode="stored_vla_joint",
                            control_step=state.latest["observation"]["control_step"],
                            targets=[target() for _ in range(50)],
                            out_of_bounds=[[] for _ in range(50)])
    pending, original = state.vla_pending, deepcopy(state.vla_pending)
    result = runtime.execution.execute_vla(state, dict(proposal_id="fixture", approved_steps=steps,
        **assessment), native_request, invalid_action_type=ValueError)
    assert state.vla_pending is None
    # The accepted prefix is read as stored, never rewritten to match feedback.
    assert pending == original
    return result


@pytest.mark.parametrize("path", ["tcp", "vla"])
@pytest.mark.parametrize("arm", ["left", "right"])
@pytest.mark.parametrize("changed", [.21, .29])
def test_interrupts_on_first_changed_step_in_either_arm(runtime, tmp_path, path, arm, changed):
    state, native = owner(lambda step: {arm: .25 if step < 4 else changed})
    result = run(runtime, state, native, path, tmp_path)
    assert result["status"] == "gripper_change_reobserve"
    assert result["executed_steps"] == len(state.calls) == 4
    assert result["observation"]["control_step"] == 4
    change, = result["gripper_event"]["changes"]
    assert change["arm"] == arm and change["commanded_opening"] == 0
    assert change["reference_opening"] == .25 and change["measured_opening"] == changed
    assert change["direction"] == ("opening" if changed > .25 else "closing")
    assert result["observation"]["cameras"]["cam_head"] == "/observations/4_cam_head.png"
    assert result["done"] is False and state.segment_done is False
    if path == "vla":
        assert [call["action"]["index"] for call in state.calls] == list(range(4))
        record = json.loads((tmp_path / "vla_review.jsonl").read_text().splitlines()[-1])
        assert record["remaining_queue_cancelled"] and record["status"] == result["status"]
        with pytest.raises(ValueError, match="Current proposal_id"):
            runtime.execution.execute_vla(state, {"proposal_id": "fixture"}, native,
                                          invalid_action_type=ValueError)


@pytest.mark.parametrize("path", ["tcp", "vla"])
def test_gradual_drift_is_compared_with_fixed_reference(runtime, tmp_path, path):
    openings = [.25, .25, .25, .245, .240, .235, .229]
    state, native = owner(lambda step: {"left": openings[min(step - 1, 6)]})
    result = run(runtime, state, native, path, tmp_path)
    assert result["status"] == "gripper_change_reobserve" and len(state.calls) == 7
    assert result["gripper_event"]["changes"][0]["reference_opening"] == .25


@pytest.mark.parametrize("first,second", [("tcp", "tcp"), ("tcp", "vla"), ("vla", "tcp")])
def test_settled_reference_persists_between_act_paths(runtime, tmp_path, first, second):
    state, native = owner(lambda step: {"left": .25 if step <= 12 else .20})
    initial = run(runtime, state, native, first, tmp_path, steps=12)
    assert initial["status"] in ("reached", "executed")
    assert initial["executed_steps"] == 12
    result = run(runtime, state, native, second, tmp_path)
    assert result["status"] == "gripper_change_reobserve" and result["executed_steps"] == 1
    assert len(state.calls) == 13


@pytest.mark.parametrize("path", ["tcp", "vla"])
def test_normal_closure_and_small_measurement_noise_do_not_interrupt(runtime, tmp_path, path):
    openings = [.9, .7, .5, .25, .251, .25]
    state, native = owner(lambda step: {"left": openings[step - 1] if step <= 6 else .25})
    result = run(runtime, state, native, path, tmp_path)
    assert result["status"] in ("reached", "executed")
    assert "gripper_event" not in result


def test_intentional_release_rearms_and_other_arm_remains_monitored(runtime):
    Monitor = import_module("gripper_monitor").GripperMonitor
    monitor = Monitor()
    obs = {"control_step": 0, "state": {arm + "_ee_joint_state": [.25] for arm in ("left", "right")}}

    def step(commands, left, right=.25):
        monitor.prepare(obs, commands)
        obs["control_step"] += 1
        obs["state"]["left_ee_joint_state"] = [left]
        obs["state"]["right_ee_joint_state"] = [right]
        return monitor.sample(obs)

    for _ in range(3):
        assert step(target(), .25) is None
    release = target()
    release["left"]["gripper_opening"] = 1.
    for left in (.4, .6, .8, 1., 1., 1.):
        assert step(release, left) is None
    event = step(release, 1., .20)
    assert [change["arm"] for change in event["changes"]] == ["right"]
    event = step(release, .96, .20)
    assert [change["arm"] for change in event["changes"]] == ["left"]


@pytest.mark.parametrize("path", ["tcp", "vla"])
def test_interruption_returns_current_three_rgb_images_through_public_act(runtime, tmp_path, path):
    from PIL import Image
    image_root = tmp_path / "observations"
    image_root.mkdir()
    for cam in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
        Image.new("RGB", (2, 2), (11, 22, 33)).save(image_root / f"4_{cam}.png")
    state, native = owner(lambda step: {"left": .251234567 if step < 4 else .211111111})
    state.active = {"record_dir": tmp_path}
    state.vla_pending = None
    if path == "vla":
        state.vla_pending = dict(proposal_id="fixture", control_step=0,
                                execution_mode="stored_vla_joint",
                                targets=[target() for _ in range(50)],
                                out_of_bounds=[[] for _ in range(50)])

    def hybrid_request(unused, req):
        assert req == {"op": "observe"}
        return deepcopy(state.public)

    request = runtime.native_request.make_request(hybrid_request, native,
        invalid_action_type=ValueError, allow_direct_targets=True)
    tools = runtime.robot_tools.RobotTools(lambda req: request(state, req),
        setting="simate_vla", image_root=image_root)
    arguments = dict(reason="Carry the object.", assessment={"current_subgoal": "Carry."})
    if path == "vla":
        arguments.update(proposal_id="fixture", steps=10)
    else:
        arguments["targets"] = [{arm: runtime.robot_tools.link6_to_tcp_target(value)
                                 for arm, value in target().items()}]
    result = tools.execute_tool("act", arguments)
    assert result.result["status"] == "gripper_change_reobserve"
    assert result.result["observation"]["control_step"] == 4
    assert result.result["observation"]["state"]["left_ee_joint_state"] == [.211]
    assert state.latest["observation"]["state"]["left_ee_joint_state"] == [.211111111]
    images = [block for block in result.content_blocks if block["type"] == "image"]
    assert len(images) == 3
    assert base64.b64decode(images[0]["source"]["data"]) == (image_root / "4_cam_head.png").read_bytes()
    assert len(state.calls) == 4


def test_owned_native_adapter_stages_the_monitor_dependency(runtime, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "entry.py").write_text("# owned native entry\n")
    (source / "hybrid_adapter.py").write_text("class Adapter: pass\n")
    config = {"adapter": "/original/entry.py:Adapter"}
    runtime.execution.stage_owned_adapter(source, config, POLICY / "skillpack/robot_runtime")
    assert (source / "gripper_monitor.py").read_bytes() == (POLICY / "skillpack/robot_runtime/gripper_monitor.py").read_bytes()


def test_rereading_feedback_does_not_arm_or_quantize_monitor(runtime):
    monitor = import_module("gripper_monitor").GripperMonitor()
    obs = {"control_step": 0, "state": {arm + "_ee_joint_state": [.251234567]
                                      for arm in ("left", "right")}}
    for _ in range(10):
        monitor.prepare(obs, target())
        assert monitor.sample(obs) is None
    # Two real samples still do not arm: repeated reads did not count.
    for step in (1, 2):
        monitor.prepare(obs, target())
        obs["control_step"] = step
        assert monitor.sample(obs) is None
    monitor.prepare(obs, target())
    obs["control_step"] = 3
    obs["state"]["left_ee_joint_state"] = [.01]
    assert monitor.sample(obs) is None
    # A fresh settled reference survives a tool boundary without rounding.
    for step in (4, 5):
        monitor.prepare(obs, target())
        obs["control_step"] = step
        assert monitor.sample(obs) is None
    monitor.prepare(obs, target())
    obs["control_step"] = 6
    obs["state"]["right_ee_joint_state"] = [.231134567]
    event = monitor.sample(obs)
    change, = event["changes"]
    assert change["arm"] == "right" and change["reference_opening"] == .251234567


@pytest.mark.parametrize("changed,interrupts", [(.230000001, False), (.23, True), (.27, True)])
def test_threshold_boundary(runtime, changed, interrupts):
    monitor = import_module("gripper_monitor").GripperMonitor()
    obs = {"control_step": 0, "state": {arm + "_ee_joint_state": [.25]
                                      for arm in ("left", "right")}}
    for step in range(1, 5):
        monitor.prepare(obs, target())
        obs["control_step"] = step
        obs["state"]["left_ee_joint_state"] = [.25 if step < 4 else changed]
        event = monitor.sample(obs)
    assert (event is not None) == interrupts
