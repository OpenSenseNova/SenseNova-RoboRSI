"""Exercise deferred public motion, native solving, real joint ACKs and replay guards.

Robot geometry is synthetic; this verifies the bridge contract, not GPU solving.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from importlib import import_module
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

fixtures = import_module('tests.unit_tests.robots.robodojo.upstream.test_hybrid_backend')
reference, robot = fixtures.reference, fixtures.robot
bridge = import_module('robots.robodojo.agents.submission.native_curobo')
backend_module = import_module('robots.robodojo.agents.submission.hybrid_backend')
model_module = import_module('robots.robodojo.agents.submission.model')
deploy = import_module('robots.robodojo.agents.submission.deploy')
profile = import_module('robots.robodojo.agents.submission.execution_profile')
P = Path(__import__('robots.robodojo.agents.submission', fromlist=['__file__']).__file__).parent


@pytest.fixture(autouse=True)
def offline_action_dimensions(monkeypatch):
    monkeypatch.setattr(bridge, "get_robot_action_dim_info", lambda _: {
        "arm_dim": [6, 6], "ee_dim": [1, 1]})


@pytest.fixture
def native(reference, robot, tmp_path):
    kinclass, _ = bridge.runtime()
    specs = json.loads(robot.metadata.read_text())['arms']
    robots = [SimpleNamespace(type='target', arm_name=a+'_arm', robot_name=a,
        arm_joints_name=specs[a]['joint_names'], urdf_path=str(robot.urdf),
        base_link='base_link', ee_link_name='link6', arm_joint_indices=list(range(6))) for a in specs]
    joints = {a: np.zeros(6) for a in specs}
    openings = {a: 1. for a in specs}
    manager = SimpleNamespace(robot_list=robots,
        robot_key=[SimpleNamespace(data=SimpleNamespace(soft_joint_pos_limits=
            backend_module._CpuArray(np.array(specs[a]['soft_joint_pos_limits'])[None]))) for a in specs],
        ik_solver={a: SimpleNamespace(ee_link='link6') for a in specs},
        control_manager=SimpleNamespace(prev_control=[{}]))
    manager.get_joint = lambda r, ids=None: {0: joints[r.robot_name].copy()}
    manager.get_link_pose = lambda r, link, is_relative: {0: specs[r.robot_name]['root_pose']}
    manager.process_name = lambda name: name+'_joint_state'
    env = SimpleNamespace(robot_manager=manager, take_action_cnt=[0], end_flag=[False],
                          sim=SimpleNamespace(_sim_step_counter=0))
    kin = kinclass(env)
    def pose(a):
        t = kin.root(a)@kin.fk[a].matrix(joints[a])
        return np.r_[t[:3, 3], Rotation.from_matrix(t[:3, :3]).as_quat()[[3, 0, 1, 2]]]
    manager.get_real_endpose = lambda r: {0: pose(r.robot_name)}
    calls, commands = [], []
    def solve(target, env_idx, robot, trans):
        assert trans == 'world' and env_idx == 0
        calls.append(robot.robot_name)
        q = joints[robot.robot_name].copy(); q[1] += .02
        return {'status': 'Success', 'joint_names': robot.arm_joints_name, 'joint_value': q.tolist()}
    manager.solve_ik = solve
    def observation():
        value = fixtures.observation()
        for a in specs:
            value['state'][a+'_arm_joint_state'] = joints[a].copy()
            value['state'][a+'_ee_joint_state'] = np.array([openings[a]])
            value['state'][a+'_ee_pose'] = pose(a)
        return value
    def step(command):
        commands.append(deepcopy(command))
        for a in specs:
            joints[a] = np.array(command[a+'_arm_joint_state'])
            openings[a] = command[a+'_ee_joint_state'][0]
            manager.control_manager.prev_control[0][a+'_arm_joint_state'] = {'position': joints[a].tolist()}
        env.take_action_cnt[0] += 1; env.sim._sim_step_counter += 10
        return {'observation': {'state': {k: v.tolist() for k, v in observation()['state'].items()}}, 'done': False}
    # Stage the runtime wrapper before constructing the public facade, matching
    # production MRO: PublicAdapter -> runtime Adapter -> reference adapter.
    reference = profile.extend_reference(reference)
    directory = tmp_path/'episode'; directory.mkdir()
    backend = fixtures.HybridBackend(reference, task_name='align_blocks', episode_dir=directory,
        initial_observation=observation(), step_limit=200, metadata_path=robot.metadata,
        expected_metadata_sha256=fixtures.sha(robot.metadata), resource_root=robot.root,
        vla_port=19802, eef_execution_backend='bounded_curobo')
    backend.request(op='reset', episode={'task': 'align_blocks', 'robot': 'arx_x5'},
        record_dir=str(directory/'records'), observation_dir=str(directory/'observations'))
    action = {'mode': 'eef', 'assessment': {'current_subgoal': 'Synthetic bounded motion'},
        'reason': 'Verify native execution', 'target': {a: {'position': (pose(a)[:3]+[0, 0, .01]).tolist(),
        'quaternion_wxyz': pose(a)[3:].tolist(), 'gripper_opening': .2 if a == 'left' else .8} for a in specs}}
    yield SimpleNamespace(backend=backend, env=env, calls=calls, commands=commands, step=step,
                          observation=observation, action=action, directory=directory)
    backend.close()


def test_public_motion_reaches_native_solver_and_model_ack_once(native):
    b = native.backend
    with ThreadPoolExecutor(max_workers=1) as pool:
        work = pool.submit(b.request, op='step', action=native.action)
        model = object.__new__(model_module.Model)
        model._obs, model._ended, model._pending, model._step = native.observation(), False, None, 0
        model._connector = SimpleNamespace(next_action=b.next_action, acknowledge=b.acknowledge)
        model._awaiting_feedback = False
        request = model.get_action()[0]
        assert request['_roborsi_control'] == 'bounded_curobo'
        assert not native.calls and b.control_step == 0 and not work.done()
        result = bridge.execute_control(native.env, request, step_joint=native.step,
            initial_observation=native.observation(), record_dir=native.directory)
        assert len(native.calls) == 2 and len(native.commands) == 1
        assert native.env.take_action_cnt[0] == 1
        assert native.commands[0]['left_ee_joint_state'] == [.2]
        assert native.commands[0]['right_ee_joint_state'] == [.8]
        model._obs = {**native.observation(), 'roborsi_execution': result['roborsi_execution']}
        assert model.get_action() == []
        assert model._step == b.control_step == 1
        returned = work.result(timeout=2)
        assert returned['applied_action']['joint_command'] == native.commands[0]
        assert returned['applied_action']['execution_backend'] == 'harness_bounded_curobo'
        receipt = json.loads((b._adapter.record_dir/'bounded_curobo_execution.jsonl').read_text())
        assert receipt['receipt']['physics_steps'] == 10
        assert b._adapter._review_aid_observation['control_step'] == 1
        runtime = profile.runtime_modules(P)
        observed = []
        def read_acknowledged_observation(request):
            assert request == {'op': 'observe'}
            observed.append(deepcopy(request))
            return deepcopy(returned)
        tools = runtime.robot_tools.RobotTools(read_acknowledged_observation,
                                               setting='simate_vla')
        for name, args in [('hold_current', {}),
                           ('close_in_place', {'arm': 'left', 'opening': .1}),
                           ('move_relative', {'arm': 'right', 'delta_m': [0, 0, .01]})]:
            calculated = tools.dispatch(name, args)
            assert calculated['computed_only'] is True
            assert calculated['control_step'] == 1
            assert len(calculated['targets']) == 1
        assert native.env.take_action_cnt[0] == b.control_step == 1
        assert native.env.sim._sim_step_counter == 10
        assert len(native.calls) == 2 and len(native.commands) == 1
        assert observed == [{'op': 'observe'}]*3
        before = len(native.calls)
        with pytest.raises(ValueError, match='stale'):
            bridge.execute_control(native.env, request, step_joint=native.step,
                initial_observation=native.observation())
        assert len(native.calls) == before and len(native.commands) == 1


@pytest.mark.parametrize('defect', ['request_id', 'digest', 'step', 'physics', 'command'])
def test_bad_ack_cannot_release_or_advance_pending_motion(native, defect):
    b = native.backend
    with ThreadPoolExecutor(max_workers=1) as pool:
        work = pool.submit(b.request, op='step', action=native.action)
        pending = b.next_action(timeout=2)
        request = bridge.envelope(pending.request_id, pending.step_index, pending.action)
        result = bridge.execute_control(native.env, request, step_joint=native.step,
            initial_observation=native.observation())
        bad = deepcopy(result['roborsi_execution'])
        if defect == 'request_id': bad['request_id'] += 1
        elif defect == 'digest': bad['action_sha256'] = '0'*64
        elif defect == 'step': bad['receipt']['control_step_after'] += 1
        elif defect == 'physics': bad['receipt']['physics_steps'] = 0
        else: bad['applied_action']['joint_command']['left_ee_joint_state'] = [.5]
        with pytest.raises(ValueError, match='cuRobo'):
            b.acknowledge(pending.request_id, {**native.observation(), 'roborsi_execution': bad})
        assert b.control_step == 0 and not work.done()
        b.acknowledge(pending.request_id,
            {**native.observation(), 'roborsi_execution': result['roborsi_execution']})
        assert work.result(timeout=2)['applied_action']['execution_backend'] == 'harness_bounded_curobo'
        assert b.control_step == 1 and len(native.commands) == 1


def test_native_fault_after_physics_is_not_reexecuted(native):
    request = bridge.envelope(1, 0, native.action)
    def fault(command):
        native.step(command)
        raise RuntimeError('post-physics fixture failure')
    with pytest.raises(RuntimeError, match='do not resend'):
        bridge.execute_control(native.env, request, step_joint=fault,
            initial_observation=native.observation(), record_dir=native.directory)
    assert len(native.commands) == 1 and native.env.take_action_cnt[0] == 1
    receipt = json.loads((native.directory/'bounded_curobo_execution.jsonl').read_text())
    assert receipt['status'] == 'fault' and receipt['physics_steps'] == 10


def test_standard_deploy_uses_native_solver_then_unmodified_joint_passthrough(native):
    env = native.env
    seen, reads = [], []
    env.take_action = native.step
    env.is_episode_end = lambda: env.take_action_cnt[0] == 2
    def observe():
        reads.append(env.take_action_cnt[0])
        return native.observation()
    env.get_obs = observe
    direct = {'left_arm_joint_state': [.2]*6, 'right_arm_joint_state': [.1]*6,
              'left_ee_joint_state': [.35], 'right_ee_joint_state': [.65]}
    actions = [[bridge.envelope(1, 0, native.action)], [direct]]
    class Client:
        def call(self, func_name, obs=None):
            if func_name == 'get_action':return actions.pop(0)
            if func_name in ('update_obs', 'end_episode'):seen.append((func_name, obs))
    deploy.eval_one_episode(env, Client())
    assert len(native.calls) == 2 and len(native.commands) == 2
    assert native.commands[-1] == direct
    assert reads == [0, 1, 2]
    assert seen[1][1]['roborsi_execution']['receipt']['control_step_after'] == 1
    assert seen[-1][0] == 'end_episode' and 'roborsi_execution' not in seen[-1][1]
