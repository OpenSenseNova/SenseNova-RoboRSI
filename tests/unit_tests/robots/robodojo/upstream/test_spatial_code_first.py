"""Localization and TCP preference must preserve native VLA recovery paths."""
import json
from importlib import import_module
from pathlib import Path
import pytest

execution = import_module('robots.robodojo.agents.submission.execution_profile')
POLICY = Path(__import__('robots.robodojo.agents.submission', fromlist=['__file__']).__file__).parent


@pytest.fixture
def toolkit(tmp_path, monkeypatch):
    rt = execution.runtime_modules(POLICY).robot_tools
    workspace = tmp_path/'workspace'
    workspace.mkdir()
    latest = {'observation': {'control_step': 12, 'state': {
        'left_ee_pose': [0, 0, 1, 1, 0, 0, 0],
        'right_ee_pose': [2, 0, 1, 1, 0, 0, 0]}},
        'manual_tcp_targets_enabled': True}
    calls = []
    sequence = 0

    def call(request):
        nonlocal sequence
        calls.append(request)
        if request['op'] == 'vla_propose':
            sequence += 1
            return {'proposal_id': f'native-{sequence}'}
        return latest

    monkeypatch.setattr(rt, 'sparse_proposal', lambda proposal: dict(proposal))
    tool = rt.RobotTools(call, setting='simate_vla', direct_act_enabled=True,
                         workspace_root=workspace, image_root=tmp_path, code_first_motion=True)
    return tool, latest, calls, workspace, rt


SUBGOAL = {'subgoal': 'Place the wooden toy car into the white basket.'}
ACT = {'proposal_id': 'p1', 'steps': 20, 'reason': 'Use the inspected recovery prefix.',
       'assessment': {'current_subgoal': 'Recover the basket approach.'}}


@pytest.mark.parametrize('memory_kind', ['absent', 'empty', 'stale', 'far', 'malformed'])
def test_vla_fallback_needs_only_visual_subgoal(toolkit, memory_kind):
    tool, latest, calls, workspace, rt = toolkit
    if memory_kind != 'absent':
        data = {'schema_version': 1, 'task': 'classify_objects',
                'frame': 'environment_origin', 'units': 'meters', 'objects': {}}
        if memory_kind in ('stale', 'far'):
            data['objects'] = {'car': {'roof': {'status': 'stale' if memory_kind == 'stale' else 'localized',
                'position_m': [10, 0, 0], 'control_step': 0}}}
        (workspace/'space.json').write_text('invalid JSON' if memory_kind == 'malformed' else json.dumps(data))
    assert tool.dispatch('vla_propose', SUBGOAL)['proposal_id'] == 'p1'
    assert calls[-1] == {'op': 'vla_propose', **SUBGOAL}
    tool.dispatch('act', ACT)
    assert calls[-1]['op'] == 'vla_review'
    assert calls[-1]['decision'] == 'accept'
    assert calls[-1]['approved_steps'] == 20


@pytest.mark.parametrize('steps', [5, 10, 15, 25, 50])
def test_native_prefix_range_is_not_capped_at_ten(toolkit, steps):
    tool, latest, calls, workspace, rt = toolkit
    tool.dispatch('vla_propose', SUBGOAL)
    tool.dispatch('act', {**ACT, 'steps': steps})
    assert calls[-1]['approved_steps'] == steps


def direct_action():
    pose = {'position': [0, 0, 1], 'pitch_deg': 0, 'roll_deg': 0,
            'yaw_deg': 0, 'gripper_opening': 1}
    return {'targets': [{'left': pose, 'right': pose}], 'reason': 'Refine from measured state.',
            'assessment': {'current_subgoal': 'Adjust the TCP.'}}


def test_tcp_vla_tcp_switching_needs_no_localization_metadata(toolkit):
    tool, latest, calls, workspace, rt = toolkit
    tool.dispatch('act', direct_action())
    tool.dispatch('vla_propose', SUBGOAL)
    tool.dispatch('act', ACT)
    tool.dispatch('act', direct_action())
    assert [r['op'] for r in calls if r['op'] != 'observe'] == [
        'act_eef', 'vla_propose', 'vla_review', 'act_eef']


def test_motion_policy_advertises_fallback_and_native_range(toolkit):
    tool, latest, calls, workspace, rt = toolkit
    policy = tool.dispatch('observe', {})['motion_policy']
    assert policy['localization_required_for_vla'] is False
    assert policy['required_vla_fields'] == ['subgoal']
    assert policy['vla_step_range'] == [5, 50]
    assert 'handoff_distance_m' not in policy


@pytest.mark.parametrize('blocked', ['disabled', 'done', 'review_due'])
def test_native_permission_and_checkpoint_guards_remain(toolkit, blocked):
    tool, latest, calls, workspace, rt = toolkit
    if blocked == 'disabled':
        tool.setting = 'no_vla'
    elif blocked == 'done':
        latest['done'] = True
    else:
        latest['observation']['review_due'] = True
    with pytest.raises(rt.ToolInputError):
        tool.dispatch('vla_propose', SUBGOAL)
    assert not any(r['op'] in ('vla_propose', 'vla_review', 'act_eef') for r in calls)


def test_prefix_beyond_native_schema_range_is_rejected(toolkit):
    tool, latest, calls, workspace, rt = toolkit
    tool.dispatch('vla_propose', SUBGOAL)
    calls.clear()
    with pytest.raises(rt.ToolInputError):
        tool.dispatch('act', {**ACT, 'steps': 51})
    assert calls == []
