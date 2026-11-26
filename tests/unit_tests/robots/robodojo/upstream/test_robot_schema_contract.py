"""Model-visible robot constraints match validation on both CLI call paths."""
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

from jsonschema import Draft202012Validator, ValidationError
import pytest

P = 'robots.robodojo.agents.submission.'
profile = import_module(P+'execution_profile')
cci = import_module(P+'cci_protocol')
anthropic = import_module(P+'anthropic_messages')
POLICY = Path(__import__('robots.robodojo.agents.submission', fromlist=['__file__']).__file__).parent


@pytest.mark.parametrize('vla_backend', ['simate_beta', 'pi05', 'disabled'])
@pytest.mark.parametrize('direct_namespace', [False, True])
def test_registered_constraints_reach_direct_js_and_provider_schemas(
        tmp_path, reference, vla_backend, direct_namespace):
    frozen = profile.load_skillpack_profile(POLICY, 'align_blocks', tmp_path,
                                            vla_backend=vla_backend)
    run = tmp_path/'toolkit'
    for name in ('workspace', 'observations'):
        (run/name).mkdir(parents=True)

    def no_native_call(request):
        raise AssertionError('Schema validation must never execute an action')

    toolkit = profile.start_toolkit(SimpleNamespace(request=no_native_call), run,
                                    tmp_path, dict(vla_backend=vla_backend))
    specs = toolkit.get_tools_spec()
    registered = {s['name']: s['input_schema'] for s in specs}
    wrapper = profile.RobotCodeModeProfile(frozen, specs)
    # Reproduce the lossy CLI projection, independently of the fixed wrapper.
    def projected(value):
        if isinstance(value, dict):
            return {k: projected(v) for k,v in value.items()
                    if k not in ('maxItems', 'minimum', 'maximum', 'minLength', 'maxLength', 'default')}
        if isinstance(value, list):
            return [projected(v) for v in value]
        return value
    namespace = projected(wrapper.robot_namespace)
    request = dict(model=anthropic.MODEL, reasoning=dict(effort='xhigh'), input=[
        dict(type='additional_tools', tools=[namespace] if direct_namespace else [
            dict(type='namespace', name='functions', tools=[])]),
        dict(role='developer', content=[dict(type='input_text', text=frozen.core)]),
        dict(role='user', content=[dict(type='input_text', text='Synthetic schema fixture.')])])
    if frozen.base_rules:
        request['input'].insert(1, dict(role='developer', content=[
            dict(type='input_text', text=frozen.base_rules)]))
    original = deepcopy(request)
    result, evidence = wrapper.transform(request)
    assert request == original

    robot = next(g for g in result['input'][0]['tools'] if g['name']=='mcp__robot')
    actual = {s['name']: s['parameters'] for s in robot['tools']}
    assert actual == registered
    assert evidence['robot_registered_schemas_restored'] == sorted(frozen.expected_tools)
    assert evidence['robot_native_namespace_restored_for_code_mode'] == ([] if direct_namespace else [0])
    assert ('vla_propose' in actual) is (vla_backend != 'disabled')
    for spec in robot['tools']:
        assert spec['description'] == next(t['description'] for t in specs if t['name']==spec['name'])

    # Test both production provider encodings, not just an intermediate dict.
    encoded = cci.request_to_cci(result)
    wire = {t['name'].split('.')[-1]:t['parameters'] for t in encoded['tools']
            if t['name'].startswith('mcp__robot.')}
    assert wire == registered
    native, registry = anthropic.request_to_messages(result, {})
    bywire = {s['wire_name']:s['full_name'].split('.')[-1] for s in registry}
    assert {bywire[t['name']]:t['input_schema'] for t in native['tools']
            if bywire[t['name']] in registered} == registered

    assert 'localize' not in actual and 'spatial_memory' not in actual
    pixel = dict(camera_id='cam_head', observation_id='fixture:00000_spatial_case',
                 pixel=[1,1], coordinate_frame='environment_origin',
                 reference_point='visible_contact', known_height_m=0)
    arm = dict(position=[0,0,0], pitch_deg=0, roll_deg=90, yaw_deg=0, gripper_opening=1)
    act = dict(targets=[dict(left=arm, right=arm)]*50, max_native_steps=160,
               reason='Synthetic fixture.', assessment=dict(current_subgoal='Hold.'))
    Draft202012Validator(actual['spatial_localize_pixel']).validate(pixel)
    Draft202012Validator(actual['act']).validate(act)
    for name, bad in [
        ('spatial_localize_pixel', {**pixel, 'pixel':[1,1,1]}),
        ('spatial_localize_pixel', {**pixel, 'pixel_uncertainty_px':0}),
        ('spatial_localize_pixel', {**pixel, 'height_uncertainty_m':-1}),
        ('spatial_localize_pixel', {**pixel, 'coordinate_frame':'camera'}),
        ('spatial_localize_pixel', {**pixel, 'target_plane':dict(
            normal=[0,0,1], b=0, coordinate_frame='environment_origin', uncertainty_m=0)}),
        ('act', {**act, 'targets':act['targets']+[act['targets'][0]]}),
        ('act', {**act, 'max_native_steps':161}),
        ('act', {**act, 'reason':''}),
        ('act', {**act, 'targets':[dict(left={**arm,'roll_deg':91}, right=arm)]}),
        ('act', {**act, 'targets':[dict(left={**arm,'position':[0,0,0,0]}, right=arm)]}),
    ]:
        with pytest.raises(ValidationError):
            Draft202012Validator(actual[name]).validate(bad)
    # A downstream encoder/caller cannot mutate the registered next-turn schema.
    actual['spatial_localize_pixel']['properties']['pixel']['maxItems'] = 99
    fresh, _ = wrapper.transform(request)
    local = next(s for g in fresh['input'][0]['tools'] if g['name']=='mcp__robot'
                 for s in g['tools'] if s['name']=='spatial_localize_pixel')
    assert local['parameters']['properties']['pixel']['maxItems'] == 2
