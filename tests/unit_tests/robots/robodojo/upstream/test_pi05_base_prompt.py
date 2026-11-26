"""Historical Base rules reach the CLI once; VLA receives live subgoal only."""
from copy import deepcopy
from importlib import import_module
import hashlib
import json
from pathlib import Path
import shutil

import pytest

runtime = import_module('robots.robodojo.agents.submission.runtime_config')
profiles = import_module('robots.robodojo.agents.submission.execution_profile')
P = Path(__import__('robots.robodojo.agents.submission', fromlist=['__file__']).__file__).parent
BASE_SHA='b173dd647540fc5ca626459b1b8ba33f93d7d2ce4fe2c8c0c067e04d1c932b48'


def test_pi05_cli_uses_complete_base_developer_instructions(tmp_path,monkeypatch):
    monkeypatch.setenv('ROBORSI_RUNTIME_STATE_DIR',str(tmp_path))
    config=runtime.load_agent_config({},require_key=False)
    rules=config.document['agent']['codex_cli']['config_overrides'].get('developer_instructions','')
    assert hashlib.sha256(rules.encode()).hexdigest()==BASE_SHA
    assert rules.count('WORDING EXAMPLES')==1


@pytest.mark.parametrize('backend',['simate_beta','disabled'])
def test_other_backends_do_not_receive_pi05_rules(tmp_path,monkeypatch,backend):
    monkeypatch.setenv('ROBORSI_RUNTIME_STATE_DIR',str(tmp_path))
    config=runtime.load_agent_config({'vla_backend':backend},require_key=False)
    assert 'developer_instructions' not in config.document['agent']['codex_cli']['config_overrides']


def wire(profile, rules):
    return {'input':[
        {'role':'developer','content':[{'type':'input_text','text':profile.core}]},
        {'role':'developer','content':[{'type':'input_text','text':rules}]},
        {'type':'additional_tools','tools':[{'name':'robot','type':'namespace','tools':deepcopy(profile.deferred)}]},
    ]}


def test_wire_preserves_exactly_one_independent_developer_block(tmp_path):
    profile=profiles.load_skillpack_profile(P,'align_blocks',tmp_path,vla_backend='pi05')
    rules=(P/'skillpack/common/pi05_base_prompt.txt').read_text()
    result,evidence=profile.transform(wire(profile,rules))
    texts=[p.get('text','') for item in result['input'] for p in item.get('content',[])]
    assert sum(text==rules for text in texts)==1
    assert profile.core==(P/'skillpack/common/agent_core.md').read_text().strip()
    assert evidence['vla_backend']=='pi05'
    assert evidence['base_prompt_sha256']==BASE_SHA
    assert evidence['core_sha256']==hashlib.sha256((P/'skillpack/common/agent_core.md').read_bytes()).hexdigest()
    workspace=tmp_path/'workspace';workspace.mkdir()
    profile.stage_live_workspace(workspace,owner=tmp_path)
    profile.stage_live_workspace(workspace,owner=tmp_path)
    assert (workspace/'preparation/agent_system.md').read_text().strip()==profile.core


@pytest.mark.parametrize('damage',['missing','duplicate','system_role'])
def test_wire_rejects_missing_or_ambiguous_base_delivery(tmp_path,damage):
    profile=profiles.load_skillpack_profile(P,'align_blocks',tmp_path,vla_backend='pi05')
    rules=(P/'skillpack/common/pi05_base_prompt.txt').read_text()
    request=wire(profile,rules)
    if damage=='missing':request['input'].pop(1)
    if damage=='duplicate':request['input'].insert(1,deepcopy(request['input'][1]))
    if damage=='system_role':request['input'][1]['role']='system'
    with pytest.raises(ValueError,match='Base developer'):
        profile.transform(request)


def test_profile_cannot_be_reused_with_another_backend(tmp_path):
    profiles.load_skillpack_profile(P,'align_blocks',tmp_path,vla_backend='pi05')
    with pytest.raises(ValueError,match='backend|prompt'):
        profiles.load_skillpack_profile(P,'align_blocks',tmp_path,vla_backend='simate_beta')


@pytest.mark.parametrize('before,after',[
    ('pi05','disabled'),('simate_beta','disabled'),('disabled','pi05'),('disabled','simate_beta')])
def test_disabled_profile_cannot_cross_backend_on_recovery(tmp_path,before,after):
    profiles.load_skillpack_profile(P,'align_blocks',tmp_path,vla_backend=before)
    with pytest.raises(ValueError,match='backend|prompt'):
        profiles.load_skillpack_profile(P,'align_blocks',tmp_path,vla_backend=after)


def test_episode_configuration_reports_real_backend(tmp_path):
    profile=profiles.load_skillpack_profile(P,'align_blocks',tmp_path,vla_backend='pi05')
    cfg={'test_episodes':[{'task':'align_blocks','layout_id':14,'eval_seed':1}],
         'episode_seconds':None,'options':{'review_window_steps':1050,'review_max_windows':1}}
    data=json.loads(profiles.episode_prompt(profile,cfg,model_cfg={'vla_backend':'pi05'}).split('Episode configuration:\n')[1])
    assert data['vla_backend'] == 'pi05'
    assert data['native_step_limit'] == 1050
    assert 'layout' not in data and 'seed' not in data


def test_base_tamper_rejected_before_cli(tmp_path,monkeypatch):
    dest=tmp_path/'policy';shutil.copytree(P/'skillpack/common',dest/'skillpack/common')
    (dest/'skillpack/common/pi05_base_prompt.txt').write_text('tampered')
    with pytest.raises(ValueError,match='Base prompt'):
        runtime.load_pi05_base_prompt(dest)


def test_multiple_sentence_subgoal_reaches_original_pi05_wire(experiment, vla):
    subgoal='Grasp the board with the right gripper. Lift the board. Keep the board level during the lift.'
    proposal=experiment.request({'op':'vla_propose','subgoal':subgoal})
    assert len(vla.inputs)==1
    assert vla.inputs[0]['prompt']==subgoal
    assert set(vla.inputs[0])=={'state','images','prompt'}
    assert vla.inputs[0]['state'].shape==(14,)
    assert proposal['horizon']==50


def test_pi05_toolkit_allows_vla_and_keeps_calculations_read_only(tmp_path):
    runtime_modules=profiles.runtime_modules(P)
    calls=[]
    # The transport is external; a zero-motion observation is enough to verify
    # the toolkit forwards the exact instruction and does not infer on observe.
    def transport(request):
        calls.append(deepcopy(request))
        if request['op']=='observe':return {'done':False,'observation':{'control_step':0}}
        raise RuntimeError('Reached original VLA boundary')
    tools=runtime_modules.robot_tools.RobotTools(transport,setting='pi05_vla',image_root=tmp_path)
    tools.dispatch('observe',{})
    assert [c['op'] for c in calls]==['observe']
    text='Grasp the board. Lift the board. Keep it level during the lift.'
    with pytest.raises(RuntimeError,match='Reached original VLA boundary'):
        tools.dispatch('vla_propose',{'subgoal':text})
    assert calls[-1]=={'op':'vla_propose','subgoal':text}


def test_preflight_requires_registered_base_prompt(tmp_path):
    preflight=import_module('robots.robodojo.agents.submission.preflight')
    # A valid copy first, then remove only the source registration and re-pin
    # the outer envelope: preflight must still reject the missing inner pin.
    policy=tmp_path/'policy'
    shutil.copytree(P/'skillpack',policy/'skillpack')
    shutil.copy2(P/'submission.json',policy/'submission.json')
    frozen=policy/'skillpack/frozen_common.json'
    content=json.loads(frozen.read_text());content.pop('common/pi05_base_prompt.txt');frozen.write_text(json.dumps(content))
    submission=policy/'submission.json';doc=json.loads(submission.read_text())
    doc['skillpack']['frozen_common_sha256']=hashlib.sha256(frozen.read_bytes()).hexdigest();submission.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='Base prompt'):
        preflight.validate_skillpack(policy,'align_blocks')
