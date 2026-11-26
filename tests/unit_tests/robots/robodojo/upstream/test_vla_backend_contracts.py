"""Backend selection reaches installation, episode config and service ownership."""
from importlib import import_module
import json
from pathlib import Path

import pytest
import yaml

runtime = import_module('robots.robodojo.agents.submission.runtime_config')
rgb = import_module('robots.robodojo.agents.submission.native_rgb_episode')
pi05 = import_module('robots.robodojo.agents.submission.pi05_runtime')
P = Path(__import__('robots.robodojo.agents.submission', fromlist=['__file__']).__file__).parent


@pytest.mark.parametrize('cfg,expected', [
    ({}, {'backend':'pi05','checkpoint_path':'checkpoints/pi05-arx-x5-seed0-step59999','setting':'pi05_vla'}),
    ({'vla_backend':'simate_beta'}, {'backend':'simate_beta','checkpoint_path':'checkpoints/simate-beta-robodojo-eval','setting':'simate_vla'}),
    ({'checkpoint_path':'/weights/original/59999'}, {'backend':'pi05','checkpoint_path':'/weights/original/59999','setting':'pi05_vla'}),
    ({'vla_backend':'disabled','checkpoint_path':None}, {'backend':'disabled','checkpoint_path':None,'setting':'tcp_only'}),
])
def test_effective_backend_and_checkpoint(cfg, expected):
    assert runtime.resolve_vla_config(cfg) == expected


@pytest.mark.parametrize('backend', ['unknown', '', None])
def test_unknown_backend_rejected_before_startup(backend):
    with pytest.raises(ValueError, match='vla_backend'):
        runtime.resolve_vla_config({'vla_backend':backend})


def test_default_deployment_matches_published_agent():
    cfg = yaml.safe_load((P/'deploy.yml').read_text())
    assert cfg['vla_backend']=='simate_beta'
    assert cfg['checkpoint_path']=='checkpoints/simate-beta-robodojo-eval'
    assert cfg['skill_library_mode']=='general_only'
    assert cfg['eef_execution_backend_by_task']=={}


def test_rgb_case_without_shared_port_uses_owned_pi05(monkeypatch):
    monkeypatch.setenv('ROBORSI_SHARED_VLA_PORT','19806')
    monkeypatch.setenv('ROBORSI_SHARED_VLA_RECEIPT','/old/service.json')
    assert rgb.configure_vla({}) == {'vla_backend':'pi05','checkpoint_path':'checkpoints/pi05-arx-x5-seed0-step59999'}
    import os
    assert 'ROBORSI_SHARED_VLA_PORT' not in os.environ
    assert 'ROBORSI_SHARED_VLA_RECEIPT' not in os.environ


def test_rgb_case_selects_explicit_shared_service(monkeypatch):
    monkeypatch.delenv('ROBORSI_SHARED_VLA_PORT',raising=False)
    cfg=rgb.configure_vla({'vla_backend':'pi05','shared_vla_port':19802,
                           'shared_vla_receipt':'/owned/pi05/service-identity.json'})
    import os
    assert cfg['vla_backend']=='pi05'
    assert os.environ['ROBORSI_SHARED_VLA_PORT']=='19802'
    assert os.environ['ROBORSI_SHARED_VLA_RECEIPT']=='/owned/pi05/service-identity.json'


def test_disabled_rgb_cannot_inherit_or_attach_a_vla(monkeypatch):
    import os
    monkeypatch.setenv('ROBORSI_SHARED_VLA_PORT','19806')
    monkeypatch.setenv('ROBORSI_SHARED_VLA_RECEIPT','/old/service.json')
    assert rgb.configure_vla({'vla_backend':'disabled','shared_vla_port':'unused',
                              'checkpoint_path':'/unused/weights'}) == {
        'vla_backend':'disabled','checkpoint_path':None}
    assert 'ROBORSI_SHARED_VLA_PORT' not in os.environ
    assert 'ROBORSI_SHARED_VLA_RECEIPT' not in os.environ


@pytest.mark.parametrize('port',[True,0,65536,'wrong'])
def test_rgb_rejects_invalid_shared_port(port):
    with pytest.raises(ValueError,match='shared_vla_port'):
        rgb.configure_vla({'shared_vla_port':port})


def test_pi05_installer_resolves_selected_deployment(tmp_path, monkeypatch):
    cfg=tmp_path/'experiment.yaml'
    cfg.write_text(yaml.safe_dump({'vla_backend':'pi05','checkpoint_path':str(tmp_path/'weights')}))
    monkeypatch.setenv('ROBORSI_DEPLOY_CONFIG',str(cfg))
    assert pi05.resolve_pi05_checkpoint(P)==(tmp_path/'weights').resolve()


def test_shared_pi05_requires_verified_receipt(tmp_path):
    identity = import_module('robots.robodojo.agents.submission.vla_identity')
    with pytest.raises(ValueError, match='receipt'):
        identity.validate_shared_metadata({}, backend='pi05', port=19802, receipt_path=None, policy_dir=P)


@pytest.mark.parametrize('damage',['backend','weights','port','service_id'])
def test_shared_pi05_rejects_wrong_service(tmp_path, damage):
    identity = import_module('robots.robodojo.agents.submission.vla_identity')
    expected=identity.pi05_identity(P)
    wire={**expected,'service_id':'example-service'}
    receipt={'schema_version':1,'port':19802,'identity':wire,'checkpoint_files_verified':True}
    receipt_path=tmp_path/'receipt.json';receipt_path.write_text(json.dumps(receipt))
    metadata={'roborsi_vla_identity':dict(wire)}
    port=19802
    if damage=='backend':metadata['roborsi_vla_identity']['backend']='simate_beta'
    if damage=='weights':metadata['roborsi_vla_identity']['checkpoint_metadata_sha256']='0'*64
    if damage=='port':port=19803
    if damage=='service_id':metadata['roborsi_vla_identity']['service_id']='different-service'
    with pytest.raises(ValueError, match='identity|receipt'):
        identity.validate_shared_metadata(metadata,backend='pi05',port=port,receipt_path=receipt_path,policy_dir=P)


def test_shared_pi05_accepts_verified_service(tmp_path):
    identity = import_module('robots.robodojo.agents.submission.vla_identity')
    wire={**identity.pi05_identity(P),'service_id':'example-service'}
    receipt_path=tmp_path/'receipt.json'
    receipt_path.write_text(json.dumps({'schema_version':1,'port':19802,'identity':wire,'checkpoint_files_verified':True}))
    identity.validate_shared_metadata({'roborsi_vla_identity':wire},backend='pi05',port=19802,receipt_path=receipt_path,policy_dir=P)


@pytest.mark.parametrize('field,value',[('vla_backend','simate_beta'),('vla_backend','disabled'),('vla_base_prompt_sha256','different-prompt')])
def test_recovery_cannot_change_backend_or_wording(field,value):
    resume=import_module('robots.robodojo.agents.submission.hybrid_resume')
    old={'options':{'vla_backend':'pi05','vla_base_prompt_sha256':'verified-base'},'episode_seconds':None}
    new={**old,'options':{**old['options'],field:value}}
    with pytest.raises(ValueError,match='original action/review/context options'):
        resume.validate_policy(old,new)


def test_service_identity_rejects_another_openpi_configuration():
    identity=import_module('robots.robodojo.agents.submission.vla_identity')
    with pytest.raises(ValueError,match='config'):
        identity.pi05_identity(P,config_name='pi05_libero')


def test_pi05_installer_and_model_share_checkpoint_default(tmp_path,monkeypatch):
    cfg=tmp_path/'experiment.yaml'
    cfg.write_text(yaml.safe_dump({'vla_backend':'pi05','bench_name':'RoboDojo','ckpt_name':'roborsi-codex-v1','env_cfg_type':'arx_x5','action_type':'joint','seed':0}))
    monkeypatch.setenv('ROBORSI_DEPLOY_CONFIG',str(cfg))
    assert pi05.resolve_pi05_checkpoint(P)==(P/'checkpoints/pi05-arx-x5-seed0-step59999').resolve()
