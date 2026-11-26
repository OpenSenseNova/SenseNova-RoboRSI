"""Camera capture to packaged skill input to spatial memory, without a model."""
from importlib import import_module
import json
from pathlib import Path
from types import SimpleNamespace, ModuleType
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

native=import_module('robots.robodojo.agents.submission.native_camera')
spatial=import_module('robots.robodojo.agents.submission.spatial_observation')
backend=import_module('robots.robodojo.agents.submission.hybrid_backend')
P=Path(__import__('robots.robodojo.agents.submission', fromlist=['__file__']).__file__).parent
NAMES=('cam_head','cam_left_wrist','cam_right_wrist')


@pytest.fixture
def capture(monkeypatch):
    projection=np.zeros((4,4));projection[0,0]=projection[1,1]=2;projection[2,3]=-1
    class Reader:
        def attach(self,product):self.attached=True
        def detach(self):self.attached=False
        def get_data(self):
            return dict(cameraModel='pinhole',renderProductResolution=[100,100],cameraProjection=projection)
    class Camera:
        def initialize(self):self.active=True
        def resume(self):self.active=True
        def pause(self):self.active=False
        def get_render_product_path(self):return '/synthetic/render'
        def get_rgb(self):return np.full((100,100,4),150,dtype=np.uint8)
        def get_world_pose(self,*,camera_axes):
            assert camera_axes=='ros'
            return np.array([5,7,1]),np.array([0,1,0,0])
    core=ModuleType('omni.replicator.core');core.annotators=SimpleNamespace(get=lambda _:Reader())
    replicator=ModuleType('omni.replicator');replicator.core=core
    omni=ModuleType('omni');omni.replicator=replicator
    for name,value in [('omni',omni),('omni.replicator',replicator),('omni.replicator.core',core)]:
        monkeypatch.setitem(sys.modules,name,value)
    env=SimpleNamespace(camera_manager=SimpleNamespace(camera_names=[list(NAMES)],cameras=[[Camera() for _ in NAMES]]),
        take_action_cnt=[12],scene_manager=SimpleNamespace(env_origins=np.array([[5,7,0]])),render=lambda:None)
    obs=dict(vision={n:dict(color=np.zeros((100,100,3),dtype=np.uint8)) for n in NAMES},additional_info={})
    return env,obs


def test_capture_optical_origin_and_projection_conventions(capture):
    env,obs=capture
    native.calibrated_observation(env,obs)
    assert env.take_action_cnt==[12]
    value=spatial.validate(obs['additional_info']['spatial_calibration'],obs['vision'],step=12)
    for c in value['cameras'].values():
        assert np.asarray(c['K'])==pytest.approx(np.array([[100,0,49.5],[0,100,49.5],[0,0,1]]))
        assert np.asarray(c['T_base_camera_cv'])[:3,3]==pytest.approx([0,0,1])
    assert all(not c.active for c in env.camera_manager.cameras[0])
    # Repeated capture resumes existing products and reads the current native step.
    env.take_action_cnt[0]=13
    native.calibrated_observation(env,obs)
    assert obs['additional_info']['spatial_calibration']['control_step']==13


def test_packaged_observation_runs_geometry_and_updates_memory(capture,tmp_path):
    env,obs=capture;native.calibrated_observation(env,obs)
    calibration=spatial.validate(obs['additional_info']['spatial_calibration'],obs['vision'],step=12)
    directory=tmp_path/'evaluation-0001';directory.mkdir()
    records = import_module('robots.robodojo.agents.submission.runtime_records')
    # Real episodes retain the initial JSON snapshot while later steps live in
    # the journal. A fresh step-12 observation must still localize successfully.
    records.write(directory/'latest_spatial.json', {'case_file':'previous.json','control_step':0})
    for name in NAMES:Image.fromarray(obs['vision'][name]['color']).save(directory/f'00000_{name}.jpg')
    ready=spatial.stage(calibration,directory,0,step=12)
    assert ready['control_step']==12
    assert json.loads((directory/'latest_spatial.json').read_text())['control_step']==0
    assert ready['image_sizes']=={name:{'width':100,'height':100} for name in NAMES}
    memory=tmp_path/'space.json'
    memory.write_text(json.dumps(dict(schema_version=1,task='fixture',frame='environment_origin',units='meters',objects={})))
    command=[sys.executable,'-B',str(P/'skillpack/common/skills/robodojo-spatial-localize/scripts/run_localize.py'),
        '--case',str(directory/'00000_spatial_case.json'),'--object','block','--description','visible plane contact',
        '--observation','cam_head,49.5,49.5','--plane-z','0','--height','0','--memory',str(memory),
        '--output',str(tmp_path/'localized')]
    result=subprocess.run(command,capture_output=True,text=True,check=True,cwd=tmp_path)
    assert json.loads(result.stdout)['status']=='localized'
    point=json.loads(memory.read_text())['objects']['block']['contact']
    assert point['position_m']==pytest.approx([0,0,0]) and point['control_step']==12
    # Frozen evidence is checked before another localization is attempted.
    (directory/'00000_cam_head.jpg').write_bytes(b'changed image')
    failure=subprocess.run(command,capture_output=True,text=True,cwd=tmp_path)
    assert failure.returncode!=0 and 'image checksum differs' in failure.stderr
    records.write(directory/'latest_spatial.json', {'case_file':'00001_spatial_case.json','control_step':13})
    failure=subprocess.run(command,capture_output=True,text=True,cwd=tmp_path)
    assert failure.returncode!=0 and 'observation is stale' in failure.stderr


def test_localization_rejects_state_change_during_geometry(capture, tmp_path):
    env,obs=capture;native.calibrated_observation(env,obs)
    calibration=spatial.validate(obs['additional_info']['spatial_calibration'],obs['vision'],step=12)
    directory=tmp_path/'evaluation-0001';directory.mkdir()
    for name in NAMES:Image.fromarray(obs['vision'][name]['color']).save(directory/f'00000_{name}.jpg')
    spatial.stage(calibration,directory,0,step=12)
    records=import_module('robots.robodojo.agents.submission.runtime_records')
    # Transfer the fixture's sole journal writer to the subprocess that injects
    # the later host update; two concurrent writers are intentionally forbidden.
    records.close(directory/'latest_spatial.json')
    memory=tmp_path/'space.json'
    memory.write_text(json.dumps(dict(schema_version=1,task='fixture',frame='environment_origin',units='meters',objects={})))
    script=P/'skillpack/common/skills/robodojo-spatial-localize/scripts/run_localize.py'
    # Inject a host state change only after the actual geometry subprocess has
    # completed, so this exercises the final guard rather than the entry check.
    program='''import importlib,json,runpy,subprocess,sys
from pathlib import Path
records=importlib.import_module('robots.robodojo.agents.submission.runtime_records')
original=subprocess.run
def moved(*args,**kwargs):
    result=original(*args,**kwargs)
    records.write(Path(sys.argv[2]).parent/'latest_spatial.json',
                  dict(case_file='next.json',control_step=13))
    return result
subprocess.run=moved
script=sys.argv[1]
sys.path.insert(0,str(Path(script).parent))
sys.argv=[script,'--case',sys.argv[2],*sys.argv[3:]]
runpy.run_path(script,run_name='__main__')
'''
    result=subprocess.run([sys.executable,'-B','-c',program,str(script),str(directory/'00000_spatial_case.json'),
        '--object','block','--description','visible plane contact','--observation','cam_head,49.5,49.5',
        '--plane-z','0','--height','0','--memory',str(memory),'--output',str(tmp_path/'localized')],
        capture_output=True,text=True,cwd=tmp_path)
    assert result.returncode==2, result.stderr
    assert json.loads(result.stdout)['status']=='stale_observation', result.stdout
    assert json.loads(memory.read_text())['objects']=={}


def test_stale_or_wrong_size_calibration_is_rejected(capture):
    env,obs=capture;native.calibrated_observation(env,obs)
    c=obs['additional_info']['spatial_calibration']
    with pytest.raises(ValueError,match='control step'):spatial.validate(c,obs['vision'],step=13)
    c['cameras']['cam_head']['image_size']=[480,640]
    with pytest.raises(ValueError,match='dimensions'):spatial.validate(c,obs['vision'])


def test_public_observation_preserves_full_precision_calibration(capture):
    env,obs=capture;native.calibrated_observation(env,obs)
    obs['instruction']='Move the block.'
    obs['state']={name:np.zeros(size) for name,size in (*backend._JOINT_CHANNELS,('left_ee_pose',7),('right_ee_pose',7))}
    result=backend._observation(obs)
    assert result['additional_info']['spatial_calibration']==obs['additional_info']['spatial_calibration']
