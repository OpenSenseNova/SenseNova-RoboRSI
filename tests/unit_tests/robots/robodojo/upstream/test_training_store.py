"""Pre-decode source fidelity, compact boundaries and terminal-only retention.

Synthetic provider/native fixtures only; no model, simulator, decode or QC job.
"""
from copy import deepcopy
from importlib import import_module
import base64
import io
import json
import os
from pathlib import Path
import shutil
import threading

import numpy as np
from PIL import Image
import pytest

store = import_module('robots.robodojo.agents.submission.training_store')
codec = import_module('robots.robodojo.agents.submission.image_codec')


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def message(text, role='user', id=None):
    item = {'type':'message','role':role,'content':[{'type':'output_text' if role=='assistant' else 'input_text','text':text}]}
    if id:item['id']=id
    return item


def request(items, session='s1'):
    return {'model':'fixture','instructions':'system', 'tools':[], 'input':deepcopy(items),
        'client_metadata':{'x-codex-turn-metadata':json.dumps({'session_id':session})}}


def png():
    b=io.BytesIO();Image.new('RGB',(16,16),(234,20,8)).save(b,format='PNG')
    return b.getvalue()


def image(raw):
    return {'type':'input_image','image_url':'data:image/png;base64,'+base64.b64encode(raw).decode()}


def test_compact_is_its_own_sample_and_tail_survives_without_next_request(tmp_path):
    root=tmp_path/'agent/training'
    value=store.TrainingStore(root,metadata={'task_id':'align_blocks'})
    history=[message('episode')]
    output=[{'type':'reasoning','id':'rs1','encrypted_content':'seal-1','summary':[{'text':'not hidden think'}]},
        message('working','assistant','msg1'),
        {'type':'custom_tool_call','id':'tool1','call_id':'c1','namespace':'functions','name':'exec','input':'text(1)'}]
    raw=request(history);before=deepcopy(raw)
    value.append(raw,output,kind='actor',request_id=1,response_id='r1')
    assert raw==before
    tool={'type':'custom_tool_call_output','call_id':'c1','output':[image(png())]}
    history+=output+[tool]
    output2=[message('done with stage','assistant','msg2')]
    value.append(request(history),output2,kind='actor',request_id=2,response_id='r2')
    compact_input=history+output2+[message('compact now')]
    summary=[{'type':'reasoning','id':'rscompact','encrypted_content':'seal-compact'},
             message('actual generated summary','assistant','summary')]
    value.append(request(compact_input),summary,kind='compact',request_id=3,response_id='r3')
    after=[message('actual generated summary'),message('continue')]
    last=[{'type':'function_call','id':'fc4','call_id':'c4','namespace':'mcp__robot','name':'observe','arguments':'{}'}]
    value.append(request(after),last,kind='actor',request_id=4,response_id='r4')
    native=tmp_path/'agent/codex-home/sessions/2026/11/06/native.jsonl';native.parent.mkdir(parents=True)
    tail={'type':'function_call_output','call_id':'c4','output':[{'type':'input_text','text':'last native result'},image(png())]}
    native.write_text('\n'.join(json.dumps({'type':'response_item','payload':item}) for item in output+output2+last+[tail])+'\n')
    value.close(session_root=tmp_path/'agent')
    actor,compact=rows(root/'actor.jsonl'),rows(root/'compact.jsonl')
    assert len(actor)==2 and len(compact)==1
    assert [r['segment_id'] for r in actor]==['000001','000003']
    assert compact[0]['segment_id']=='000002'
    assert len(actor[0]['targets'])==2 and len(compact[0]['targets'])==1
    assert actor[-1]['messages'][-1]['role']=='tool'
    assert actor[-1]['messages'][-1]['content'][0]['text']=='last native result'
    assert actor[-1]['targets'][0]['native_consumed'] is True
    assert all(row['score'] is None for row in actor+compact)
    data=json.dumps(actor+compact)
    assert 'seal-1' in data and 'seal-compact' in data
    assert 'not hidden think' in data and 'data:image' not in data
    assert all(m['reasoning_content']=='' for r in actor+compact for m in r['messages'] if 'reasoning_content' in m)
    pictures=list((root/'images').glob('*.jpg'))
    assert len(pictures)==1  # Repeated observation is stored once.
    rgb=codec.decode_image_bit(pictures[0].read_bytes())
    assert np.abs(rgb.astype(int)-np.array([234,20,8])).max()<=3
    assert not (root/'segments').exists()


def test_context_rewrites_and_invocation_changes_are_separate_rows(tmp_path):
    value=store.TrainingStore(tmp_path/'training')
    initial=[message('start')];out=[message('answer','assistant','m1')]
    value.append(request(initial),out,kind='actor',request_id=1,response_id='r1')
    # A retry/restart with the identical initial prompt is not a continuation.
    value.append(request(initial),out,kind='actor',request_id=2,response_id='r2')
    value.append(request(initial+out,session='s2'),[message('new','assistant','m2')],kind='actor',request_id=3,response_id='r3')
    value.close()
    assert len(rows(tmp_path/'training/actor.jsonl'))==3


def test_sse_cipher_variants_preserve_both_seals_without_duplicate_targets(tmp_path):
    done={'type':'reasoning','id':'same','encrypted_content':'done-seal'}
    terminal={**done,'encrypted_content':'terminal-seal'}
    body=('data: '+json.dumps({'type':'response.output_item.done','item':done})+'\n\n'
          +'data: '+json.dumps({'type':'response.completed','response':{'output':[terminal]}})+'\n\n').encode()
    output,alternatives=store.response_items(body,'text/event-stream',{'output':[done]})
    assert output==[done] and alternatives==[terminal]
    value=store.TrainingStore(tmp_path/'training')
    value.append(request([message('start')]),output,kind='actor',request_id=1,response_id='r1',alternatives=alternatives)
    value.close()
    assert rows(tmp_path/'training/actor.jsonl')[0]['messages'][-1]['reasoning_items']==[done]
    assert [r['item'] for r in rows(tmp_path/'training/cipher_variants.jsonl')]==[done,terminal]


def test_actual_tools_and_freeform_inputs_are_retained(tmp_path):
    value=store.TrainingStore(tmp_path/'training')
    definition={'type':'namespace','name':'functions','tools':[{'type':'custom','name':'exec',
        'description':'execute','format':{'type':'grammar','definition':'complete grammar'}}]}
    req=request([{'type':'additional_tools','tools':[definition]},message('start')])
    code='text(`hello\n world`);'
    output=[{'type':'custom_tool_call','id':'fc','call_id':'c','namespace':'functions','name':'exec','input':code}]
    value.append(req,output,kind='actor',request_id=1,response_id='r')
    value.close()
    row=rows(tmp_path/'training/actor.jsonl')[0]
    assert row['native_tool_schema']==[definition]
    assert row['tools'][0]['function']['name']=='functions.exec'
    assert row['messages'][-1]['tool_calls'][0]['function']=={'name':'functions.exec','arguments':{'input':code}}
    assert 'complete grammar' in row['tools'][0]['function']['description']


def test_score_binding_and_reduction_require_completed_same_case(tmp_path):
    case=tmp_path/'case';runtime=case/'trajectories/align_blocks/seed-0/id/runtime-a'
    root=runtime/'agent/training'
    value=store.TrainingStore(root,metadata={'task_id':'align_blocks'})
    value.append(request([message('start')]),[message('answer','assistant','m')],kind='actor',request_id=1,response_id='r')
    value.close()
    store.write(runtime/'episode-owner.json',{'closed':True,'native_identity':{'task':'align_blocks','layout':0,'scene_sha256':'scene'}})
    store.write(case/'native-identity.json',{'layout_sha256':'scene'})
    store.write(runtime/'archive.pending.json',{'native_terminal_accepted':False})
    workspace=runtime/'agent/workspace';workspace.mkdir();(workspace/'needed.txt').write_text('recovery')
    terminal=case/'native-terminal.json';store.write(terminal,{'done':True,'metrics':{'score':.5,'success':False,'layout_id':0}})
    assert store.finalize_case(case,terminal)==1
    assert workspace.exists()  # Not an accepted native terminal yet.
    assert rows(root/'actor.jsonl')[0]['score']==.5
    store.write(runtime/'archive.pending.json',{'native_terminal_accepted':True})
    assert store.compact_closed_runtime(runtime) is True
    assert not workspace.exists() and (root/'actor.jsonl').is_file()
    store.write(case/'native-identity.json',{'layout_sha256':'wrong scene'})
    with pytest.raises(ValueError,match='scene'):
        store.finalize(root,terminal)
    with pytest.raises(ValueError,match='same|share a case'):
        wrong=tmp_path/'another/native-terminal.json';store.write(wrong,{'done':True,'metrics':{'score':1}})
        store.finalize(root,wrong)


def test_jpeg_is_stamped_without_recompression_and_legacy_is_unchanged():
    b=io.BytesIO();Image.new('RGB',(16,16),(240,20,10)).save(b,format='JPEG',quality=95,subsampling=0)
    original=b.getvalue();jpg=codec.encode_image_bit(original,standard=True)
    assert jpg!=original and jpg.replace(b'\xff\xfe\x00\x0aXPL-RGB1',b'',1)==original
    assert np.array_equal(codec.decode_image_bit(original,standard=True),codec.decode_image_bit(jpg))
    assert codec.encode_image_bit(jpg,standard=True)==jpg
    # Opt-in must not reinterpret existing unmarked legacy buffers globally.
    assert not np.array_equal(codec.decode_image_bit(original),codec.decode_image_bit(jpg))


def test_seal_version_change_does_not_create_a_false_context_boundary(tmp_path):
    value=store.TrainingStore(tmp_path/'training')
    history=[message('start')]
    first={'type':'reasoning','id':'rs','encrypted_content':'done-seal'}
    value.append(request(history),[first],kind='actor',request_id=1,response_id='r1')
    history.append({**first,'encrypted_content':'native-seal'})
    value.append(request(history),[message('end','assistant','m2')],kind='actor',request_id=2,response_id='r2')
    value.close()
    assert len(rows(tmp_path/'training/actor.jsonl'))==1
    assert {r['item']['encrypted_content'] for r in rows(tmp_path/'training/cipher_variants.jsonl')}=={'done-seal','native-seal'}


@pytest.mark.parametrize('mode',['training','raw'])
def test_local_provider_path_discards_full_payloads_only_in_training_mode(tmp_path,monkeypatch,mode):
    import urllib.request
    guard=import_module('robots.robodojo.agents.submission.skillpack.capture_guard')
    cci=import_module('robots.robodojo.agents.submission.cci_protocol')
    config=import_module('roborsi.planner.provider_capture').ProviderCaptureConfig
    from types import SimpleNamespace

    original=urllib.request.urlopen
    class Response:
        status=200
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self):return json.dumps({'id':'r1','status':'completed',
            'output':[{'type':'reasoning','id':'rs','encrypted_content':'opaque-seal','summary':[]},
                      message('answer','assistant','m1')],
            'usage':{'input_tokens':100,'output_tokens':5,'input_tokens_details':{'cached_tokens':90}}}).encode()
    def upstream(req,*args,**kwargs):
        return Response() if req.full_url.startswith('http://fixture.invalid/') else original(req,*args,**kwargs)
    monkeypatch.setattr(urllib.request,'urlopen',upstream)
    adapter=cci.CciAdapter('http://fixture.invalid/v1','fixture',tmp_path/'cci',save_raw=mode=='raw')
    adapter.start()
    capture=guard.GuardedCapture(config(adapter.base_url,'runtime'),tmp_path/'api',case_dir=tmp_path,
        model='fixture',model_label='fixture',cohort_root=tmp_path,
        prompt_profile=SimpleNamespace(transform=lambda x:(x,{})),storage_mode=mode)
    capture.start()
    try:
        req=request([message('start')]);req['reasoning']={'effort':'xhigh'};req['stream']=False
        call=urllib.request.Request(capture.base_url+'/responses',data=json.dumps(req).encode(),headers={'Content-Type':'application/json'})
        reply=json.load(urllib.request.urlopen(call,timeout=10))
        assert reply['status']=='completed' and reply['output'][-1]['content'][0]['text']=='answer'
    finally:
        capture.stop();adapter.stop()
    assert bool(list((tmp_path/'api').glob('*.request.json')))==(mode=='raw')
    assert bool(list((tmp_path/'api').glob('*.response.bin')))==(mode=='raw')
    assert bool(list((tmp_path/'cci').glob('*.request.json')))==(mode=='raw')
    assert (tmp_path/'usage.jsonl').exists() and (tmp_path/'cci/000001.receipt.json').exists()
    if mode=='training':
        assert rows(tmp_path/'training/actor.jsonl')[0]['messages'][-1]['reasoning_items'][0]['encrypted_content']=='opaque-seal'
        assert rows(tmp_path/'api/requests.jsonl')[0]['raw_payload_retained'] is False


def test_compact_publication_keeps_source_and_skips_duplicate_workspaces(tmp_path,monkeypatch):
    archive=import_module('robots.robodojo.agents.submission.trajectory_archive')
    source=tmp_path/'source'
    value=store.TrainingStore(source/'agent/training')
    value.append(request([message('start')]),[message('end','assistant','m')],kind='actor',request_id=1,response_id='r')
    value.close()
    workspace=source/'agent/workspace';workspace.mkdir();(workspace/'large.png').write_bytes(b'fixture pixels')
    monkeypatch.setenv('ROBORSI_TRACE_ROOT',str(tmp_path/'archive'))
    target=archive.archive_episode(source,task='align_blocks',episode_id='fixture')
    from pathlib import Path
    target=Path(target)
    assert (target/'runtime/agent/training/actor.jsonl').is_file()
    assert not (target/'runtime/agent/workspace').exists()
    assert (workspace/'large.png').is_file()  # Publication never deletes source.


def make_capture(tmp_path):
    guard=import_module('robots.robodojo.agents.submission.skillpack.capture_guard')
    config=import_module('roborsi.planner.provider_capture').ProviderCaptureConfig
    return guard.GuardedCapture(config('https://fixture.invalid/v1','runtime'),tmp_path/'api',
        case_dir=tmp_path,model='fixture',model_label='fixture',cohort_root=tmp_path,
        storage_mode='training')


def test_capture_timeout_keeps_segments_open_until_final_writer_has_stopped(tmp_path):
    capture=make_capture(tmp_path)
    ready=threading.Event();release=threading.Event()
    def writer():
        ready.set()
        assert release.wait(5)
        capture.training_store.append(request([message('start')]),
            [{'type':'reasoning','id':'late','encrypted_content':'final-seal'},
             message('last response','assistant','last')],kind='actor',request_id=1,response_id='r1')
    thread=threading.Thread(target=writer)
    capture._thread=thread
    thread.start()
    assert ready.wait(1)
    try:
        with pytest.raises(RuntimeError,match='cleanup deadline'):
            capture.stop(timeout_s=0)
        assert json.loads((tmp_path/'training/manifest.json').read_text())['state']=='open'
    finally:
        release.set();thread.join(5)
    assert not thread.is_alive()
    capture.stop(timeout_s=1)
    actor=rows(tmp_path/'training/actor.jsonl')
    assert actor[0]['messages'][-1]['content'][0]['text']=='last response'
    assert actor[0]['messages'][-1]['reasoning_items'][0]['encrypted_content']=='final-seal'
    assert json.loads((tmp_path/'training/manifest.json').read_text())['state']=='closed'


def test_stopped_capture_keeps_persistence_error_and_closes_only_collected_source(tmp_path):
    capture=make_capture(tmp_path)
    capture.training_store.append(request([message('start')]),
        [message('retained','assistant','m')],kind='actor',request_id=1,response_id='r1')
    capture._errors.append('OSError')
    with pytest.raises(RuntimeError,match='persist all response records'):
        capture.stop()
    assert json.loads((tmp_path/'training/manifest.json').read_text())['state']=='closed'
    assert rows(tmp_path/'training/actor.jsonl')[0]['messages'][-1]['content'][0]['text']=='retained'
    assert json.loads((tmp_path/'training/native-evidence.json').read_text())['gaps']==[
        {'kind':'capture_persistence_error','error_type':'OSError'}]


@pytest.mark.parametrize('publication',['manifest.json','native-evidence.json'])
def test_capture_retries_failed_training_publication_without_losing_segments(tmp_path,monkeypatch,publication):
    capture=make_capture(tmp_path)
    capture.training_store.append(request([message('start')]),
        [message('retained after ENOSPC','assistant','m')],kind='actor',request_id=1,response_id='r1')
    original=store.publish_document
    failures=[]
    def write_with_full_disk(path,value):
        if path==tmp_path/'training'/publication and not failures:
            failures.append(str(path))
            raise OSError(28,'Synthetic disk full')
        return original(path,value)
    monkeypatch.setattr(store,'publish_document',write_with_full_disk)
    with pytest.raises(OSError,match='Synthetic disk full'):
        capture.stop()
    assert capture._closed and not capture.training_store.closed
    assert (tmp_path/'.runtime/training.journal.jsonl').is_file()
    capture.stop()
    assert capture.training_store.closed
    assert json.loads((tmp_path/'training/manifest.json').read_text())['state']=='closed'
    assert (tmp_path/'training/native-evidence.json').is_file()
    assert rows(tmp_path/'training/actor.jsonl')[0]['messages'][-1]['content'][0]['text']=='retained after ENOSPC'
    assert not (tmp_path/'training/segments').exists()


def closed_image_case(tmp_path):
    case=tmp_path/'case';trajectory=case/'trajectories/align_blocks/seed-1/episode'
    runtime=trajectory/'runtime-a';root=runtime/'agent/training'
    value=store.TrainingStore(root,metadata={'task_id':'align_blocks'})
    value.append(request([message('start'),{'type':'message','role':'user','content':[image(png())]}]),
        [message('answer','assistant','m')],kind='actor',request_id=1,response_id='r1')
    value.close()
    store.write(runtime/'episode-owner.json',{'closed':True,
        'native_identity':{'task':'align_blocks','layout':4,'scene_sha256':'scene'}})
    store.write(runtime/'archive.pending.json',{'native_terminal_accepted':True,
        'source':str(runtime),'trajectory_directory':str(trajectory)})
    store.write(case/'native-identity.json',{'layout_sha256':'scene'})
    terminal=case/'native-terminal.json'
    store.write(terminal,{'done':True,'metrics':{'score':1,'success':True,'layout_id':4}})
    workspace=runtime/'agent/workspace';workspace.mkdir();(workspace/'recovery.txt').write_text('keep until verified')
    shared=trajectory/'controller/observations/evaluation-0001/frame.jpg'
    legacy=runtime/'agent/controller/observations/frame.jpg'
    for path in [shared,legacy]:
        path.parent.mkdir(parents=True);path.write_bytes(b'derived controller frame')
    journal=trajectory/'controller/acknowledgements.jsonl';journal.write_text('original ACK\n')
    return case,trajectory,runtime,root,terminal,workspace,shared,legacy,journal


def test_terminal_scoring_interruption_keeps_complete_rows_and_can_retry(tmp_path,monkeypatch):
    _,_,_,root,terminal,*_=closed_image_case(tmp_path)
    original=(root/'actor.jsonl').read_bytes()
    replace=os.replace
    def interrupted(source,target):
        if Path(target)==root/'actor.jsonl':
            raise OSError('injected terminal publication failure')
        return replace(source,target)
    monkeypatch.setattr(os,'replace',interrupted)
    with pytest.raises(OSError,match='terminal publication'):
        store.finalize(root,terminal)
    assert (root/'actor.jsonl').read_bytes()==original
    assert not (root/'COMMITTED').exists()
    monkeypatch.setattr(os,'replace',replace)
    store.finalize(root,terminal)
    assert store.valid_commit(root)
    assert rows(root/'actor.jsonl')[0]['score']==1


def test_commit_marker_requires_exact_files_and_is_idempotent(tmp_path):
    _,_,_,root,terminal,*_=closed_image_case(tmp_path)
    (root/'COMMITTED').write_bytes(b'{"incomplete":')
    store.finalize(root,terminal)
    before=(root/'COMMITTED').read_bytes()
    store.finalize(root,terminal)
    assert (root/'COMMITTED').read_bytes()==before
    marker=json.loads(before)
    marker['files'].pop('native-evidence.json')
    (root/'COMMITTED').write_text(json.dumps(marker)+'\n')
    assert not store.valid_commit(root)
    with pytest.raises(ValueError,match='Committed native result changed'):
        store.finalize(root,terminal)


def test_committed_score_cannot_change_and_cipher_variants_are_checked(tmp_path):
    _,_,_,root,terminal,*_=closed_image_case(tmp_path)
    (root/'cipher_variants.jsonl').write_text(json.dumps(dict(opaque='original'))+'\n')
    store.finalize(root,terminal)
    assert store.valid_commit(root)
    (root/'cipher_variants.jsonl').write_text(json.dumps(dict(opaque='changed'))+'\n')
    assert not store.valid_commit(root)
    store.write(terminal,{'done':True,'metrics':{'score':0,'layout_id':4}})
    with pytest.raises(ValueError,match='native result changed'):
        store.finalize(root,terminal)


def test_terminal_cleanup_reaches_shared_trajectory_observations_and_keeps_journals(tmp_path):
    case,trajectory,runtime,root,terminal,workspace,shared,legacy,journal=closed_image_case(tmp_path)
    outside=tmp_path/'other';outside.mkdir();(outside/'keep.jpg').write_bytes(b'not this episode')
    (shared.parent/'external').symlink_to(outside,target_is_directory=True)
    assert store.finalize_case(case,terminal)==1
    assert not shared.exists() and not legacy.exists() and not workspace.exists()
    assert journal.read_text()=='original ACK\n'
    assert (outside/'keep.jpg').read_bytes()==b'not this episode'
    assert len(list((root/'images').glob('*.jpg')))==1
    assert rows(root/'actor.jsonl')[0]['score']==1
    assert any('../controller/observations/' in name
               for name in json.loads((runtime/'storage-receipt.json').read_text())['removed'])


def test_all_continuation_stores_are_scored_before_shared_cleanup(tmp_path):
    case,trajectory,runtime,root,terminal,workspace,shared,legacy,journal=closed_image_case(tmp_path)
    sibling=trajectory/'runtime-b'
    shutil.copytree(runtime,sibling)
    assert store.finalize_case(case,terminal)==2
    assert all((path/'storage-receipt.json').is_file() for path in (runtime,sibling))
    assert not shared.exists()
    assert all(rows(path/'agent/training/actor.jsonl')[0]['score']==1 for path in (runtime,sibling))
    assert journal.read_text()=='original ACK\n'


@pytest.mark.parametrize('damage',['missing_image','corrupt_image','unresolved_reference','active_sibling','closed_sibling_incomplete','missing_evidence','capture_gap','foreign_trajectory'])
def test_cleanup_retains_recovery_material_until_training_source_is_verified(tmp_path,damage):
    case,trajectory,runtime,root,terminal,workspace,shared,legacy,journal=closed_image_case(tmp_path)
    picture=next((root/'images').glob('*.jpg'))
    if damage=='missing_image':picture.unlink()
    elif damage=='corrupt_image':picture.write_bytes(b'corrupt')
    elif damage=='unresolved_reference':
        actor=rows(root/'actor.jsonl')[0]
        actor['messages'][2]['content'][0]['paths']=['images/not-captured.jpg']
        (root/'actor.jsonl').write_text(json.dumps(actor)+'\n')
    elif damage=='active_sibling':
        store.write(trajectory/'runtime-b/episode-owner.json',{'closed':False})
    elif damage=='closed_sibling_incomplete':
        sibling=trajectory/'runtime-b'
        shutil.copytree(runtime,sibling)
        next((sibling/'agent/training/images').glob('*.jpg')).unlink()
    elif damage=='capture_gap':
        store.write(root/'native-evidence.json',{'gaps':[{'kind':'capture_persistence_error'}]})
    elif damage=='missing_evidence':
        (root/'native-evidence.json').unlink()
    elif damage=='foreign_trajectory':
        store.write(runtime/'archive.pending.json',{'native_terminal_accepted':True,
            'trajectory_directory':str(tmp_path/'another_episode')})
    assert store.finalize_case(case,terminal)==(2 if damage=='closed_sibling_incomplete' else 1)
    assert workspace.exists() and shared.exists() and legacy.exists()
    assert journal.read_text()=='original ACK\n'
    assert not (runtime/'storage-receipt.json').exists()
