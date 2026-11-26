"""Bounded case-local scratch with a single durable training source.

The launcher opts in explicitly. Existing episodes keep their original layout.
Training publication precedes cleanup; native scores are never inferred here.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import tempfile
import time

BINDING_ENV='ROBORSI_EPHEMERAL_CASE_BINDING'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def publish(path,value):
    with Path(path).open('x')as stream:
        stream.write(json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n')
        stream.flush();os.fsync(stream.fileno())

def create(case,temporary_parent):
    case=Path(case);parent=Path(temporary_parent)
    if (not case.is_absolute()or case.is_symlink()or case.resolve()!=case
            or case.stat().st_uid!=os.geteuid()or not(case/'case.json').is_file()):
        raise ValueError('Scratch requires an existing owned immutable case')
    if not parent.is_absolute()or parent.is_symlink()or parent.resolve()!=parent or parent==Path('/'):
        raise ValueError('Scratch requires an explicit real private temporary parent')
    parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    if parent.stat().st_uid!=os.geteuid():raise ValueError('Temporary parent belongs to another owner')
    if(case/'runtime-binding.json').exists():raise ValueError('Existing episode binding cannot be repeated')
    temporary=Path(tempfile.mkdtemp(prefix='case-',dir=parent))
    value=dict(format='xpolicy_ephemeral_case_v1',case_path=str(case),temporary_root=str(temporary),
        temporary_parent=str(parent),host=socket.gethostname(),uid=os.geteuid(),launcher_pid=os.getpid(),
        plan_sha256=sha(case/'case.json'),training_root=str(case/'training'),at=time.time())
    source=case/'runtime-source.json'
    if source.exists():
        if source.is_symlink():raise ValueError('Runtime source binding cannot be redirected')
        value['runtime_source_sha256']=sha(source)
    publish(temporary/'.owned-by-case.json',dict(case_path=str(case),plan_sha256=value['plan_sha256'],
                    temporary_root=str(temporary),host=value['host'],uid=value['uid']))
    value['temporary_owner_sha256']=sha(temporary/'.owned-by-case.json')
    publish(case/'runtime-binding.json',value)
    return temporary

def binding(case):
    case=Path(case).resolve(strict=True);path=case/'runtime-binding.json'
    if path.is_symlink():raise ValueError('Runtime binding cannot be a symlink')
    value=json.loads(path.read_bytes());temporary=Path(value['temporary_root']);parent=Path(value['temporary_parent'])
    if (value['format']!='xpolicy_ephemeral_case_v1'or value['case_path']!=str(case)
            or value['plan_sha256']!=sha(case/'case.json')or value['uid']!=os.geteuid()
            or value['host']!=socket.gethostname()or temporary.parent!=parent
            or value['training_root']!=str(case/'training')or temporary.is_symlink()
            or parent.is_symlink()or parent.resolve()!=parent):
        raise ValueError('Runtime binding differs from this exact owned case/node')
    if temporary.exists()and(temporary.resolve()!=temporary or temporary.stat().st_uid!=os.geteuid()):
        raise ValueError('Temporary root ownership or identity changed')
    if 'runtime_source_sha256'in value:
        source=case/'runtime-source.json'
        if source.is_symlink()or sha(source)!=value['runtime_source_sha256']:
            raise ValueError('Runtime source identity changed')
    if temporary.exists():
        owner=temporary/'.owned-by-case.json'
        if owner.is_symlink()or not owner.is_file()or sha(owner)!=value['temporary_owner_sha256']:
            raise ValueError('Temporary case ownership receipt changed')
        if json.loads(owner.read_bytes())!={k:value[k]for k in ('case_path','plan_sha256','temporary_root','host','uid')}:
            raise ValueError('Temporary root belongs to another case')
    return value

def bound_runtime(path,case):
    try:
        value=binding(case);path=Path(path).resolve(strict=True)
        return path.is_relative_to(Path(value['temporary_root'])/'trajectories')
    except(OSError,ValueError,KeyError,TypeError):return False

def prepare_training(requested):
    """Keep the original logical path, bind content directly to durable training."""
    option=os.environ.get(BINDING_ENV)
    if not option:return None
    record=Path(option)
    if record.name!='runtime-binding.json':raise ValueError('Unexpected explicit case binding')
    value=binding(record.parent);requested=Path(requested)
    if (not requested.is_absolute()or requested.name!='training'or requested.parent.name!='agent'
            or not requested.parent.resolve(strict=True).is_relative_to(Path(value['temporary_root'])/'trajectories')
            or requested.exists()or requested.is_symlink()):
        raise ValueError('Training destination differs from the bound runtime')
    durable=Path(value['training_root'])
    if durable.is_symlink():raise ValueError('Durable training cannot be redirected')
    durable.mkdir(exist_ok=False)
    requested.symlink_to(durable,target_is_directory=True)
    return dict(binding_path=str(record),binding_sha256=sha(record),logical_training_path=str(requested),
                durable_training_path=str(durable),journal_name='events.jsonl',
                **({'runtime_source':json.loads((record.parent/'runtime-source.json').read_bytes())}
                   if 'runtime_source_sha256'in value else {}))

def finish(case):
    """Same native/training acceptance as the normal publisher, with no move/copy."""
    from . import case_storage as cases
    from .training_store import finalize,valid_commit
    case=cases.owned_case(case);value=binding(case);temporary=Path(value['temporary_root'])
    with cases.writer(case):
        if(case/'DELIVERED').exists():
            if not cases.valid_delivery(case):raise ValueError('Existing delivery changed')
            return json.loads((case/'DELIVERED').read_bytes())
        terminal=case/'native-terminal.json';native=json.loads(terminal.read_bytes())
        if native.get('done')is not True:raise ValueError('Real native terminal is required')
        acquired=json.loads((temporary/'native/pool-lease.json').read_bytes())
        released=json.loads((temporary/'native/pool-close.json').read_bytes())
        if(not released.get('lease_released')or released.get('lease')!=acquired['lease']
                or released.get('simulator')!=acquired['simulator']):raise ValueError('Native lease is still held or changed')
        stores=list(temporary.glob('trajectories/*/seed-*/*/runtime-*/agent/training'))
        if len(stores)!=1 or not stores[0].is_symlink()or stores[0].resolve()!=case/'training':
            raise ValueError('Exactly one bound durable training source is required')
        plan=json.loads((case/'case.json').read_bytes());manifest=json.loads((stores[0]/'manifest.json').read_bytes())
        if manifest['metadata'].get('task_id')!=plan['task']or manifest['metadata'].get('model')!=plan['model']:
            raise ValueError('Training source differs from this frozen task/model')
        finalize(stores[0],terminal)
        if not valid_commit(case/'training'):raise ValueError('Durable training independent readback failed')
        record=dict(format='xpolicy_case_delivery_v2',plan_sha256=sha(case/'case.json'),native_result_sha256=sha(terminal),
                    training_roots={'training':sha(case/'training/COMMITTED')},
                    native_lease=dict(lease=released['lease'],simulator=released['simulator'],lease_released=True),
                    ephemeral_binding_sha256=sha(case/'runtime-binding.json'))
        cases.publish(case/'DELIVERED',record)
        if not cases.valid_delivery(case):raise ValueError('Independent final case readback failed')
        return record

def cleanup(case):
    """Delete only this exited episode's scratch after durable capture validation."""
    from . import case_storage as cases
    from .training_store import _training_images_complete
    from .append_journal import read_records
    case=cases.owned_case(case);value=binding(case);temporary=Path(value['temporary_root'])
    with cases.writer(case):
        if temporary.exists():
            lease=temporary/'native/pool-lease.json'
            if lease.exists():
                close=temporary/'native/pool-close.json'
                if not close.is_file()or not json.loads(close.read_bytes()).get('lease_released'):
                    raise ValueError('Native lease remains active; preserve scratch')
            from .runtime_records import read
            for owner in temporary.glob('trajectories/*/seed-*/*/runtime-*/episode-owner.json'):
                if not read(owner).get('closed'):raise ValueError('Model owner remains active; preserve scratch')
        if(case/'DELIVERED').exists():
            if not cases.valid_delivery(case):raise ValueError('Changed delivery; preserve scratch')
            if json.loads((case/'DELIVERED').read_bytes()).get('ephemeral_binding_sha256')!=sha(case/'runtime-binding.json'):
                raise ValueError('Committed runtime binding changed; preserve scratch')
        elif(case/'training').exists():
            root=case/'training';manifest=json.loads((root/'manifest.json').read_bytes())
            if manifest.get('state')!='closed'or not _training_images_complete(root,manifest):
                raise ValueError('Capture is not durably closed; preserve scratch')
            events=read_records(root/'events.jsonl')
            # A partial/model-error capture retains score=null and its exact API ledger.
            partial=dict(at=time.time(),score=None,native_terminal=False,
                manifest_sha256=sha(root/'manifest.json'),event_records=len(events),
                event_tail_sha256=events[-1]['sha256']if events else None,
                files={p.name:sha(p)for p in root.iterdir()if p.is_file()and not p.is_symlink()})
            prior=case/'PARTIAL_CAPTURE'
            if prior.exists():
                saved=json.loads(prior.read_bytes())
                if saved['manifest_sha256']!=partial['manifest_sha256']or saved['files']!=partial['files']:
                    raise ValueError('Partial training changed; preserve scratch')
            else:publish(prior,partial)
        elif not(case/'attention.json').is_file():
            raise ValueError('No terminal or pre-capture error; preserve scratch')
        if temporary.exists():shutil.rmtree(temporary)
        if temporary.exists():raise ValueError('Own temporary root was not removed')
        if(case/'DELIVERED').exists():
            # Final actor/compact/native accounting are complete and independently checked.
            (case/'training/events.jsonl').unlink(missing_ok=True)
            management=case/'.runtime'
            if management.is_symlink():raise ValueError('Management directory changed; preserve it')
            if management.exists():shutil.rmtree(management)
        marker=case/'TEMP_CLEANED'
        if not marker.exists():publish(marker,dict(at=time.time(),temporary_root=str(temporary),
                    durable_training=str(case/'training'),binding_sha256=sha(case/'runtime-binding.json')))
