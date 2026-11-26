"""Terminal-only publication of a training-first case; no content archive copy."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil

from .training_store import finalize_case, valid_commit


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(path, value):
    """One complete document, never a live progress snapshot."""
    with Path(path).open('w') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
        stream.flush();os.fsync(stream.fileno())


def owned_case(path):
    path=Path(path)
    if not path.is_absolute() or path.is_symlink() or path.resolve(strict=True)!=path:
        raise ValueError('Minimal case requires a real absolute case directory')
    if path.stat().st_uid!=os.geteuid():
        raise ValueError('Minimal case belongs to another owner')
    return path


@contextmanager
def writer(case):
    # Keep this inode; unlinking a flock file would allow a second writer.
    with (case/'.writer.lock').open('a') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def prepare(case, plan):
    """Plan contains lightweight immutable scene/model/code references only."""
    case=owned_case(case)
    with writer(case):
        if (case/'DELIVERED').exists():
            raise ValueError('Case already delivered; do not repeat it')
        path=case/'case.json'
        if path.exists():
            if json.loads(path.read_bytes())!=plan:
                raise ValueError('A case cannot change its frozen plan')
        else:
            publish(path,plan)
        runtime=case/'.runtime'
        if runtime.is_symlink():
            raise ValueError('Case runtime cannot reference another directory')
        runtime.mkdir(exist_ok=True)
    return runtime


def valid_delivery(case):
    case=Path(case)
    try:
        raw=(case/'DELIVERED').read_bytes()
        if not raw.endswith(b'\n'):
            return False
        record=json.loads(raw)
        if (record['format']!='xpolicy_case_delivery_v2'
                or record['plan_sha256']!=sha(case/'case.json')
                or record['native_result_sha256']!=sha(case/'native-terminal.json')
                or not isinstance(record['training_roots'],dict) or not record['training_roots']):
            return False
        for relative, digest in record['training_roots'].items():
            name=Path(relative)
            root=case/name
            if (name.is_absolute() or '..' in name.parts or name.parts[0]!='training'
                    or root.is_symlink() or not root.resolve().is_relative_to(case.resolve()/'training')
                    or sha(root/'COMMITTED')!=digest or not valid_commit(root)):
                return False
        return True
    except (OSError,ValueError,KeyError,TypeError):
        return False


def finish(case):
    """Only a closed native lease and independently valid training may reduce runtime."""
    case=owned_case(case)
    with writer(case):
        runtime=case/'.runtime'
        marker=case/'DELIVERED'
        if marker.exists() and not marker.read_bytes().endswith(b'\n'):
            marker.unlink()  # Only this exited finalizer's incomplete publication.
        if marker.exists():
            if not valid_delivery(case):
                raise ValueError('Committed case changed; preserve runtime')
            if runtime.exists():
                if runtime.is_symlink():raise ValueError('Runtime escapes case')
                shutil.rmtree(runtime)
            return json.loads(marker.read_bytes())
        if runtime.is_symlink() or not runtime.is_dir():
            raise ValueError('Missing owned runtime; preserve the case')
        terminal=case/'native-terminal.json'
        result=json.loads(terminal.read_bytes())
        if result.get('done') is not True:
            raise ValueError('A real native terminal result is required')
        released=json.loads((runtime/'native/pool-close.json').read_bytes())
        acquired=json.loads((runtime/'native/pool-lease.json').read_bytes())
        if (released.get('lease_released') is not True
                or released.get('lease')!=acquired['lease']
                or released.get('simulator')!=acquired['simulator']):
            raise ValueError('The native episode lease is still held')
        plan=json.loads((case/'case.json').read_bytes())
        assembly=runtime/'training-publication.json'
        if assembly.exists():
            raw=assembly.read_bytes()
            if not raw.endswith(b'\n'):
                assembly.unlink()
        if assembly.exists():
            operations=json.loads(assembly.read_bytes())
        else:
            roots=sorted(runtime.glob('trajectories/*/seed-*/*/runtime-*/agent/training'))
            if not roots:raise ValueError('Case has no captured training source')
            finalize_case(case,terminal)
            operations=[]
            for index,root in enumerate(roots):
                manifest=json.loads((root/'manifest.json').read_bytes())
                if not valid_commit(root):raise ValueError('Training source is incomplete; retain runtime')
                if (manifest['metadata'].get('task_id')!=plan['task']
                        or manifest['metadata'].get('model')!=plan['model']):
                    raise ValueError('Training source differs from the frozen case')
                target='training' if len(roots)==1 else f'training/attempt-{index+1:04d}'
                operations.append(dict(source=root.relative_to(case).as_posix(),target=target,
                                       commit_sha256=sha(root/'COMMITTED')))
            publish(assembly,operations)
        delivered={}
        for operation in operations:
            source_name,target_name=Path(operation['source']),Path(operation['target'])
            if (source_name.is_absolute() or '..' in source_name.parts
                    or source_name.parts[:2]!=('.runtime','trajectories')
                    or target_name.is_absolute() or '..' in target_name.parts
                    or target_name.parts[0]!='training'):
                raise ValueError('Training publication path escaped this case')
            source,target=case/source_name,case/target_name
            if (not source.resolve().is_relative_to(runtime)
                    or not target.parent.resolve().is_relative_to(case)):
                raise ValueError('Training publication reference escaped this case')
            if target.exists():
                if source.exists():raise ValueError('Conflicting training publication')
            else:
                if source.is_symlink() or target.is_symlink():raise ValueError('Training source cannot be a symlink')
                target.parent.mkdir(parents=True,exist_ok=True)
                source.rename(target)  # Exactly once at terminal publication, not per image/step.
            if not valid_commit(target) or sha(target/'COMMITTED')!=operation['commit_sha256']:
                raise ValueError('Published training source changed; retain runtime')
            delivered[target_name.as_posix()]=operation['commit_sha256']
        record=dict(format='xpolicy_case_delivery_v2',plan_sha256=sha(case/'case.json'),
                    native_result_sha256=sha(terminal),training_roots=delivered,
                    native_lease=dict(lease=released['lease'],simulator=released['simulator'],
                                      lease_released=True))
        publish(marker,record)
        if not valid_delivery(case):raise ValueError('Final case readback failed; retain runtime')
        shutil.rmtree(runtime)
        return record
