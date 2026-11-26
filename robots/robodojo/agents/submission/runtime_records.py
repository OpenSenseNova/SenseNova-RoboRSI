"""Small host-owned state deltas; live readers never need a renamed snapshot."""
from copy import deepcopy
import json
import os
from pathlib import Path

try:
    from .append_journal import shared, shared_writer, read_records, difference, apply_changes, encoded
except ImportError:  # Standalone, original recovery CLI.
    from append_journal import shared, shared_writer, read_records, difference, apply_changes, encoded


def journal_path(path):
    return Path(path).parent/'.runtime/state.journal.jsonl'


def _states(writer):
    if not hasattr(writer, 'runtime_states'):
        writer.runtime_states = {}
        for record in writer.records:
            if record['kind']=='runtime_state':
                value=record['data']
                writer.runtime_states[value['name']]=apply_changes(writer.runtime_states.get(value['name']),value['changes'])
    return writer.runtime_states


def read(path):
    path=Path(path)
    if path.is_symlink():
        raise ValueError('Host state may not be a symlink')
    jp=journal_path(path)
    writer=shared(jp)
    if writer is not None:
        value=_states(writer).get(path.name)
        if value is not None:
            return deepcopy(value)
    elif jp.exists():
        value=None
        for record in read_records(jp):
            if record['kind']=='runtime_state' and record['data']['name']==path.name:
                value=apply_changes(value,record['data']['changes'])
        if value is not None:
            return value
    return json.loads(path.read_bytes())


def write(path,value):
    path=Path(path)
    if path.is_symlink():
        raise ValueError('Host state may not be a symlink')
    jp=journal_path(path)
    writer=shared_writer(jp)
    with writer.lock:
        states=_states(writer)
        changes=difference(states.get(path.name),value)
        if changes:
            writer.append('runtime_state',dict(name=path.name,changes=changes))
            states[path.name]=deepcopy(value)
        # Stable identity is written once. A closed owner's final snapshot remains
        # independently readable after its temporary journal is safely removed.
        if not path.exists() or value.get('closed') is True:
            with path.open('wb') as stream:
                stream.write(encoded(value)+b'\n')
                stream.flush()
                os.fsync(stream.fileno())


def close(path):
    writer=shared(journal_path(path))
    if writer is not None:
        writer.close()
