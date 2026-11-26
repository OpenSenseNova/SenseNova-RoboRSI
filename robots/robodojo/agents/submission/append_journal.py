"""Single-writer, append-only recovery records without metadata renames."""
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import threading

_shared = {}
_shared_lock = threading.RLock()


def register(writer):
    with _shared_lock:
        path = writer.path.resolve()
        previous = _shared.get(path)
        if previous is not None and previous.fd is not None and previous is not writer:
            raise RuntimeError('Another live case journal is already registered')
        _shared[path] = writer


def shared(path):
    with _shared_lock:
        writer = _shared.get(Path(path).resolve())
        return writer if writer is not None and writer.fd is not None else None


def shared_writer(path):
    """Serialize first publication too; threads share one kernel writer lock."""
    with _shared_lock:
        writer = shared(path)
        if writer is None:
            writer = Journal(path)
            register(writer)
        return writer


def read_records(path):
    """Read-only readers consume complete checksummed lines, never a torn tail."""
    result, previous = [], '0'*64
    with Path(path).open('rb') as stream:
        for line in stream:
            if not line.endswith(b'\n'):
                break
            value = json.loads(line)
            digest = value.pop('sha256')
            if (value['sequence'] != len(result)+1 or value['previous'] != previous
                    or hashlib.sha256(encoded(value)).hexdigest() != digest):
                raise ValueError('Journal integrity failure; preserve source')
            result.append(dict(value, sha256=digest))
            previous = digest
    return result


def encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')


def difference(before, after, path=()):
    """Exact JSON changes; unchanged history is never serialized into a record."""
    if type(before) is not type(after):
        return [['set', list(path), after]]
    if isinstance(after, dict):
        result = [['del', [*path, key]] for key in before if key not in after]
        for key, value in after.items():
            result += (difference(before[key], value, (*path, key)) if key in before
                       else [['set', [*path, key], value]])
        return result
    if isinstance(after, list):
        result = []
        for index in range(min(len(before), len(after))):
            result += difference(before[index], after[index], (*path, index))
        if len(after) > len(before):
            result.append(['extend', list(path), after[len(before):]])
        elif len(after) < len(before):
            result.append(['truncate', list(path), len(after)])
        return result
    return [] if before == after else [['set', list(path), after]]


def apply_changes(value, changes):
    value = deepcopy(value)
    for change in changes:
        op, path, *argument = change
        if not path and op == 'set':
            value = deepcopy(argument[0])
            continue
        target = value
        for key in (path if op in ('extend', 'truncate') else path[:-1]):
            target = target[key]
        if op == 'set':
            target[path[-1]] = deepcopy(argument[0])
        elif op == 'del':
            del target[path[-1]]
        elif op == 'extend':
            target.extend(deepcopy(argument[0]))
        elif op == 'truncate':
            del target[argument[0]:]
        else:
            raise ValueError('Unknown journal change: '+op)
    return value


class Journal:
    """The lock and durable ACK belong to this writer, never to a heartbeat."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
        self.lock = threading.RLock()
        self.failed = False
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.records = self._recover()
        except BaseException:
            os.close(self.fd)
            self.fd = None
            raise
        self.sequence = len(self.records)
        self.digest = self.records[-1]['sha256'] if self.records else '0'*64

    def _recover(self):
        records, previous, end = [], '0'*64, 0
        with os.fdopen(os.dup(self.fd), 'rb') as stream:
            stream.seek(0)
            while True:
                line = stream.readline()
                if not line:
                    return records
                if not line.endswith(b'\n'):
                    # Only the unacknowledged trailing fragment may be removed.
                    os.ftruncate(self.fd, end)
                    os.fsync(self.fd)
                    return records
                value = json.loads(line)
                digest = value.pop('sha256')
                if (value['sequence'] != len(records)+1 or value['previous'] != previous
                        or hashlib.sha256(encoded(value)).hexdigest() != digest):
                    raise ValueError('Journal integrity failure; preserve source')
                previous = digest
                records.append(dict(value, sha256=digest))
                end = stream.tell()

    def append(self, kind, data):
        return self.append_many([(kind, data)])[-1]

    def append_many(self, records):
        with self.lock:
            if self.fd is None or self.failed:
                raise RuntimeError('Journal is closed or has an unacknowledged write failure')
            pending, previous = [], self.digest
            for kind, data in records:
                value = dict(sequence=self.sequence+len(pending)+1, previous=previous,
                             kind=kind, data=deepcopy(data))
                previous = hashlib.sha256(encoded(value)).hexdigest()
                pending.append(dict(value, sha256=previous))
            if not pending:
                return []
            data = b''.join(encoded(value)+b'\n' for value in pending)
            try:
                offset = 0
                while offset < len(data):
                    written = os.write(self.fd, data[offset:])
                    if written <= 0:
                        raise OSError('Journal write made no progress')
                    offset += written
                os.fsync(self.fd)
            except BaseException:
                self.failed = True
                raise
            self.records.extend(pending)
            self.sequence += len(pending)
            self.digest = previous
            return pending

    def close(self):
        with self.lock:
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
