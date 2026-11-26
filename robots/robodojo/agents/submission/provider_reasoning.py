"""Retain provider-returned reasoning before the CLI projection discards fields.

Opaque content, native plaintext and public summaries have separate semantics.
This module never decodes, reconstructs or generates reasoning.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def state(value, *, present=True):
    if not present:
        return 'absent'
    if value is None:
        return 'null'
    if value in ('', [], {}):
        return 'empty'
    return 'present_nonempty'


def plaintext(item):
    """Only explicit native plaintext; summary and ciphertext are never think."""
    parts = []
    top = item.get('native_response_reasoning_content')
    if isinstance(top, str):
        return top
    for key in ('reasoning_content', 'reasoning_text'):
        if key in item:
            value = item[key]
            if isinstance(value, str):
                return value
    if item.get('type') != 'reasoning':
        return ''
    if isinstance(item.get('text'), str):
        return item['text']
    content = item.get('content')
    if isinstance(content, str):
        return content
    for block in content if isinstance(content, list) else []:
        if isinstance(block, dict) and block.get('type') in ('reasoning_text', 'text') and isinstance(block.get('text'), str):
            parts.append(block['text'])
    return ''.join(parts)


def retain(directory, *, request_id, request, response, raw_body, projected_body):
    """One durable source record per real upstream attempt, including absence."""
    from .training_store import write, training_journal_path
    directory = Path(directory)
    fields = []
    def field(obj, key, path):
        present = key in obj
        value = obj.get(key)
        raw = encoded(value) if present else None
        fields.append(dict(path=path + [key], state=state(value, present=present),
                           json_type=type(value).__name__ if present else None,
                           value=deepcopy(value) if present else None,
                           encoded_bytes=len(raw) if raw is not None else None,
                           sha256=sha(raw) if raw is not None else None))
    field(response, 'reasoning_content', [])
    items = []
    for index, item in enumerate(response.get('output', [])):
        if not isinstance(item, dict):
            continue
        if item.get('type') == 'reasoning' or any(k in item for k in ('reasoning_content', 'reasoning_text')):
            items.append(dict(output_index=index, item=deepcopy(item)))
            for key in ('reasoning_content', 'reasoning_text', 'summary'):
                field(item, key, ['output', index])
            if item.get('type') == 'reasoning':
                for key in ('content', 'text', 'encrypted_content'):
                    field(item, key, ['output', index])
    value = dict(format='xpolicy_provider_reasoning_source_v1', request_id=request_id,
                 response_id=response.get('id'), model=response.get('model', request.get('model')),
                 request_sha256=sha(encoded(request)), upstream_body_sha256=sha(raw_body),
                 upstream_body_bytes=len(raw_body), projected_body_sha256=sha(projected_body),
                 native_items=items, fields=fields, reasoning_decoded=False,
                 source='actual_upstream_before_CLI_projection')
    from .append_journal import shared
    writer = shared(training_journal_path(directory.parent))
    if writer is not None:
        writer.append('provider_reasoning', value)
        return value
    path = directory / f'{request_id:06d}.json'
    write(path, value)
    manifest = directory / 'manifest.json'
    index = json.loads(manifest.read_bytes()) if manifest.exists() else dict(format='xpolicy_provider_reasoning_manifest_v1', files={})
    index['files'][path.name] = dict(sha256=sha(path.read_bytes()), request_id=request_id,
        response_id=value['response_id'], upstream_body_sha256=value['upstream_body_sha256'])
    write(manifest, index)
    return value


def load(directory, request_id, response_id):
    directory = Path(directory)
    from .append_journal import shared, read_records
    from .training_store import training_journal_path
    journal = training_journal_path(directory.parent)
    writer = shared(journal)
    candidates = None
    if writer is not None or journal.exists():
        records = writer.records if writer is not None else read_records(journal)
        candidates = [r['data'] for r in records if r['kind']=='provider_reasoning'
                      and r['data']['request_id']==request_id]
    else:
        # After successful reduction, every target's exact native record is in
        # its training row; the runtime journal is no longer a dependency.
        embedded = []
        for kind in ('actor', 'compact'):
            path = directory.parent/(kind+'.jsonl')
            if path.exists():
                with path.open() as stream:
                    for line in stream:
                        embedded.extend(json.loads(line).get('native_sources', []))
        if embedded:
            candidates = [s for s in embedded if s['request_id']==request_id]
    if candidates is not None:
        if len(candidates)!=1 or candidates[0]['response_id']!=response_id:
            raise ValueError('Provider reasoning response identity mismatch')
        value = deepcopy(candidates[0])
        return value, dict(storage='row.native_sources', request_id=request_id,
                           sha256=sha(encoded(value)), response_id=response_id)
    manifest = directory / 'manifest.json'
    if not manifest.exists():
        raise ValueError('Provider reasoning manifest missing')
    index = json.loads(manifest.read_bytes()); name = f'{request_id:06d}.json'
    ref = index['files'][name]; raw = (directory / name).read_bytes()
    if sha(raw) != ref['sha256']:
        raise ValueError('Provider reasoning source hash mismatch')
    value = json.loads(raw)
    if value['request_id'] != request_id or value['response_id'] != response_id or ref['response_id'] != response_id:
        raise ValueError('Provider reasoning response identity mismatch')
    return value, dict(path='provider_reasoning/' + name, sha256=sha(raw), response_id=response_id)
