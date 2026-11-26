"""Decode-source storage: continuous actor segments and real compactions only.

This is not a Swift renderer or a quality checker. No reasoning is decoded.
The forwarded request is authoritative; images are externalized once to JPEG.
"""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import hashlib
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import threading
try:
    from .append_journal import Journal, difference, apply_changes, register
except ImportError:
    from append_journal import Journal, difference, apply_changes, register


def digest(value):
    return hashlib.sha256(value).hexdigest()


def training_journal_path(root):
    root=Path(root)
    return root/'events.jsonl' if (root/'events.jsonl').exists() else root.parent/'.runtime/training.journal.jsonl'


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def publish_document(path, value):
    """Uncommitted output: readers must wait for the final completion marker."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n')
        stream.flush()
        os.fsync(stream.fileno())


def storage_mode(config=None):
    value = (config or {}).get('storage_mode', os.environ.get('ROBORSI_STORAGE_MODE', 'training'))
    if value not in ('training', 'raw'):
        raise ValueError('storage_mode must be training or raw')
    return value


def response_items(body, content_type, response):
    """Use the SSE items delivered to the CLI, retaining cipher alternatives."""
    done = []
    terminal = (response or {}).get('output', [])
    if 'text/event-stream' in content_type.lower():
        for frame in body.decode('utf-8-sig').replace('\r\n', '\n').split('\n\n'):
            data = '\n'.join(line[5:].lstrip() for line in frame.splitlines() if line.startswith('data:'))
            if not data or data == '[DONE]':
                continue
            event = json.loads(data)
            if event.get('type') == 'response.output_item.done':
                done.append(event['item'])
            elif event.get('type') == 'response.completed':
                terminal = event['response'].get('output', [])
    # Do not concatenate the terminal copy with the streamed copy.
    return done or terminal, terminal


def tool_schema(items, prefix=''):
    result = []
    for item in items:
        if item.get('type') == 'namespace':
            name = item['name'] + '.'
            result.extend(tool_schema(item['tools'], prefix + name))
        elif item.get('type') in ('function', 'custom'):
            fn = deepcopy(item.get('function', item))
            kind = fn.pop('type', 'function')
            fn['name'] = prefix + fn['name']
            if kind == 'custom':
                format = fn.pop('format', {})
                fn['description'] = fn.get('description', '') + '\nNative freeform input format: ' + json.dumps(format, ensure_ascii=False)
                fn['parameters'] = {'type': 'object', 'properties': {'input': {'type': 'string'}}, 'required': ['input'], 'additionalProperties': False}
            result.append({'type': 'function', 'function': fn})
        else:
            raise ValueError('Unsupported native tool declaration')
    return result


class TrainingStore:
    def __init__(self, root, *, metadata=None):
        self.root = Path(root)
        try:
            from .ephemeral_runtime import prepare_training
        except ImportError:
            from ephemeral_runtime import prepare_training
        ephemeral = prepare_training(self.root)
        if ephemeral is None:
            self.root.mkdir(parents=True, exist_ok=False)
        (self.root / 'images').mkdir()
        self.metadata = dict(metadata or {})
        if ephemeral is not None:
            self.metadata['ephemeral_runtime'] = ephemeral
        self.metadata.setdefault('storage_layout', 'append_v2')
        if self.metadata['storage_layout'] not in ('append_v2', 'legacy_v1'):
            raise ValueError('Unknown training storage layout')
        self.append_only = self.metadata['storage_layout'] == 'append_v2'
        if ephemeral is not None and not self.append_only:
            raise ValueError('Ephemeral training requires the append-only layout')
        if not self.append_only:
            (self.root / 'segments').mkdir()
        self.lock = threading.RLock()
        self.current = None
        self.rows = []
        self.image_map = {}
        self.seals = {}
        self.variant_written = set()
        self.closed = False
        self.native_reasoning = {}
        self._journal_states = {}
        self._pending_records = []
        self.journal = None
        if self.append_only:
            self.journal = Journal(self.root/'events.jsonl' if ephemeral is not None else
                                   self.root.parent/'.runtime/training.journal.jsonl')
            self.journal.append('header', dict(metadata=self.metadata, format='xpolicy_training_journal_v2'))
            register(self.journal)
        self._manifest()

    def _publish(self, path, value):
        (publish_document if self.append_only else write)(path, value)

    def _flush_journal(self):
        if self.append_only and self._pending_records:
            self.journal.append_many(self._pending_records)
            self._pending_records.clear()

    def _manifest(self):
        metadata = dict(self.metadata)
        if self.append_only and self.closed and metadata.pop('provider_reasoning_manifest', None):
            metadata['provider_reasoning_storage'] = 'actor_compact_native_sources'
        if self.append_only and self.closed and metadata.pop('provider_summary_manifest', None):
            metadata['provider_summary_storage'] = 'actor_compact_provider_summary_sources'
        self._publish(self.root / 'manifest.json', {
            'format': 'xpolicy_decode_source_v1', 'state': 'closed' if self.closed else 'open',
            'metadata': metadata, 'actor_rows': sum(r['kind'] == 'actor' for r in self.rows),
            'compact_rows': sum(r['kind'] == 'compact' for r in self.rows),
            'score': None, 'score_status': 'awaiting_native_result',
            'images': self.image_map, 'reasoning_decoded': False, 'quality_checked': False,
        })

    def _image(self, url):
        if not isinstance(url, str) or not url.startswith('data:image/') or ';base64,' not in url:
            raise ValueError('Training images require a captured data:image URL')
        header, payload = url.split(',', 1)
        raw = base64.b64decode(payload, validate=True)
        source_sha = digest(raw)
        if source_sha not in self.image_map:
            from .image_codec import encode_image_bit
            jpg = encode_image_bit(raw, quality=95, standard=True, subsampling='444')
            name = 'images/' + digest(jpg) + '.jpg'
            path = self.root / name
            if not path.exists():
                with path.open('xb') as stream:
                    stream.write(jpg)
                    stream.flush()
                    if self.append_only:
                        os.fsync(stream.fileno())
            elif digest(path.read_bytes()) != digest(jpg):
                raise ValueError('Existing training image is incomplete or changed')
            self.image_map[source_sha] = {'path': name, 'sha256': digest(jpg),
                'source_mime': header[5:].split(';')[0], 'source_sha256': source_sha,
                'jpeg_recompressed': not raw.startswith(b'\xff\xd8')}
            if self.append_only:
                self._pending_records.append(('image', dict(source_sha256=source_sha, value=self.image_map[source_sha])))
        return self.image_map[source_sha]['path']

    def _clean(self, value):
        if isinstance(value, list):
            return [self._clean(v) for v in value]
        if not isinstance(value, dict):
            return value
        value = deepcopy(value)
        if value.get('type') in ('input_image', 'image_url'):
            url = value['image_url']
            attributes = {k:self._clean(v) for k,v in value.items() if k not in ('type','image_url')}
            if isinstance(url, dict):
                attributes.update({k:v for k,v in url.items() if k != 'url'})
                url = url['url']
            return {**attributes,'type': 'image_paths', 'paths': [self._image(url)]}
        if value.get('type') == 'reasoning':
            # Source retention keeps every native field. Summaries remain
            # separate from explicit plaintext and opaque content.
            return value
        return {k: self._clean(v) for k, v in value.items()}

    def _messages(self, items, instructions):
        messages, pending, tools = [], None, []
        def flush():
            nonlocal pending
            if pending is not None:
                messages.append(pending)
                pending = None
        def assistant():
            nonlocal pending
            if pending is None:
                pending = {'role': 'assistant', 'content': [], 'reasoning_content': '',
                           'reasoning_items': [], 'tool_calls': [], 'item_order': []}
            return pending
        def content(value):
            if value is None:
                return []
            if isinstance(value, str):
                return [{'type': 'text', 'text': value}]
            result = []
            for p in value:
                if p.get('type') in ('input_text', 'output_text', 'text'):
                    result.append({'type': 'text', 'text': p['text']})
                elif p.get('type') == 'image_paths':
                    result.append(p)
                else:
                    # Source retention is not a training renderer. Preserve new
                    # native block types for the downstream decoder/adapter.
                    result.append(p)
            return result
        if instructions:
            messages.append({'role': 'system', 'content': instructions})
        for item in items:
            kind = item.get('type', 'message')
            if kind == 'additional_tools':
                tools.extend(item.get('tools', []))
                continue
            if kind == 'reasoning':
                a = assistant()
                a['item_order'].append({'type': kind, 'id': item.get('id'), 'reasoning_index': len(a['reasoning_items'])})
                a['reasoning_items'].append(item)
                from .provider_reasoning import plaintext
                a['reasoning_content'] += plaintext(item)
                if 'summary' in item:
                    a.setdefault('provider_summaries', []).append(deepcopy(item['summary']))
            elif kind in ('function_call', 'custom_tool_call'):
                a = assistant()
                name = item['name']
                if item.get('namespace'):
                    name = item['namespace'] + '.' + name
                arguments = {'input': item['input']} if kind == 'custom_tool_call' else item['arguments']
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        pass  # Invalid model arguments are still source data.
                a['item_order'].append({'type': kind, 'id': item.get('id'), 'tool_call_index': len(a['tool_calls'])})
                a['tool_calls'].append({'id': item['call_id'], 'type': 'function',
                                        'function': {'name': name, 'arguments': arguments}})
            elif kind in ('function_call_output', 'custom_tool_call_output'):
                flush()
                messages.append({'role': 'tool', 'tool_call_id': item['call_id'], 'content': content(item.get('output'))})
            elif kind == 'message' or 'role' in item:
                role = item['role']
                if role == 'assistant':
                    a = assistant()
                    parts = content(item.get('content'))
                    a['item_order'].append({'type': 'message', 'id': item.get('id'),
                        'channel': item.get('channel'), 'content_start': len(a['content']), 'content_count': len(parts)})
                    a['content'].extend(parts)
                    from .provider_reasoning import plaintext
                    a['reasoning_content'] += plaintext(item)
                else:
                    flush()
                    messages.append({'role': role, 'content': content(item.get('content'))})
            elif kind == 'compaction':
                a = assistant()
                a.setdefault('provider_items', []).append(item)
            else:
                flush()
                messages.append({'role': 'assistant', 'content': [], 'native_item': item})
        flush()
        return messages, tools

    def _row(self, state):
        row = {**self.metadata, 'format': 'xpolicy_decode_source_v1', 'kind': state['kind'],
               'segment_id': state['id'], 'boundary': state['boundary'], 'score': None,
               'score_status': 'awaiting_native_result', 'targets': deepcopy(state['targets']),
               'invocation_id': state['invocation'],
               'instructions': state['instructions'], 'model': state['model']}
        row['messages'], nested = self._messages(state['input'] + state['output'] + state.get('tail', []), state['instructions'])
        row['tools'] = tool_schema(state['tools'] + nested)
        row['native_tool_schema'] = state['tools'] + nested
        if self.append_only and self.metadata.get('provider_reasoning_manifest'):
            from .provider_reasoning import load
            row['native_sources'] = []
            for target in row['targets']:
                source, ref = load(self.root/'provider_reasoning', target['request_id'], target['response_id'])
                row['native_sources'].append(source)
                target['provider_reasoning_source'] = dict(storage='row.native_sources',
                    request_id=target['request_id'], response_id=target['response_id'], sha256=ref['sha256'])
            row.pop('provider_reasoning_manifest', None)
        if self.append_only and (self.metadata.get('provider_summary_manifest') or
                any(event['kind']=='provider_summary' for event in self.journal.records)):
            request_ids = {t['request_id'] for t in row['targets']}
            row['provider_summary_sources'] = [deepcopy(event['data']) for event in self.journal.records
                if event['kind']=='provider_summary' and event['data']['request_id'] in request_ids]
            if {s['request_id'] for s in row['provider_summary_sources']} != request_ids:
                raise ValueError('Provider summary source is missing; preserve runtime')
            row.pop('provider_summary_manifest', None)
        return row

    def _save(self, state):
        if self.append_only:
            # Targets/consumption may change in the previous segment too.
            # Keep only JSON changes, never another full historical request.
            snapshot = deepcopy(state)
            changes = difference(self._journal_states.get(state['id']), snapshot)
            if changes:
                self._pending_records.append(('segment', dict(id=state['id'], changes=changes)))
                self._journal_states[state['id']] = snapshot
        else:
            write(self.root / 'segments' / (state['id'] + '.json'), self._row(state))
            self._manifest()

    def _retain_seals(self, invocation, items, evidence):
        if self.append_only:
            for item in items:
                if item.get('type') != 'reasoning' or not item.get('encrypted_content'):
                    continue
                identity = (invocation, item.get('id'))
                versions = self.seals.setdefault(identity, {})
                seal = item['encrypted_content']
                if seal not in versions:
                    record = {**evidence, 'invocation_id':invocation, 'score':None, 'item':self._clean(item)}
                    versions[seal] = record
                    self._pending_records.append(('seal', dict(identity=list(identity), seal=seal, record=record)))
            return
        with (self.root/'cipher_variants.jsonl').open('a') as stream:
            for item in items:
                if item.get('type') != 'reasoning' or not item.get('encrypted_content'):
                    continue
                identity=(invocation,item.get('id'))
                versions=self.seals.setdefault(identity,{})
                versions.setdefault(item['encrypted_content'],{**evidence,'invocation_id':invocation,
                    'score':None,'item':self._clean(item)})
                if len(versions)>1 or evidence['source']=='native_session_unmapped':
                    for seal, record in versions.items():
                        key=(identity,seal)
                        if key not in self.variant_written:
                            stream.write(json.dumps(record,ensure_ascii=False)+'\n')
                            self.variant_written.add(key)

    def append(self, request, output, *, kind, request_id, response_id, alternatives=()):
        with self.lock:
            if self.closed:
                raise RuntimeError('Training store is closed')
            inputs = request.get('input', [])
            if isinstance(inputs, str):
                inputs = [{'type': 'message', 'role': 'user', 'content': inputs}]
            native_source = None
            output = deepcopy(output)
            if self.metadata.get('provider_reasoning_manifest'):
                from .provider_reasoning import load
                source, native_source = load(self.root/'provider_reasoning', request_id, response_id)
                byid = {v['item'].get('id'):v['item'] for v in source['native_items'] if v['item'].get('id')}
                byindex = {v['output_index']:v['item'] for v in source['native_items']}
                for index, item in enumerate(output):
                    native = byid.get(item.get('id'))
                    if native is None and item.get('id') is None:
                        candidate = byindex.get(index)
                        if candidate and candidate.get('id') is None and candidate.get('type') == item.get('type'):
                            native = candidate
                    if native is not None:
                        # Enrich the training copy only; the CLI wire is unchanged.
                        item.update(deepcopy(native))
                top = next(f for f in source['fields'] if f['path'] == ['reasoning_content'])
                from .provider_reasoning import plaintext
                if top['state'] != 'absent' and output and not any(plaintext(i) for i in output):
                    output[0]['native_response_reasoning_content'] = deepcopy(top['value'])
            # Earlier CLI histories may omit plaintext. Restore only exact item
            # identities within the same native invocation from retained outputs.
            meta = request.get('client_metadata') or {}
            turn = meta.get('x-codex-turn-metadata') or {}
            if isinstance(turn, str): turn = json.loads(turn)
            invocation_key = turn.get('session_id', turn.get('session'))
            for item in output:
                if item.get('id'):
                    self.native_reasoning[(invocation_key,item['id'])] = deepcopy(item)
            inputs = [{**deepcopy(self.native_reasoning.get((invocation_key,i.get('id')),{})), **i}
                      if i.get('type') in ('reasoning','message') else i for i in inputs]
            inputs, output = self._clean(inputs), self._clean(output)
            context = {'instructions': request.get('instructions') or '', 'tools': self._clean(request.get('tools', [])),
                       'model': request.get('model')}
            metadata = request.get('client_metadata') or {}
            native = metadata.get('x-codex-turn-metadata', {})
            if isinstance(native, str):
                native = json.loads(native)
            invocation = native.get('session_id', native.get('session'))
            current = self.current
            if current is not None:
                self._tail(current, inputs)
            def comparable(items):
                result = []
                for item in items:
                    ignored = {'status','metadata','internal_chat_message_metadata_passthrough'}
                    if item.get('type') == 'reasoning' and item.get('id'):
                        ignored.add('encrypted_content')  # All seal variants are retained below.
                    result.append({k:v for k,v in item.items() if k not in ignored})
                return result
            prefix = current['input'] + current['output'] if current is not None else []
            prefix_matches = current is not None and comparable(inputs[:len(prefix)]) == comparable(prefix)
            if prefix_matches:
                current['targets'][-1]['native_consumed'] = True
                current['targets'][-1]['consumption_source'] = 'subsequent_request'
                self._save(current)
            continuing = (kind == 'actor' and current is not None
                and all(current[k] == context[k] for k in context)
                and current['invocation'] == invocation
                and prefix_matches)
            if not continuing:
                boundary = ('actual_compaction' if kind == 'compact' else 'initial' if not self.rows
                            else 'after_compaction' if self.rows[-1]['kind']=='compact' else 'context_boundary')
                state = {**context, 'input': inputs, 'output': output, 'targets': [],
                         'kind': kind, 'id': f'{len(self.rows)+1:06d}', 'boundary': boundary, 'invocation': invocation}
                self.rows.append({'kind': kind, 'id': state['id']})
            else:
                state = current
                state['input'], state['output'] = inputs, output
            state['tail'] = []
            state['targets'].append({'request_id': request_id, 'response_id': response_id,
                'output_item_ids': [i['id'] for i in output if i.get('id')], 'native_consumed': None,
                **({'provider_reasoning_source':native_source} if native_source else {}),
                'generation': {k:request[k] for k in ('reasoning','max_output_tokens','temperature','top_p',
                    'tool_choice','parallel_tool_calls','text','include') if k in request}})
            self.current = state if kind == 'actor' else None
            self._save(state)
            # Keep distinct terminal/done ciphertext versions without duplicating
            # full responses or full historical requests.
            for source, items in (('request.input',inputs),('output_item.done',output),('response.completed',alternatives)):
                self._retain_seals(invocation,items,{'request_id':request_id,'response_id':response_id,
                    'segment_id':state['id'],'source':source})
            self._flush_journal()
            return {'kind': kind, 'segment_id': state['id']}

    def _tail(self, state, items):
        calls = {i['call_id'] for i in state['output'] if i.get('type') in ('function_call', 'custom_tool_call')}
        tail = [i for i in items if i.get('type') in ('function_call_output', 'custom_tool_call_output') and i.get('call_id') in calls]
        if tail:
            state['tail'] = tail
            self._save(state)

    def close(self, *, session_root=None, capture_errors=(), closure_evidence=None):
        with self.lock:
            if self.closed:
                return
            consumed, native = set(), []
            gaps = [{'kind': 'capture_persistence_error', 'error_type': name}
                    for name in capture_errors]
            calls = {i['call_id'] for i in self.current['output'] if i.get('call_id')} if self.current else set()
            if session_root is not None:
                # Only this runtime's private sessions, never global ~/.codex.
                for path in sorted(Path(session_root).rglob('sessions/**/*.jsonl')):
                    session_id=None
                    for number, line in enumerate(path.open(), 1):
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            gaps.append({'path': str(path), 'line': number, 'kind': 'partial_native_line'})
                            continue
                        if row.get('type')=='session_meta':
                            session_id=(row.get('payload') or {}).get('id')
                        if row.get('type') == 'response_item':
                            item = row.get('payload') or {}
                            if item.get('id'):
                                consumed.add(item['id'])
                            if item.get('type')=='reasoning' and item.get('encrypted_content'):
                                keys=[key for key in self.seals if key[1]==item.get('id')]
                                invocation=session_id
                                if (invocation,item.get('id')) not in self.seals and len(keys)==1:
                                    invocation=keys[0][0]
                                source='native_session' if (invocation,item.get('id')) in self.seals else 'native_session_unmapped'
                                self._retain_seals(invocation,[item],{'source':source,
                                    'native_file':str(path.relative_to(session_root)),'native_line':number})
                            if item.get('type') in ('function_call_output', 'custom_tool_call_output') and item.get('call_id') in calls:
                                native.append(item)
            if self.current is not None:
                # Match exact call IDs, retaining the native order and images.
                self._tail(self.current, self._clean([i for i in native if i.get('call_id') in calls]))
            self._flush_journal()
            for kind in ('actor', 'compact'):
                path = self.root / (kind + '.jsonl')
                temporary = path if self.append_only else path.with_suffix('.jsonl.pending')
                with temporary.open('w') as stream:
                    for ref in self.rows:
                        if ref['kind'] == kind:
                            row = (self._row(self._journal_states[ref['id']]) if self.append_only else
                                   json.loads((self.root / 'segments' / (ref['id'] + '.json')).read_text()))
                            for target in row['targets']:
                                ids = target['output_item_ids']
                                if kind == 'actor' and not target['native_consumed']:
                                    target['native_consumed'] = bool(ids and set(ids) <= consumed)
                                    if target['native_consumed']:target['consumption_source']='native_session'
                            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n')
                    stream.flush()
                    if self.append_only:
                        os.fsync(stream.fileno())
                if not self.append_only:
                    os.replace(temporary, path)
            from .training_quality import audit_rows, rows_from_directory
            quality=audit_rows(rows_from_directory(self.root),terminal=closure_evidence)
            write(self.root/'quality.json',quality)
            self.metadata['training_quality']={'schema_version':1,'path':'quality.json',
                'sha256':digest((self.root/'quality.json').read_bytes()),
                'uncompleted_calls':quality['unpaired_call_ids'],
                'mask_consumer':'scripts/roborsi_training_view.py'}
            if self.append_only:
                variants = [record for versions in self.seals.values() for record in versions.values()
                            if len(versions)>1 or record['source']=='native_session_unmapped']
                if variants:
                    with (self.root/'cipher_variants.jsonl').open('w') as stream:
                        for record in variants:
                            stream.write(json.dumps(record,ensure_ascii=False)+'\n')
                        stream.flush()
                        os.fsync(stream.fileno())
            self.closed = True
            try:
                self._manifest()
                self._publish(self.root/'native-evidence.json', {'consumed_output_item_ids': sorted(consumed),
                    'gaps': gaps, 'session_root': str(session_root) if session_root else None,
                    'uncompleted_calls':quality['unpaired_call_ids'],
                    'uncompleted_calls_are_not_cancelled_without_receipt':True,
                    **({'api_attempts':[dict(kind=event['kind'],**deepcopy(event['data']))
                        for event in self.journal.records if event['kind'] in ('api_intent','api_result')]}
                       if self.append_only else {})})
            except BaseException:
                # Rows and segments remain recoverable if final publication
                # runs out of space. The same capture must be able to retry.
                self.closed = False
                raise
            if self.append_only:
                self.journal.close()
            else:
                shutil.rmtree(self.root / 'segments')

    @classmethod
    def recover(cls, root, *, expected_metadata):
        """Recover the same source after its writer exited; never create a new budget."""
        root = Path(root)
        writer = Journal(training_journal_path(root))
        try:
            header = writer.records[0]
            assert header['kind']=='header' and header['data']['format']=='xpolicy_training_journal_v2'
            metadata = header['data']['metadata']
            if any(key in metadata and key not in expected_metadata
                   for key in ('model', 'task_id', 'source_commit', 'harness_commit')):
                raise ValueError('Recovery requires the original source/model identity')
            for key, value in expected_metadata.items():
                if metadata.get(key) != value:
                    raise ValueError('Recovery source/model identity changed: '+key)
            self = cls.__new__(cls)
            self.root, self.metadata, self.append_only = root, metadata, True
            self.lock = threading.RLock()
            self.current, self.closed = None, False
            self.rows, self.image_map, self.seals = [], {}, {}
            self.variant_written, self.native_reasoning = set(), {}
            self._journal_states, self._pending_records = {}, []
            self.journal = writer
            register(writer)
            for event in writer.records[1:]:
                data = event['data']
                if event['kind']=='image':
                    image_path = root/data['value']['path']
                    if digest(image_path.read_bytes()) != data['value']['sha256']:
                        raise ValueError('Recovery image is incomplete or changed')
                    self.image_map[data['source_sha256']] = data['value']
                elif event['kind']=='segment':
                    state = apply_changes(self._journal_states.get(data['id']), data['changes'])
                    if data['id'] not in self._journal_states:
                        self.rows.append(dict(id=state['id'], kind=state['kind']))
                    self._journal_states[data['id']] = state
                elif event['kind']=='seal':
                    self.seals.setdefault(tuple(data['identity']), {})[data['seal']] = data['record']
            for ref in self.rows:
                state = self._journal_states[ref['id']]
                for item in state['input']+state['output']:
                    if item.get('id'):
                        self.native_reasoning[(state['invocation'],item['id'])] = deepcopy(item)
            if self.rows and self.rows[-1]['kind']=='actor':
                self.current = deepcopy(self._journal_states[self.rows[-1]['id']])
            return self
        except BaseException:
            writer.close()
            raise


def inside_case_trajectories(path, case):
    case=Path(case).resolve()
    path=Path(path).resolve()
    normal = any(path.is_relative_to(parent) for parent in
                 (case/'trajectories',case/'.runtime/trajectories'))
    if normal:return True
    try:
        from .ephemeral_runtime import bound_runtime
    except ImportError:
        from ephemeral_runtime import bound_runtime
    return bound_runtime(path,case)


def finalize(root, native_result):
    """One terminal publisher; an exited capture cannot be finalized twice concurrently."""
    root = Path(root)
    directory = root.parent/'.runtime'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'training-finalize.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _finalize(root, native_result)


def _finalize(root, native_result):
    """Attach only an external native score; never infer it from model text."""
    root = Path(root)
    result = json.loads(Path(native_result).read_text())
    metrics = result.get('metrics', result.get('native_metrics'))
    if result.get('done') is not True or not isinstance(metrics, dict):
        raise ValueError('A completed native terminal result is required')
    score = metrics.get('score')
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Native score must be a finite value in [0,1]')
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['state'] != 'closed':
        raise ValueError('Close the capture before attaching a native score')
    # Native terminal + owner must be from the same case and scene. The score
    # lives outside the model input and can only be attached by the supervisor.
    runtime = root.parent.parent
    try:
        from .hybrid_resume import read_json
    except ImportError:
        from hybrid_resume import read_json
    owner = read_json(runtime/'episode-owner.json')
    case = Path(native_result).resolve().parent
    if not inside_case_trajectories(runtime, case) or not owner.get('closed'):
        raise ValueError('Native result and closed trajectory owner must share a case')
    identity = owner.get('native_identity', {})
    task = manifest['metadata'].get('task_id')
    if task and identity.get('task') != task:
        raise ValueError('Native task differs from captured task')
    if 'layout_id' in metrics and metrics['layout_id'] != identity.get('layout'):
        raise ValueError('Native layout differs from captured episode')
    native = case/'native-identity.json'
    if native.is_file():
        saved = json.loads(native.read_text())
        if identity.get('scene_sha256') != saved.get('layout_sha256'):
            raise ValueError('Native scene differs from captured episode')
    append_only = manifest['metadata'].get('storage_layout')=='append_v2'
    native_sha = digest(Path(native_result).read_bytes())
    if manifest.get('native_result', {}).get('sha256', native_sha) != native_sha:
        raise ValueError('Recorded native result changed; preserve existing source')
    committed = root/'COMMITTED'
    if append_only and committed.exists():
        raw_commit = committed.read_bytes()
        if not raw_commit.endswith(b'\n'):
            # An exited writer's incomplete final marker never published data.
            committed.unlink()
            receipt = None
        else:
            receipt = json.loads(raw_commit)
        if receipt is not None:
            if (manifest.get('native_result', {}).get('sha256') != native_sha
                    or not valid_commit(root, manifest)):
                raise ValueError('Committed native result changed; preserve existing source')
            return manifest
    for name in ('actor.jsonl', 'compact.jsonl', 'cipher_variants.jsonl'):
        path = root / name
        if not path.exists():
            continue
        # One replacement at the terminal boundary, never per step/request.
        # Keep the complete previous file if scoring is interrupted midway.
        temporary = path.with_name(name + '.pending')
        with path.open() as source, temporary.open('w') as target:
            for line in source:
                row = json.loads(line)
                row.update(score=score, score_status='native_terminal', success=metrics.get('success'))
                target.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    manifest.update(score=score, score_status='native_terminal', native_metrics=metrics,
        native_result={'path': str(native_result), 'sha256': native_sha})
    write(root / 'manifest.json', manifest)
    if append_only:
        evidence = root/'native-evidence.json'
        if (not _training_images_complete(root, manifest) or not evidence.is_file()
                or json.loads(evidence.read_bytes()).get('gaps')):
            # The real physical result remains valid; training publication is
            # independently blocked, and recovery material must remain intact.
            manifest['training_commit_status'] = 'blocked_incomplete_source'
            write(root/'manifest.json', manifest)
            return manifest
        receipt = dict(format='xpolicy_training_commit_v2', native_result_sha256=native_sha,
                       manifest_sha256=digest((root/'manifest.json').read_bytes()),
                       files={name:digest((root/name).read_bytes()) for name in
                              sorted(_commit_members(root,manifest))})
        with committed.open('x') as stream:
            stream.write(json.dumps(receipt,separators=(',', ':'))+'\n')
            stream.flush()
            os.fsync(stream.fileno())
    return manifest


def _commit_members(root,manifest):
    root=Path(root);members={'actor.jsonl','compact.jsonl','native-evidence.json'}
    if (root/'cipher_variants.jsonl').exists():members.add('cipher_variants.jsonl')
    quality=(manifest.get('metadata')or{}).get('training_quality')
    if quality:
        path=root/'quality.json'
        if (quality.get('schema_version')!=1 or quality.get('path')!='quality.json'
                or path.is_symlink()or not path.is_file()
                or digest(path.read_bytes())!=quality.get('sha256')):
            raise ValueError('Training quality file is missing or differs from its manifest')
        members.add('quality.json')
    return members


def valid_commit(root, manifest=None):
    root = Path(root)
    try:
        manifest = manifest or json.loads((root/'manifest.json').read_bytes())
        raw = (root/'COMMITTED').read_bytes()
        if not raw.endswith(b'\n'):
            return False
        receipt = json.loads(raw)
        expected = _commit_members(root,manifest)
        if (receipt.get('format') != 'xpolicy_training_commit_v2'
                or set(receipt['files']) != expected
                or any((root/name).is_symlink() for name in expected)
                or receipt['native_result_sha256'] != manifest['native_result']['sha256']
                or receipt['manifest_sha256'] != digest((root/'manifest.json').read_bytes())
                or any(digest((root/name).read_bytes())!=sha for name,sha in receipt['files'].items())):
            return False
        return _training_images_complete(root, manifest)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def finalize_case(case, native_result):
    case = Path(case)
    stores = [*case.glob('trajectories/*/seed-*/*/runtime-*/agent/training'),
              *case.glob('.runtime/trajectories/*/seed-*/*/runtime-*/agent/training')]
    for root in stores:
        finalize(root, native_result)
    for root in stores:
        compact_closed_runtime(root.parent.parent)
    return len(stores)


def compact_closed_runtime(runtime):
    """Terminal-only reduction. Interrupted recovery state is never removed."""
    runtime = Path(runtime)
    try:
        from .hybrid_resume import read_json
    except ImportError:
        from hybrid_resume import read_json
    owner = read_json(runtime/'episode-owner.json')
    manifest = runtime / 'agent/training/manifest.json'
    pending = runtime/'archive.pending.json'
    if (not owner.get('closed') or not pending.is_file()
        or not json.loads(pending.read_text()).get('native_terminal_accepted')
        or not manifest.is_file() or json.loads(manifest.read_text()).get('score_status') != 'native_terminal'):
        return False
    pending_record = json.loads(pending.read_text())
    saved = json.loads(manifest.read_text())
    training = manifest.parent
    if saved['metadata'].get('storage_layout')=='append_v2' and not valid_commit(training, saved):
        return False
    if saved.get('state') != 'closed' or not _training_images_complete(training, saved):
        return False
    trajectory = runtime.parent
    declared = pending_record.get('trajectory_directory')
    if declared and Path(declared).resolve() != trajectory.resolve():
        return False
    case = Path(saved['native_result']['path']).resolve().parent
    if not inside_case_trajectories(trajectory, case):
        return False
    # Controller observations may be shared by several continuation runtimes.
    # Never prune them while another owner can still be reading or writing.
    for sibling in trajectory.glob('runtime-*'):
        sibling_owner = sibling/'episode-owner.json'
        if not sibling_owner.is_file() or not json.loads(sibling_owner.read_text()).get('closed'):
            return False
        sibling_training = sibling/'agent/training'
        sibling_manifest = sibling_training/'manifest.json'
        if not sibling_manifest.is_file():
            return False
        sibling_saved = json.loads(sibling_manifest.read_text())
        if (sibling_saved.get('state') != 'closed'
            or sibling_saved.get('score_status') != 'native_terminal'
            or (sibling != runtime and not _training_images_complete(sibling_training, sibling_saved))):
            return False
        sibling_evidence = sibling_training/'native-evidence.json'
        if not sibling_evidence.is_file() or json.loads(sibling_evidence.read_text()).get('gaps'):
            return False
    receipt = runtime/'storage-receipt.json'
    removed = json.loads(receipt.read_text())['removed'] if receipt.is_file() else []
    for relative in ('agent/cli-input', 'agent/prompt-input', 'agent/codex-home', 'agent/workspace',
                     'agent/controller/checkpoints', 'agent/controller/segments',
                     'workspace', 'prepared-skills', 'codex-home'):
        path = runtime / relative
        if path.is_symlink():
            path.unlink()
            removed.append(relative)
        elif path.is_dir():
            shutil.rmtree(path)
            removed.append(relative)
    # The canonical training image set is now self-contained. Native per-step
    # replay images are regenerable; retain the action/state/ACK text journals.
    for observations in (runtime/'agent/controller/observations', trajectory/'controller/observations'):
        if not observations.is_dir() or observations.is_symlink():
            continue
        if not inside_case_trajectories(observations, case):
            continue
        for directory, directories, files in os.walk(observations, followlinks=False):
            directories[:] = [name for name in directories if not (Path(directory)/name).is_symlink()]
            for name in files:
                path = Path(directory)/name
                if not path.is_symlink() and path.suffix.lower() in ('.jpg','.jpeg','.png','.webp'):
                    path.unlink()
                    removed.append(os.path.relpath(path, runtime))
    write(receipt, {'storage_mode':'training', 'removed':removed,
        'canonical_images':'agent/training/images', 'native_score_attached':True})
    return True


def _training_images_complete(root, manifest):
    """Require intact, self-contained training rows before pruning observations."""
    try:
        if (root/'images').is_symlink():
            return False
        known = {}
        for item in manifest.get('images', {}).values():
            relative = Path(item['path'])
            path = root/relative
            if (relative.is_absolute() or '..' in relative.parts or path.is_symlink()
                or not path.resolve().is_relative_to((root/'images').resolve())
                or not path.is_file() or digest(path.read_bytes()) != item['sha256']):
                return False
            known[item['path']] = item

        def references(value):
            if isinstance(value, dict):
                if value.get('type') == 'image_paths':
                    if any(path not in known for path in value['paths']):
                        raise ValueError('Unresolved training image')
                for item in value.values():
                    references(item)
            elif isinstance(value, list):
                for item in value:
                    references(item)

        for kind in ('actor', 'compact'):
            rows = 0
            with (root/(kind+'.jsonl')).open() as stream:
                for line in stream:
                    references(json.loads(line))
                    rows += 1
            if rows != manifest[kind+'_rows']:
                return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Attach an official native result to compact training-source rows')
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--native-result', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({'stores_finalized': finalize_case(args.case, args.native_result)}))
