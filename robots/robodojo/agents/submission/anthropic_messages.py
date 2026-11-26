"""Exact Fable Messages transport inside the existing capture; no extra server/planner.

Responses-facing records are a declared protocol projection. Authoritative native
Messages requests/responses, real thinking summaries and signatures remain separate.
Only this model's capture is affected; the frozen tools and executor are unchanged.
"""
from __future__ import annotations
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path
import os
import asyncio,hashlib,json,re,time
import httpx

MODEL='claude-fable-5-2/anthropic_S/sfp'
ROUTE_MODELS=(MODEL,'claude-fable-5-2/anthropic_L/sfo')
ROUTE_ATTEMPT_LIMIT=3
NATIVE_OUTPUT_LIMIT=65536  # Required Messages API envelope, not an episode/Native budget.
DOCS='https://platform.claude.com/docs/en/build-with-claude/thinking'
PREFIX_DOCS='https://platform.claude.com/docs/en/build-with-claude/preserved-thinking'
BINDING_BETA='thinking-binding-controls-2026-08-01'
STATE_FORMAT='fable_native_transport_state_v3'
HISTORY_CACHE_POLICY='rolling_user_history_ephemeral_v1'

def _without_cache_control(value):
 if isinstance(value,dict):return {k:_without_cache_control(v)for k,v in value.items()if k!='cache_control'}
 if isinstance(value,list):return [_without_cache_control(v)for v in value]
 return value

def cache_message_history(native):
 """Cache growing history without changing content or signed thinking.

 Keep the preceding user boundary explicit as well as the newest boundary,
 so the previous cache write remains reachable beyond the 20-block lookback.
 Static tools/system consume at most two of the four available breakpoints.
 """
 before=encoded(_without_cache_control(native));boundaries=[]
 for message in native['messages']:
  if message['role']!='user':continue
  eligible=[b for b in message['content']if b.get('type')in('text','image','tool_result')and(b.get('type')!='text'or b.get('text'))]
  if eligible:boundaries.append(eligible[-1])
 for block in boundaries[-2:]:block['cache_control']=dict(type='ephemeral')
 if encoded(_without_cache_control(native))!=before:raise ValueError('History cache changed native request content')

def history_cache_metadata(native):
 paths=[]
 for section in ['tools','system']:
  for index,block in enumerate(native.get(section,[])):
   if block.get('cache_control'):paths.append(dict(path=f'{section}.{index}',type=block.get('type','tool'),cache_control=deepcopy(block['cache_control'])))
 for mi,message in enumerate(native['messages']):
  for bi,block in enumerate(message['content']):
   if block.get('cache_control'):paths.append(dict(path=f'messages.{mi}.content.{bi}',role=message['role'],type=block['type'],cache_control=deepcopy(block['cache_control'])))
 if len(paths)>4:raise ValueError('Native request exceeds four cache breakpoints')
 return dict(policy=HISTORY_CACHE_POLICY,breakpoints=paths,semantic_payload_sha256=sha(encoded(_without_cache_control(native))),thinking_signatures_modified=False)

async def operator_pause_boundary(owner,logical_request_sha256):
 """Wait inside the original request; never resend an unresolved paused call."""
 case=Path(owner.case).resolve()
 scope=next((p for p in case.parents if p.parent.name=='cases'),None)
 if scope is None:return
 marker=scope.parent.parent.parent/'infra/runner/operator-pause-history-cache.json'
 pending=set(getattr(owner,'_operator_pause_inflight_request_ids',()))
 original=getattr(owner,'_operator_pause_original_request_sha256',None)
 duplicate=bool(original and logical_request_sha256==original)
 unknown=next((row for row in reversed(owner.rows)if row.get('request_id')in pending and
               (row.get('native_outcome_unknown')or row.get('transport_error')or row.get('recoverable_capture_error'))),None)
 quarantine=case/'operator-pause-unresolved.json'
 if (unknown or duplicate)and not quarantine.exists():
  save(quarantine,dict(at=time.time(),reason='paused_inflight_model_request_outcome_unresolved',
       original_request_id=unknown['request_id']if unknown else getattr(owner,'_operator_pause_original_request_id',None),provider_request_id=unknown.get('provider_request_id')if unknown else None,
       logical_request_sha256=logical_request_sha256,original_paused_request_reissue=duplicate,
       outcome_unknown=True,billing_unknown=True,automatic_resend=False,episode_unchanged=True))
 while marker.exists()or quarantine.exists():await asyncio.sleep(0.25)

def sha(raw):return hashlib.sha256(raw).hexdigest()
def normalized_native_usage(u):
 if not isinstance(u,dict):return None
 base=u.get('input_tokens');read=u.get('cache_read_input_tokens');write=u.get('cache_creation_input_tokens');output=u.get('output_tokens');usage=dict(anthropic_native=deepcopy(u))
 if all(type(v)is int for v in [base,read,write]):usage['input_tokens']=base+read+write
 if type(output)is int:usage['output_tokens']=output
 if 'input_tokens'in usage and 'output_tokens'in usage:usage['total_tokens']=usage['input_tokens']+output
 if type(read)is int:usage['input_tokens_details']=dict(cached_tokens=read)
 if type(write)is int:usage['cache_write_tokens']=write
 return usage
def encoded(value):return json.dumps(value,ensure_ascii=False,separators=(',',':')).encode()
def journal_states(journal):
 if not hasattr(journal,'fable_states'):
  from .append_journal import apply_changes
  journal.fable_states={}
  for event in journal.records:
   if event['kind']=='fable_state':
    data=event['data'];journal.fable_states[data['name']]=apply_changes(journal.fable_states.get(data['name']),data['changes'])
 return journal.fable_states

def save(path,value,journal=None):
 path=Path(path)
 if journal is not None:
  # Whole native requests include repeated images/history. The training source
  # already owns that content; only its hash and causal state are needed here.
  if path.name.endswith(('.request.json','.tool-map.json')):return
  from .append_journal import difference
  states=journal_states(journal);changes=difference(states.get(path.name),value)
  if changes:journal.append('fable_state',dict(name=path.name,changes=changes));states[path.name]=deepcopy(value)
  return
 path.parent.mkdir(parents=True,exist_ok=True)
 q=path.with_name(path.name+'.pending');q.write_bytes(encoded(value));q.replace(path)

def request_kind(raw):
 metadata=raw.get('client_metadata')or{}
 turn=metadata.get('x-codex-turn-metadata')or{}
 if isinstance(turn,str):turn=json.loads(turn)
 return 'compaction' if metadata.get('request_kind')=='compaction' or turn.get('request_kind')=='compaction' else 'actor'

def all_declarations(raw):
 tools=deepcopy(raw.get('tools')or[])
 for item in raw.get('input',[]) if isinstance(raw.get('input'),list) else []:
  if item.get('type')=='additional_tools':tools+=item.get('tools')or[]
 return declarations(tools)

def resolve_registry(raw, previous=()):
 """Keep history aliases across compact without granting unknown tools.

 The summarizer retains the prior native declaration prefix, with tool_choice
 none. Tool availability to the actor still comes only from its own request.
 """
 registry=deepcopy(list(previous));byfull={x['full_name']:x for x in registry}
 incoming=all_declarations(raw)
 for spec in incoming:
  old=byfull.get(spec['full_name'])
  if old is not None:
   spec['wire_name']=old['wire_name']
   registry[registry.index(old)]=spec
  else:
   if any(x['wire_name']==spec['wire_name'] for x in registry):
    spec['wire_name']='tool_'+sha(spec['full_name'].encode())[:40]
   registry.append(spec)
  byfull[spec['full_name']]=spec
 assert len({x['wire_name']for x in registry})==len(registry),'Tool aliases collided'
 if request_kind(raw)=='compaction':active=registry
 else:active=[byfull[x['full_name']]for x in incoming]
 return active,registry

def _without_thinking(value):
 if isinstance(value,dict):return {k:_without_thinking(v)for k,v in value.items()if k!='cache_control'}
 if isinstance(value,list):return [_without_thinking(x)for x in value if not(isinstance(x,dict)and x.get('type')in('thinking','redacted_thinking'))]
 return value

def prefix_hash(native,messages):
 # Earlier signed blocks are not themselves part of the provider prefix check.
 return sha(encoded(_without_thinking(dict(system=native.get('system',[]),tools=native.get('tools',[]),messages=messages))))

def _prefix_before(messages,mi,bi):
 prefix=deepcopy(messages[:mi]);blocks=deepcopy(messages[mi]['content'][:bi])
 if blocks:prefix.append(dict(role=messages[mi]['role'],content=blocks))
 return prefix

def audit_prefix(native,ledger):
 findings=[]
 for mi,message in enumerate(native['messages']):
  for bi,block in enumerate(message['content']):
   if block['type']not in ('thinking','redacted_thinking'):continue
   signature=block.get('signature',block.get('data'));entry=ledger.get(sha(signature.encode()))
   if entry is None:raise ValueError('Unknown native thinking signature')
   expected=entry.get('prefix_sha256');actual=prefix_hash(native,_prefix_before(native['messages'],mi,bi))
   if expected and actual!=expected:
    findings.append(dict(path=f'messages.{mi}.content.{bi}',reason='local_prefix_binding_mismatch',expected_sha256=expected,actual_sha256=actual))
   elif not expected:findings.append(dict(path=f'messages.{mi}.content.{bi}',reason='prefix_binding_not_recorded'))
 return findings

def session_binding(owner):
 p=Path(owner.case).parent/'episode-owner.json'
 if not p.is_file():return dict(model=MODEL,fixture_or_unbound_runtime=str(Path(owner.case).resolve()))
 value=json.loads(p.read_bytes());identity=value.get('identity')or{}
 episode=value.get('trajectory_id')or value.get('episode_id')
 if not episode:raise ValueError('Native owner has no episode identity')
 return dict(model=MODEL,episode_id=episode,task=identity.get('task'),eval_seed=identity.get('eval_seed'),layout=identity.get('layout'))

def load_state(owner,directory,sources=(),journal=None):
 """Only explicit, hash-pinned state from the same native episode may migrate."""
 binding=session_binding(owner);path=directory/'transport-state.json'
 state=dict(format=STATE_FORMAT,binding=binding,tool_registry=[],thinking_ledger={},inherited_sources=[])
 candidates=list(sources);values=[];current=None
 if journal is not None:
  current=journal_states(journal).get(path.name)
 elif path.exists():candidates.append(dict(path=str(path),sha256=sha(path.read_bytes())))
 for source in candidates:
  p=Path(source['path']);raw=p.read_bytes()
  if sha(raw)!=source['sha256']:raise ValueError('Resume transport state hash mismatch')
  values.append((p,raw,json.loads(raw)))
 if current is not None:values.append((None,None,current))
 for p,raw,old in values:
  if old.get('format')!=STATE_FORMAT or old.get('binding')!=binding:raise ValueError('Resume transport state belongs to a different model/episode')
  if sources and 'episode_id'not in binding:raise ValueError('Cannot inherit an unbound native episode')
  for field in ('credential_binding_sha256','prefix_policy','route_policy_version','route_request_sha256','route_request_completed','route_request_runtime','previous_exhausted_route_round','route_attempts','next_route_model','case_channel_pin'):
   if field in old:
    if field=='credential_binding_sha256'and field in state and state[field]!=old[field]:raise ValueError('Resume credential binding conflict')
    state[field]=deepcopy(old[field])
  state['inherited_sources'].extend(old.get('inherited_sources',[]))
  byfull={x['full_name']:x for x in state['tool_registry']}
  for item in old['tool_registry']:
   if item['full_name']in byfull and byfull[item['full_name']]!=item:raise ValueError('Resume tool registry conflict')
   if item['full_name']not in byfull:state['tool_registry'].append(item);byfull[item['full_name']]=item
  for key,value in old['thinking_ledger'].items():
   if key=='__output_blocks__':
    target=state['thinking_ledger'].setdefault(key,{})
    for k,v in value.items():
     if k in target and target[k]!=v:raise ValueError('Resume native output block conflict')
     target[k]=v
   else:
    if key in state['thinking_ledger']and state['thinking_ledger'][key]!=value:raise ValueError('Resume thinking ledger conflict')
    state['thinking_ledger'][key]=value
  if p is not None and str(p)!=str(path):state['inherited_sources'].append(dict(path=str(p),sha256=sha(raw)))
 state['inherited_sources']=list({(x['path'],x['sha256']):x for x in state['inherited_sources']}.values())
 return state

def declarations(items,prefix=''):
 result=[]
 for item in items:
  kind=item['type']
  if kind=='namespace':result+=declarations(item['tools'],prefix+item['name']+'.')
  elif kind in ['function','custom']:
   full=prefix+item['name'];name=re.sub(r'[^A-Za-z0-9_-]','__',full)
   if len(name)>110:name=name[:80]+'_'+sha(full.encode())[:24]
   params=deepcopy(item.get('parameters'))if kind=='function'else dict(type='object',properties={'_raw':dict(type='string')},required=['_raw'],additionalProperties=False)
   result.append(dict(full_name=full,wire_name=name,kind=kind,description=item.get('description',''),parameters=params))
  else:raise ValueError('Unmapped tool declaration: '+str(kind))
 assert len({x['full_name']for x in result})==len(result)
 # Distinct source names may have the same printable flattening; never alias them.
 counts={x['wire_name']:sum(y['wire_name']==x['wire_name']for y in result)for x in result}
 for x in result:
  if counts[x['wire_name']]>1:x['wire_name']='tool_'+sha(x['full_name'].encode())[:40]
 assert len({x['wire_name']for x in result})==len(result)
 return result

def content(parts):
 if isinstance(parts,str):return [dict(type='text',text=parts)]
 out=[]
 for p in parts or []:
  t=p['type']
  if t in ['input_text','output_text','text']:out.append(dict(type='text',text=p['text']))
  elif t=='input_image':
   url=p['image_url'];url=url['url']if isinstance(url,dict)else url
   if url.startswith('data:'):
    head,data=url.split(',',1);assert head.endswith(';base64')
    out.append(dict(type='image',source=dict(type='base64',media_type=head[5:-7],data=data)))
   elif url.startswith(('https://','http://')):out.append(dict(type='image',source=dict(type='url',url=url)))
   else:raise ValueError('Unsupported image transport')
  else:raise ValueError('Unmapped public content: '+str(t))
 return out

def request_to_messages(raw,ledger,registry=()):
 assert raw['model']==MODEL and raw.get('reasoning',{}).get('effort')=='xhigh'
 assert not raw.get('previous_response_id'),'Full explicit history required'
 items=raw.get('input',[])
 if isinstance(items,str):items=[dict(role='user',content=items)]
 items=deepcopy(items);system=[];messages=[]
 if raw.get('instructions'):system.append(dict(type='text',text=raw['instructions']))
 active,history=resolve_registry(raw,registry);byfull={x['full_name']:x for x in history}
 blocks=ledger.get('__output_blocks__',{})
 def append(role,blocks):
  if not blocks:return
  if messages and messages[-1]['role']==role:messages[-1]['content']+=blocks
  else:messages.append(dict(role=role,content=blocks))
 for item in items:
  kind=item.get('type');role=item.get('role')
  if kind=='additional_tools':continue
  if kind=='reasoning':
   signature=item.get('encrypted_content')
   if signature:
    entry=ledger.get(sha(signature.encode()))
    if entry is None:raise ValueError('Unknown native thinking signature; cannot reconstruct history')
    block=deepcopy(entry['block'])
    assert block.get('signature',block.get('data'))==signature
    # Round-trip actual provider bytes, never synthesize or rewrite thinking.
    append('assistant',[block])
   elif item.get('summary'):raise ValueError('Summary-only reasoning has no native continuity evidence')
   continue
  if role in ['system','developer']:
   if messages:raise ValueError('Mid-history system content cannot be silently moved')
   system+=content(item['content']);continue
  if kind in ['function_call','custom_tool_call']:
   name=item['name'];ns=item.get('namespace')
   if ns:name=ns+'.'+name
   if name not in byfull:raise ValueError('Historical tool is not in the declared schema: '+name)
   spec=byfull[name]
   args={'_raw':item['input']}if kind=='custom_tool_call'else json.loads(item['arguments'])
   block=dict(type='tool_use',id=item['call_id'],name=spec['wire_name'],input=args)
   retained=blocks.get('call:'+item['call_id'])
   if retained is not None:
    if any(retained.get(k)!=v for k,v in block.items()):raise ValueError('Historical tool call differs from native returned block')
    block=deepcopy(retained)
   append('assistant',[block]);continue
  if kind in ['function_call_output','custom_tool_call_output']:
   append('user',[dict(type='tool_result',tool_use_id=item['call_id'],content=content(item['output']))]);continue
  if role in ['user','assistant']:append(role,content(item['content']));continue
  raise ValueError('Unmapped causal input: '+str(kind))
 if not messages or messages[-1]['role']!='user':raise ValueError('Fable requires a final user/tool-result turn; assistant prefill is unsupported')
 native_tools=[dict(name=x['wire_name'],description=x['description'],input_schema=x['parameters'])for x in active]
 # Stable native cache prefix; no prompt edits and no rewritten conversation.
 if system:system[-1]['cache_control']=dict(type='ephemeral')
 if native_tools:native_tools[-1]['cache_control']=dict(type='ephemeral')
 cap=raw.get('max_output_tokens')or NATIVE_OUTPUT_LIMIT
 assert type(cap)is int and cap>0
 payload=dict(model=MODEL,output_config=dict(effort='xhigh'),thinking=dict(type='adaptive',display='summarized'),max_tokens=cap,messages=messages,stream=False)
 if system:payload['system']=system
 if native_tools:payload.update(tools=native_tools,tool_choice=(dict(type='none')if request_kind(raw)=='compaction'else dict(type='auto',disable_parallel_tool_use=True)))
 for key in ['temperature','top_p','top_k']:
  if key in raw:payload[key]=raw[key]
 cache_message_history(payload)
 return payload,history

def normalize_response(native,specs,ledger,wire_response_sha256,native_request=None):
 if native.get('type')=='error' or native.get('error'):return native,[]
 requested=(native_request or{}).get('model',MODEL)
 if requested not in ROUTE_MODELS or native.get('model')not in(requested,'claude-fable-5-2'):raise ValueError('Provider returned a different or missing model identity')
 rid=native['id'];out=[];summaries=[];byname={x['wire_name']:x for x in specs}
 for index,block in enumerate(native['content']):
  iid=rid+'_'+str(index);kind=block['type']
  if kind in ['thinking','redacted_thinking']:
   signature=block.get('signature')if kind=='thinking'else block.get('data')
   if not isinstance(signature,str)or not signature:raise ValueError('Native thinking block has no continuity signature')
   entry=dict(block=deepcopy(block),native_message_id=rid,native_response_sha256=wire_response_sha256)
   if native_request is not None:
    prefix=deepcopy(native_request['messages'])
    before=[x for x in native['content'][:index]if x['type']not in('thinking','redacted_thinking')]
    if before:prefix.append(dict(role='assistant',content=before))
    entry['prefix_sha256']=prefix_hash(native_request,prefix)
   old=ledger.get(sha(signature.encode()))
   if old and old['block']!=entry['block']:raise ValueError('Provider reused a signature with different thinking bytes')
   ledger[sha(signature.encode())]=entry
   text=block.get('thinking','');summary=[dict(type='summary_text',text=text)]if text else []
   # The opaque bytes here are the provider's actual encrypted full-thinking
   # signature/data, not OpenAI ciphertext or locally generated reasoning.
   out.append(dict(type='reasoning',id=iid,summary=summary,encrypted_content=signature))
   summaries.append(dict(native_message_id=rid,block_index=index,source_block_type=kind,summary=text,signature=signature,native_response_sha256=wire_response_sha256,source='provider_returned_summarized_thinking',thinking_display='summarized'))
  elif kind=='text':out.append(dict(type='message',id=iid,role='assistant',status='completed',content=[dict(type='output_text',text=block['text'],annotations=[])]))
  elif kind=='tool_use':
   spec=byname.get(block['name']);full=spec['full_name']if spec else block['name'];namespace,_,name=full.rpartition('.')
   common=dict(id=iid,call_id=block['id'],name=name or full,status='completed')
   if namespace:common['namespace']=namespace
   args=block['input']
   ledger.setdefault('__output_blocks__',{})['call:'+block['id']]=deepcopy(block)
   if spec and spec['kind']=='custom'and isinstance(args,dict)and set(args)=={'_raw'}and isinstance(args['_raw'],str):out.append(dict(type='custom_tool_call',input=args['_raw'],**common))
   else:out.append(dict(type='function_call',arguments=json.dumps(args,ensure_ascii=False,separators=(',',':')),**common))
  else:raise ValueError('Unmapped native output block: '+str(kind))
 usage=normalized_native_usage(native.get('usage'))
 stop=native.get('stop_reason');status='incomplete'if stop=='max_tokens'else'completed'if stop in ['end_turn','tool_use','stop_sequence']else'failed'
 error=None if status!='failed'else dict(type='content_policy_violation'if stop=='refusal'else'anthropic_stop_reason',message=str(stop))
 metadata=dict(native_stop_reason=stop,provider_input_transformations=deepcopy(native.get('input_transformations',[])),canonical_model=MODEL,native_request_model=requested,native_response_model=native.get('model'))
 return dict(id=rid,object='response',created_at=int(time.time()),model=MODEL,status=status,output=out,usage=usage,error=error,incomplete_details=dict(reason='max_output_tokens')if status=='incomplete'else None,metadata=metadata),summaries

def retain_training_summaries(owner,stem,native_request,native_body,projected,summaries,status):
 """Keep provider summaries as referenced source attachments, never as raw think."""
 if getattr(owner,'storage_mode','raw')!='training':return
 training=Path(owner.case)/'training';assert training.is_dir()
 directory=training/'provider_summaries'
 journal=getattr(getattr(owner,'training_store',None),'journal',None)
 request_id=int(stem);owner_path=Path(owner.case).parent/'episode-owner.json'
 identity=dict(path=str(owner_path),sha256=sha(owner_path.read_bytes()))if owner_path.exists()else None
 value=dict(format='anthropic_provider_summary_source_v1',request_id=request_id,wire_model=native_request['model'],canonical_model=MODEL,effort='xhigh',thinking_display='summarized',HTTP_status=status,episode_owner=identity,native_request_sha256=sha(encoded(native_request)),native_response_sha256=sha(native_body),projected_response_sha256=sha(projected),summaries=summaries,summary_absent_reason=None if summaries else 'provider_no_thinking_block'if 200<=status<300 else 'native_HTTP_error',source='actual_provider_response',signature_preserved_opaque=True)
 try:value['returned_model']=json.loads(native_body).get('model')
 except (ValueError,AttributeError):value['returned_model']=None
 try:value['provider_input_transformations']=json.loads(native_body).get('input_transformations',[])
 except (ValueError,AttributeError):value['provider_input_transformations']=None
 value['native_prefix_policy']=(native_request.get('thinking')or{}).get('block_binding')
 if journal is not None:
  journal.append('provider_summary',value);return
 directory.mkdir(exist_ok=True)
 manifest=directory/'manifest.json';index=json.loads(manifest.read_bytes())if manifest.exists()else dict(format='anthropic_provider_summary_manifest_v1',model=MODEL,thinking_display='summarized',files={})
 path=directory/(stem+'.json');save(path,value);index['files'][path.name]=dict(sha256=sha(path.read_bytes()),request_id=request_id,summary_blocks=len(summaries),native_response_sha256=value['native_response_sha256']);save(manifest,index)


def retryable_error(status,body):
 """Transient HTTP errors may retry; explicit terminal provider blocks may not."""
 if status not in (429,500,502,503,504,529):return False
 try:value=json.loads(body)
 except (ValueError,TypeError):return True
 error=encoded(value.get('error')).decode().lower()if isinstance(value,dict)else''
 return not any(code in error for code in ('content_policy','content_filter','forbidden','insufficient_quota',
                                         'invalid_api_key','authentication_error','billing_hard_limit'))


class NativeResponseUnknown(RuntimeError):
 """The retained provider reply could not be interpreted as a usable response."""


class AnthropicMessagesTransport:
 """Per-capture adapter; no global HTTP client patches or extra retry owner."""
 def __init__(self, *, state_sources=(), prefix_mismatch_behavior='audit_only', request_deadline_seconds=300):
  if prefix_mismatch_behavior not in ('error','audit_only'):
   raise ValueError('Native prefix policy must retain original signed history')
  if type(request_deadline_seconds)not in(int,float)or not 0<request_deadline_seconds<=300:
   raise ValueError('Invalid native request deadline')
  self.state_sources=tuple(state_sources)
  self.prefix_policy=prefix_mismatch_behavior
  self.deadline=request_deadline_seconds
  routes=json.loads(os.environ.get('ROBORSI_FABLE_ROUTE_MODELS',json.dumps(ROUTE_MODELS)))
  if not isinstance(routes,list)or not routes or any(r not in ROUTE_MODELS for r in routes)or len(set(routes))!=len(routes):
   raise ValueError('Fable routes must explicitly name only the authorized aliases')
  self.routes=tuple(routes)
  self.max_attempts=ROUTE_ATTEMPT_LIMIT*len(self.routes)

 def next_route(self,selected,attempts,routes=None):
  routes=routes or self.routes
  return next((r for r in routes[routes.index(selected):]if attempts.get(r,0)<ROUTE_ATTEMPT_LIMIT),selected)

 def attempt_receipt(self,owner,request_id):
  name='%06d.receipt.json'%request_id
  journal=getattr(getattr(owner,'training_store',None),'journal',None)
  if journal is not None:return journal_states(journal).get(name,{})
  path=owner.output_dir/'messages-wire'/name
  return json.loads(path.read_bytes())if path.exists()else{}

 @asynccontextmanager
 async def stream(self,owner,client,method,url,**kw):
  target=httpx.URL(url)
  if method!='POST' or not target.path.endswith('/responses') or target.query:
   raise ValueError('Native Messages transport requires the configured Responses endpoint without query')
  if owner.model!=MODEL:
   raise ValueError('Native Messages transport received another model')
  raw=json.loads(kw['content'])
  await operator_pause_boundary(owner,sha(encoded(raw)))
  directory=owner.output_dir/'messages-wire'
  journal=getattr(getattr(owner,'training_store',None),'journal',None)
  if journal is None:directory.mkdir(exist_ok=True)
  persist=lambda path,value:save(path,value,journal=journal)
  stem='%06d'%owner._sequence
  state=load_state(owner,directory,self.state_sources,journal=journal);ledger=state['thinking_ledger']
  from .case_channel_pin import resolve
  case_pin=resolve(owner);routes=(case_pin['channel'],)if case_pin else self.routes
  self.case_pin_active=case_pin is not None
  if case_pin:
   prior_pin=state.get('case_channel_pin')
   if prior_pin and(prior_pin['assignment_id']!=case_pin['assignment_id']or prior_pin['channel']!=case_pin['channel']):raise ValueError('Same case persistent channel assignment changed')
   state['case_channel_pin']={k:case_pin[k]for k in ['assignment_id','channel','sha256']}
  self.max_attempts=ROUTE_ATTEMPT_LIMIT*len(routes)
  headers=[(k,v)for k,v in kw['headers']if k.lower()not in [b'content-length',b'anthropic-version',b'anthropic-beta']]
  credential=next((v for k,v in headers if k.lower()==b'authorization'),None)
  fingerprint=sha(credential)if credential is not None else None
  if state.get('credential_binding_sha256',fingerprint)!=fingerprint:
   raise ValueError('Native thinking continuity requires the same credential')
  state['credential_binding_sha256']=fingerprint
  headers.extend([(b'anthropic-version',b'2023-06-01'),(b'anthropic-beta',BINDING_BETA.encode())])
  native,specs=request_to_messages(raw,ledger,state['tool_registry'])
  # The frozen Agent/session identity remains canonical. The actual provider
  # request and source receipts explicitly retain the selected routing alias.
  request_sha256=sha(encoded(raw))
  same_request=state.get('route_policy_version')in(4,5)and state.get('route_request_sha256')==request_sha256 and not state.get('route_request_completed')
  runtime=str(Path(owner.case).resolve())
  previous_attempts=state.get('route_attempts',{})
  if same_request and self.state_sources and state.get('route_request_runtime')not in(None,runtime)and sum(previous_attempts.values())>=self.max_attempts:
   # Only a new, explicitly pinned native continuation can open another
   # request-recovery round after the scheduler deferred this exhausted case.
   state['previous_exhausted_route_round']=dict(runtime=state['route_request_runtime'],request_sha256=request_sha256,attempts=deepcopy(previous_attempts))
   same_request=False
  selected=state.get('next_route_model',routes[0])if same_request else routes[0]
  attempts=state.get('route_attempts',{})if same_request else{}
  if any(r not in self.routes or type(n)is not int or not 0<=n<=ROUTE_ATTEMPT_LIMIT for r,n in attempts.items()):
   raise ValueError('Invalid inherited Fable route attempt budget')
  if case_pin:
   if any(r not in routes for r in attempts):state['pre_pin_route_attempts']=deepcopy(attempts)
   attempts={r:n for r,n in attempts.items()if r in routes};selected=routes[0]
  state.update(route_policy_version=5 if case_pin else 4,route_request_sha256=request_sha256,route_request_completed=False,route_request_runtime=runtime,route_attempts=attempts)
  if selected not in routes:raise ValueError('Inherited Fable route is not authorized')
  selected=self.next_route(selected,attempts,routes)
  if attempts.get(selected,0)>=ROUTE_ATTEMPT_LIMIT:raise ValueError('Fable request route attempts exhausted')
  state['next_route_model']=selected
  native['model']=selected
  if self.prefix_policy=='error':native['thinking']['block_binding']=dict(prefix_mismatch_behavior='error')
  findings=audit_prefix(native,ledger)
  persist(directory/(stem+'.prefix-audit.json'),dict(local_findings=findings,policy=self.prefix_policy,
       provider_binding_enforcement_verified=False,raw_history_modified_by_transport=False,documentation=PREFIX_DOCS))
  if findings and self.prefix_policy=='error':
   raise ValueError('Native thinking prefix changed before provider call')
  _,state['tool_registry']=resolve_registry(raw,state['tool_registry'])
  state['prefix_policy']=self.prefix_policy;persist(directory/'transport-state.json',state)
  wire=encoded(native);persist(directory/(stem+'.request.json'),native);persist(directory/(stem+'.tool-map.json'),specs)
  started=time.time();native_target=target.copy_with(path=target.path[:-len('/responses')]+'/messages')
  attempts[selected]=attempts.get(selected,0)+1
  persist(directory/'transport-state.json',state)
  intent=dict(started_at=started,wire_model=selected,canonical_model=MODEL,authorized_route_models=list(routes),logical_request_sha256=request_sha256,request_kind=request_kind(raw),native_request_sha256=sha(wire),
              history_cache=history_cache_metadata(native),
              hard_deadline_seconds=self.deadline,credential_headers_persisted=False,
              route_attempt=attempts[selected],route_attempt_limit=ROUTE_ATTEMPT_LIMIT,route_attempts_total=sum(attempts.values()),
              route_attempts=deepcopy(attempts),route_budget_exhausted=sum(attempts.values())>=self.max_attempts)
  if case_pin:intent['case_channel_pin']={k:case_pin.get(k)for k in ['assignment_id','channel','path','sha256','pinned_at','legacy_mixed']}
  persist(directory/(stem+'.intent.json'),intent)
  try:
   response=await asyncio.wait_for(client.post(native_target,content=wire,headers=headers),timeout=self.deadline)
  except BaseException as error:
   retry_eligible=isinstance(error,Exception)
   if retry_eligible:
    state['next_route_model']=self.next_route(selected,attempts,routes)
    persist(directory/'transport-state.json',state)
   persist(directory/(stem+'.receipt.json'),dict(intent,completed_at=time.time(),completed=False,
         error_type=type(error).__name__,usage_unknown=True,outcome_unknown=True,retry_eligible=retry_eligible,
         next_route_model=state.get('next_route_model',selected),upstream_generation_cancel_confirmed=False))
   raise
  body=response.content;summaries=[]
  provider_ids={name:response.headers[name]for name in ['x-request-id','request-id','x-oneapi-request-id','x-new-api-request-id']if response.headers.get(name)}
  if provider_ids:intent['provider_request_ids']=provider_ids;intent['provider_request_id']=provider_ids.get('x-oneapi-request-id')or provider_ids.get('x-new-api-request-id')or provider_ids.get('x-request-id')or provider_ids.get('request-id')
  # This transport never retries. Only the existing bounded capture may issue
  # another attempt, including user-authorized retries of uncertain outcomes.
  # Earlier attempts and any unknown generation/billing stay in the ledger.
  retry_eligible=retryable_error(response.status_code,body)
  try:response_value=response.json()
  except ValueError:response_value=None
  intent['native_reported_usage']=normalized_native_usage(response_value.get('usage'))if isinstance(response_value,dict)else None
  outcome_unknown=retry_eligible and(not isinstance(response_value,dict)or not response_value.get('error')
      or any(response_value.get(k)for k in ('id','content','output','choices','stop_reason','usage')))
  state['next_route_model']=selected
  if retry_eligible and len(routes)>1:
   state['next_route_model']=self.next_route(selected,attempts,routes)
  persist(directory/'transport-state.json',state)
  if journal is None or not 200<=response.status_code<300:
   directory.mkdir(exist_ok=True);(directory/(stem+'.response.bin')).write_bytes(body)
  try:
   if 200<=response.status_code<300:
    try:
     value,summaries=normalize_response(response_value,specs,ledger,sha(body),native_request=native)
     if isinstance(response_value,dict)and response_value.get('error')and retryable_error(500,body):
      raise NativeResponseUnknown('Provider error inside HTTP success')
    except (ValueError,KeyError,TypeError,AttributeError,AssertionError)as error:
     raise NativeResponseUnknown(type(error).__name__)from error
    if value.get('status')=='completed':state.update(next_route_model=routes[0],route_request_completed=True)
    state['thinking_ledger']=ledger;persist(directory/'transport-state.json',state)
    persist(directory/'thinking-ledger.json',ledger)
    if raw.get('stream'):
     from .cci_protocol import sse
     projected=sse(value);ctype='text/event-stream'
    else:projected=encoded(value);ctype='application/json'
   else:projected=body;ctype=response.headers.get('content-type','application/json')
   retain_training_summaries(owner,stem,native,body,projected,summaries,response.status_code)
   persist(directory/(stem+'.receipt.json'),dict(intent,completed_at=time.time(),HTTP_status=response.status_code,
        native_response_sha256=sha(body),projected_response_sha256=sha(projected),summary_blocks=len(summaries),
        thinking_display='summarized',projection_declared=True,prefix_policy=self.prefix_policy,
        retry_eligible=retry_eligible,outcome_unknown=outcome_unknown,next_route_model=state['next_route_model']))
   status=response.status_code
  except Exception as error:
   # Keep every unusable paid response before an authorized retry. Local
   # durability failures still stop dispatch so the source can be repaired.
   directory.mkdir(exist_ok=True)
   failure=directory/(stem+'.retention-failed.response.bin')
   with failure.open('wb')as stream:
    stream.write(body);stream.flush();os.fsync(stream.fileno())
   unknown=isinstance(error,NativeResponseUnknown)
   if unknown:
    state['next_route_model']=self.next_route(selected,attempts,routes)
    persist(directory/'transport-state.json',state)
    persist(directory/(stem+'.receipt.json'),dict(intent,completed_at=time.time(),HTTP_status=response.status_code,
         native_response_sha256=sha(body),projection_error=type(error).__name__,outcome_unknown=True,
         usage_unknown=True,retry_eligible=True,next_route_model=state['next_route_model']))
   projected=encoded(dict(error=dict(type='native_response_unknown'if unknown else'source_retention_error',message=type(error).__name__)))
   status=409;ctype='application/json'
  reply=httpx.Response(status,headers={'content-type':ctype},stream=httpx.ByteStream(projected),request=httpx.Request(method,url))
  try:yield reply
  finally:await reply.aclose()
