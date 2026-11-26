"""Lossless public tool/message transport adapter for the frozen CCI models.

Codex namespace/custom tools become the training schema functions.exec({_raw}).
No robot action is executed here. Both upstream wire and CLI response are saved.
"""
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import hashlib
import math
import threading
import time
import urllib.request
import urllib.error

CCI_MAX_OUTPUT_TOKENS = 65536


def declarations(items, prefix=''):
    result=[]
    for item in items:
        kind=item['type']
        if kind=='namespace':
            result.extend(declarations(item['tools'],prefix+item['name']+'.'))
        elif kind in ('function','custom'):
            result.append(dict(type='function',name=prefix+item['name'],
                description=item.get('description',''),parameters=item.get('parameters') if kind=='function' else
                dict(type='object',properties={'_raw':{'type':'string'}},required=['_raw'],additionalProperties=False)))
        else: raise ValueError('Unsupported tool declaration: '+kind)
    return result


def request_to_cci(request, *, enable_thinking=True):
    if type(enable_thinking) is not bool:
        raise ValueError('CCI enable_thinking must be a boolean')
    raw=deepcopy(request)
    tools=declarations(raw.get('tools',[]))
    source=raw.get('input',[])
    if isinstance(source,str): source=[dict(type='message',role='user',content=[dict(type='input_text',text=source)])]
    items=[]
    system=[raw['instructions']] if raw.get('instructions') else []
    for item in source:
        kind=item.get('type')
        if kind=='additional_tools':
            tools.extend(declarations(item['tools']));continue
        if kind=='reasoning':
            # No decryption/reconstruction; private reasoning stays in raw logs.
            continue
        if item.get('role') in ('system','developer'):
            if items: raise ValueError('Unexpected mid-history system instruction')
            content=item['content']
            system.append(content if isinstance(content,str) else '\n'.join(p['text'] for p in content));continue
        if kind in ('custom_tool_call','function_call'):
            name=item['name']
            if item.get('namespace'):name=item['namespace']+'.'+name
            elif name in ('exec','wait'):name='functions.'+name
            args=json.dumps({'_raw':item['input']},ensure_ascii=False) if kind=='custom_tool_call' else item['arguments']
            item=dict(type='function_call',call_id=item['call_id'],name=name,arguments=args)
        elif kind in ('custom_tool_call_output','function_call_output'):
            item=dict(type='function_call_output',call_id=item['call_id'],output=item['output'])
        items.append(item)
    result=dict(model=raw['model'],instructions='\n\n'.join(system),input=items,
        tools=tools,tool_choice='auto',parallel_tool_calls=False,stream=False,store=False,
        reasoning=raw.get('reasoning',{'effort':'xhigh'}),
        chat_template_kwargs={'enable_thinking':enable_thinking,'reasoning_effort':(raw.get('reasoning') or {}).get('effort','xhigh')},
        temperature=0)
    requested_cap=raw.get('max_output_tokens')
    if requested_cap is not None and (type(requested_cap) is not int or requested_cap<=0):
        raise ValueError('Invalid CCI max_output_tokens')
    result['max_output_tokens']=min(requested_cap,CCI_MAX_OUTPUT_TOKENS) if requested_cap is not None else CCI_MAX_OUTPUT_TOKENS
    def image_defaults(value):
        if isinstance(value,dict):
            if value.get('type')=='input_image':value.setdefault('detail','auto')
            for child in value.values():image_defaults(child)
        elif isinstance(value,list):
            for child in value:image_defaults(child)
    image_defaults(result['input'])
    return result


def response_to_cli(response, *, tools=(), alias_events=None):
    # Accept legacy spellings only for robot tools registered in this request.
    # This is response-side routing; declarations and argument bytes stay intact.
    robot_aliases={}
    for spec in declarations(tools):
        ns,_,name=spec['name'].rpartition('.')
        if ns=='mcp__robot':
            alias='mcp__robot__'+name
            robot_aliases[alias]=spec['name']
            robot_aliases['functions.'+alias]=spec['name']
    result=deepcopy(response)
    out=[]
    for item in result.get('output',[]):
        if item['type']=='reasoning':
            # Preserve opaque provider fields, never manufacture a think string.
            item={k:v for k,v in item.items() if k in ('type','id','summary','encrypted_content')}
            item.setdefault('summary',[])
        elif item['type']=='function_call':
            full=item['name']
            if item.get('namespace'):full=item['namespace']+'.'+full
            elif full in ('exec','wait'):full='functions.'+full
            if full in robot_aliases:
                full=robot_aliases[full]
                if alias_events is not None:
                    alias_events.append(dict(output_index=len(out),id=item.get('id'),call_id=item.get('call_id'),
                        original={k:item[k] for k in ('name','namespace') if k in item},
                        resolved=dict(name=full.rpartition('.')[2],namespace='mcp__robot')))
            ns,_,name=full.rpartition('.')
            if full=='functions.exec':
                try:args=json.loads(item['arguments'])
                except (ValueError,TypeError):args=None
                if isinstance(args,dict) and set(args)=={'_raw'} and isinstance(args['_raw'],str):
                    item=dict(type='custom_tool_call',id=item['id'],call_id=item['call_id'],
                              name=name,namespace=ns,input=args['_raw'],status='completed')
                else:
                    # Keep the wrong function payload intact. The native CLI
                    # returns its tool-validation error so the model can fix
                    # its own call; never execute a guessed shell command.
                    item.update(name=name,namespace=ns)
            elif full=='functions.wait':
                item.update(name=name,namespace=ns)
            elif ns=='mcp__robot':
                # Project every robot tool into the CLI namespace, including
                # localization, spatial memory and target calculators. The CLI
                # still validates the exact name and unmodified arguments.
                item.update(name=name,namespace=ns)
            # An unknown call likewise remains an unknown call for the CLI;
            # model mistakes are not synthetic HTTP/transport failures.
        out.append(item)
    result['output']=out
    return result


def sse(response):
    events=[]
    def emit(kind,**kw):events.append(dict(type=kind,sequence_number=len(events),**kw))
    initial={**response,'status':'in_progress','output':[],'usage':None}
    emit('response.created',response=initial)
    emit('response.in_progress',response=initial)
    for i,item in enumerate(response.get('output',[])):
        kind=item['type'];start=deepcopy(item)
        if kind in ('function_call','custom_tool_call'):start['arguments' if kind=='function_call' else 'input']=''
        elif kind=='message':start['content']=[]
        emit('response.output_item.added',output_index=i,item=start)
        if kind in ('function_call','custom_tool_call'):
            field='arguments' if kind=='function_call' else 'input'
            emit('response.'+kind+('_arguments' if kind=='function_call' else '_input')+'.delta',
                 output_index=i,item_id=item['id'],delta=item[field])
            emit('response.'+kind+('_arguments' if kind=='function_call' else '_input')+'.done',
                 output_index=i,item_id=item['id'],**{field:item[field]})
        elif kind=='message':
            for ci,part in enumerate(item.get('content',[])):
                emit('response.content_part.added',output_index=i,item_id=item['id'],content_index=ci,
                     part={**part,'text':''} if part.get('type')=='output_text' else part)
                if part.get('type')=='output_text':
                    emit('response.output_text.delta',output_index=i,item_id=item['id'],content_index=ci,delta=part['text'])
                    emit('response.output_text.done',output_index=i,item_id=item['id'],content_index=ci,text=part['text'])
                emit('response.content_part.done',output_index=i,item_id=item['id'],content_index=ci,part=part)
        emit('response.output_item.done',output_index=i,item=item)
    emit('response.'+response.get('status','completed'),response=response)
    return ''.join('event: '+e['type']+'\ndata: '+json.dumps(e,ensure_ascii=False)+'\n\n' for e in events).encode()


class CciAdapter:
    def __init__(self,endpoint,model,directory,request_timeout=300,sampling=None,enable_thinking=True,save_raw=True,reasoning_directory=None):
        if type(enable_thinking) is not bool:
            raise ValueError('CCI enable_thinking must be a boolean')
        self.enable_thinking=enable_thinking
        self.reasoning_directory=Path(reasoning_directory) if reasoning_directory is not None else None
        if type(save_raw) is not bool:raise ValueError('save_raw must be boolean')
        self.save_raw=save_raw
        self.endpoint=endpoint.rstrip('/')+'/responses';self.model=model
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.counter=0;self.lock=threading.Lock();self.request_timeout=request_timeout
        self.sampling={'temperature':0}
        if sampling is not None:
            if not isinstance(sampling,dict) or set(sampling)-{'temperature','top_k','top_p'}:
                raise ValueError('CCI sampling supports temperature, top_k and top_p')
            self.sampling.update(sampling)
        for name in ['temperature','top_p']:
            if name not in self.sampling:continue
            value=self.sampling[name]
            if type(value) not in [float,int] or not math.isfinite(value) or value<0 or (name=='top_p' and not 0<value<=1):
                raise ValueError('Invalid CCI '+name)
        if 'top_k' in self.sampling and (type(self.sampling['top_k']) is not int or self.sampling['top_k']<=0):
            raise ValueError('Invalid CCI top_k')
        owner=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                with owner.lock:
                    owner.counter+=1;n=owner.counter
                prefix=owner.directory/f'{n:06d}'
                request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                upstream_started=False;upstream_complete=False;upstream_body=None
                if owner.save_raw:prefix.with_suffix('.cli-request.json').write_text(json.dumps(request,ensure_ascii=False))
                try:
                    assert self.path=='/v1/responses' and request['model']==owner.model
                    payload=request_to_cci(request,enable_thinking=owner.enable_thinking)
                    payload.update(owner.sampling)
                    if owner.save_raw:prefix.with_suffix('.request.json').write_text(json.dumps(payload,ensure_ascii=False))
                    req=urllib.request.Request(owner.endpoint,data=json.dumps(payload,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
                    upstream_started=True
                    try:
                        with urllib.request.urlopen(req,timeout=owner.request_timeout) as r:status=r.status;body=r.read()
                    except urllib.error.HTTPError as error:status=error.code;body=error.read()
                    upstream_complete=True;upstream_body=body
                    if owner.save_raw:prefix.with_suffix('.response.json').write_bytes(body)
                    parsed=json.loads(body) if 200<=status<300 else {}
                    receipt=dict(at=time.time(),status=status,endpoint=owner.endpoint,model=owner.model,usage=parsed.get('usage'),response_id=parsed.get('id'))
                    prefix.with_suffix('.receipt.json').write_text(json.dumps(receipt))
                    content_type='application/json'
                    if 200<=status<300:
                        upstream_body=body
                        aliases=[]
                        response=response_to_cli(parsed,tools=payload['tools'],alias_events=aliases)
                        if aliases:
                            receipt['tool_name_aliases']=aliases
                            prefix.with_suffix('.receipt.json').write_text(json.dumps(receipt))
                        body=sse(response) if request.get('stream') else json.dumps(response,ensure_ascii=False).encode()
                        if owner.reasoning_directory is not None:
                            from .provider_reasoning import retain
                            try:
                                retain(owner.reasoning_directory,request_id=n,request=payload,response=parsed,
                                       raw_body=upstream_body,projected_body=body)
                            except Exception:
                                # The paid response remains recoverable. Do not turn a
                                # local persistence failure into an automatic API retry.
                                prefix.with_suffix('.retention-failed.response.json').write_bytes(upstream_body)
                                status=409
                                body=json.dumps({'error':{'type':'source_retention_error',
                                    'message':'Provider reasoning persistence failed; original response retained'}}).encode()
                        if request.get('stream') and 200<=status<300:content_type='text/event-stream'
                except Exception as error:
                    if upstream_complete and upstream_body is not None:
                        # Preserve malformed/projected replies even in training
                        # mode, before the output guard decides to regenerate.
                        prefix.with_suffix('.invalid-upstream.response.json').write_bytes(upstream_body)
                        receipt_path=prefix.with_suffix('.receipt.json')
                        if not receipt_path.exists():
                            receipt_path.write_text(json.dumps(dict(at=time.time(),status=status,
                                endpoint=owner.endpoint,model=owner.model,usage=None,response_id=None,
                                response_sha256=hashlib.sha256(upstream_body).hexdigest(),outcome_unknown=False)))
                    status=502;content_type='application/json'
                    timed_out=isinstance(error,TimeoutError) or isinstance(getattr(error,'reason',None),TimeoutError)
                    body=json.dumps({'error':{'type':'cci_request_timeout' if timed_out else 'cci_adapter_error',
                                             'message':str(error),
                                              'outcome_unknown':upstream_started and not upstream_complete,
                                             **({'timeout_seconds':owner.request_timeout} if timed_out else {})}}).encode()
                    prefix.with_suffix('.adapter_error.json').write_bytes(body)
                self.send_response(status);self.send_header('Content-Type',content_type)
                self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.base_url=f'http://127.0.0.1:{self.server.server_port}/v1'
    def start(self):threading.Thread(target=self.server.serve_forever,daemon=True).start()
    def stop(self):self.server.shutdown();self.server.server_close()
