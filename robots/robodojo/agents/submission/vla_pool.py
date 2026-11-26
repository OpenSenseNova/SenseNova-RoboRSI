"""Owned loopback Pi0.5 pool: short HTTP requests, authenticated model identity."""
from __future__ import annotations
import contextlib,hashlib,http.server,inspect,json,os,threading,time,uuid
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import ProxyHandler,Request,build_opener

LABEL_PORT=19829
MAX_BYTES=64*1024*1024

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
    with temporary.open('x')as stream:json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)

def codec(policy_dir):
    from .pi05_source import load_openpi_client
    return load_openpi_client(Path(policy_dir))

class PoolClient:
    def __init__(self,policy_dir):
        self.policy_dir=Path(policy_dir);self.endpoint=os.environ['ROBORSI_VLA_POOL_URL'].rstrip('/')
        parsed=urlsplit(self.endpoint)
        if parsed.scheme!='http'or parsed.hostname not in ('127.0.0.1','localhost')or parsed.username or parsed.password or parsed.query or parsed.fragment:raise ValueError('Owned Pi pool must use an explicit loopback HTTP endpoint')
        self.opener=build_opener(ProxyHandler({}));self.codec=codec(policy_dir)
        self.receipt_path=Path(os.environ['ROBORSI_VLA_POOL_RECEIPT'])
        if not self.receipt_path.is_absolute()or self.receipt_path.is_symlink():raise ValueError('Pi pool receipt must be an absolute regular file')
        self.receipt=json.loads(self.receipt_path.read_text());self.metadata=self._metadata()
    def _metadata(self):
        with self.opener.open(Request(self.endpoint+'/metadata',headers={'Connection':'close'}),timeout=15)as response:
            if response.status!=200:raise ValueError('Pi pool metadata unavailable')
            metadata=json.loads(response.read(65536))
        from .vla_identity import pi05_identity
        actual=metadata.get('roborsi_vla_identity',{});expected=pi05_identity(self.policy_dir)
        if any(actual.get(k)!=v for k,v in expected.items())or actual.get('service_id')not in metadata.get('accepted_service_ids',[]):raise ValueError('Pi pool model identity differs')
        if self.receipt.get('schema_version')!=1 or self.receipt.get('endpoint')!=self.endpoint or self.receipt.get('pool_id')!=metadata.get('pool_id')or self.receipt.get('identity')!=actual or self.receipt.get('checkpoint_files_verified')is not True:raise ValueError('Pi pool receipt differs from its live service')
        return metadata
    def request(self,payload):
        if not isinstance(payload,bytes)or not 0<len(payload)<=MAX_BYTES:raise ValueError('Invalid Pi pool payload')
        request_id=uuid.uuid4().hex;root=Path(os.environ['ROBORSI_RUNTIME_STATE_DIR'])/'vla-pool-calls';target=root/(request_id+'.json')
        ledger={'request_id':request_id,'endpoint':self.endpoint,'pool_id':self.metadata['pool_id'],'service_id':self.metadata['roborsi_vla_identity']['service_id'],'payload_sha256':hashlib.sha256(payload).hexdigest(),'attempts':[],'succeeded':False}
        attempts=int(os.environ.get('ROBORSI_VLA_POOL_ATTEMPTS','3'))
        if not 1<=attempts<=30:raise ValueError('Invalid Pi pool retry limit')
        headers={'Connection':'close','Content-Type':'application/msgpack','X-VLA-Service-Id':ledger['service_id'],'X-VLA-Pool-Id':ledger['pool_id'],'X-VLA-Request-Id':request_id}
        for number in range(1,attempts+1):
            event={'number':number,'started_at':time.time(),'outcome':'intent_before_request'};ledger['attempts'].append(event);write(target,ledger)
            try:
                with self.opener.open(Request(self.endpoint+'/request',data=payload,headers=headers),timeout=300)as response:
                    raw=response.read(MAX_BYTES+1)
                    if response.status!=200 or len(raw)>MAX_BYTES:raise ValueError('Invalid Pi pool response')
                value=self.codec.unpackb(raw)
                if not isinstance(value,dict)or value.get('error'):raise ValueError('Pi pool inference failed')
                event.update(ended_at=time.time(),outcome='success',response_sha256=hashlib.sha256(raw).hexdigest());ledger['succeeded']=True;write(target,ledger);return raw
            except Exception as error:
                event.update(ended_at=time.time(),outcome='unknown_or_failed',error_type=type(error).__name__,detail=str(error)[:500]);write(target,ledger)
        write(root/'technical-interruption.json',{'reason':'vla_pool_request_exhausted','request_id':request_id,'ledger':str(target),'unknown_outcomes_retained':True})
        raise RuntimeError('Pi pool request exhausted; preserve original episode for technical recovery')

class PoolConnection:
    def __init__(self,client):self.client=client;self.pending=None;self.closed=False
    def send(self,payload):
        if self.closed or self.pending is not None:raise RuntimeError('Pi pool connection is closed or already has a request')
        self.pending=payload
    def recv(self,timeout=None):
        if self.closed:raise RuntimeError('Pi pool connection is closed')
        if self.pending is None:return self.client.codec.Packer().pack(self.client.metadata)
        payload,self.pending=self.pending,None;return self.client.request(payload)
    def close(self,*args,**kwargs):self.closed=True
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

def install_client(policy_dir):
    import websockets.sync.client
    client=PoolClient(policy_dir);previous=websockets.sync.client.connect
    if getattr(previous,'_roborsi_pool',None)==client.endpoint:return client
    def connect(uri,*args,**kwargs):
        parsed=urlsplit(uri)
        if parsed.scheme=='ws'and parsed.hostname in ('127.0.0.1','localhost')and parsed.port==LABEL_PORT:return PoolConnection(client)
        return previous(uri,*args,**kwargs)
    connect._roborsi_pool=client.endpoint;websockets.sync.client.connect=connect
    return client

class PoolServer:
    """One verified backend member; bounded simultaneous inference requests."""
    def __init__(self,backend_port,backend_receipt,output_dir,policy_dir,max_inflight=8):
        from websockets.sync.client import connect
        from .vla_identity import validate_shared_metadata
        self.backend_port=backend_port;self.output_dir=Path(output_dir);self.policy_dir=Path(policy_dir);self.codec=codec(policy_dir)
        options={'proxy':None}if'proxy'in inspect.signature(connect).parameters else{}
        with connect(f'ws://127.0.0.1:{backend_port}',compression=None,max_size=65536,**options)as connection:metadata=self.codec.unpackb(connection.recv(timeout=15))
        validate_shared_metadata(metadata,backend='pi05',port=backend_port,receipt_path=Path(backend_receipt),policy_dir=self.policy_dir)
        self.metadata={**metadata,'pool_id':uuid.uuid4().hex,'accepted_service_ids':[metadata['roborsi_vla_identity']['service_id']],'member_count':1,'backend':'pi05','max_inflight':max_inflight}
        self.semaphore=threading.BoundedSemaphore(max_inflight);self.log_lock=threading.Lock();owner=self
        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version='HTTP/1.1'
            def log_message(self,*args):pass
            def reply(self,status,raw,kind):
                self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(raw)));self.send_header('Connection','close');self.end_headers();self.wfile.write(raw);self.close_connection=True
            def do_GET(self):
                if self.path!='/metadata':self.reply(404,b'{}','application/json');return
                self.reply(200,json.dumps(owner.metadata).encode(),'application/json')
            def do_POST(self):
                if self.path!='/request':self.reply(404,b'{}','application/json');return
                if self.headers.get('X-VLA-Pool-Id')!=owner.metadata['pool_id']or self.headers.get('X-VLA-Service-Id')not in owner.metadata['accepted_service_ids']:self.reply(409,b'{}','application/json');return
                try:size=int(self.headers.get('Content-Length','0'))
                except ValueError:size=0
                if not 0<size<=MAX_BYTES:self.reply(413,b'{}','application/json');return
                self.connection.settimeout(310);payload=self.rfile.read(size)
                if len(payload)!=size:return
                event={'request_id':self.headers.get('X-VLA-Request-Id'),'started_at':time.time(),'payload_sha256':hashlib.sha256(payload).hexdigest()}
                try:
                    with owner.semaphore:
                        with connect(f'ws://127.0.0.1:{owner.backend_port}',compression=None,max_size=MAX_BYTES,open_timeout=15,close_timeout=5,**options)as connection:
                            connection.recv(timeout=15);connection.send(payload);raw=connection.recv(timeout=300)
                    if not isinstance(raw,bytes):raise ValueError('Pi backend returned an invalid response')
                    event.update(outcome='success',response_sha256=hashlib.sha256(raw).hexdigest());self.reply(200,raw,'application/msgpack')
                except Exception as error:
                    event.update(outcome='unknown_or_failed',error_type=type(error).__name__)
                    with contextlib.suppress(OSError):self.reply(502,b'{"error":"pi backend unavailable"}','application/json')
                finally:
                    event['ended_at']=time.time()
                    with owner.log_lock:
                        with(owner.output_dir/'requests.jsonl').open('a')as stream:stream.write(json.dumps(event)+'\n');stream.flush()
        self.http=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler);self.http.daemon_threads=True
        self.endpoint=f'http://127.0.0.1:{self.http.server_port}'
        self.receipt=self.output_dir/'pool-identity.json'
        write(self.receipt,{'schema_version':1,'endpoint':self.endpoint,'pool_id':self.metadata['pool_id'],'identity':self.metadata['roborsi_vla_identity'],'checkpoint_files_verified':True,'member_count':1,'backend_receipt':str(backend_receipt),'pid':os.getpid()})
    def start(self):self.thread=threading.Thread(target=self.http.serve_forever,daemon=True);self.thread.start()
    def close(self):self.http.shutdown();self.http.server_close();self.thread.join(timeout=10)
