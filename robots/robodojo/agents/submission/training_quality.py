"""Episode-wide tool pairing and executable supervision masks; source stays intact."""
from __future__ import annotations
from copy import deepcopy
import hashlib,json,re,tarfile
from pathlib import Path

VERSION=1

def sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def texts(value):
    if isinstance(value,str):yield value
    elif isinstance(value,list):
        for part in value:yield from texts(part)
    elif isinstance(value,dict):
        if isinstance(value.get('text'),str):yield value['text']
        elif value.get('type')not in ('image','image_paths','input_image'):
            for key in ('content','output','error','message'):
                if key in value:yield from texts(value[key])

def predicted_calls(row):
    ids={v for target in row.get('targets',[])for v in target.get('output_item_ids',[])}
    selected=set()
    for message in row.get('messages',[]):
        calls=message.get('tool_calls',[])
        for order in message.get('item_order',[]):
            index=order.get('tool_call_index')
            if order.get('id')in ids and type(index)is int and 0<=index<len(calls):selected.add(calls[index]['id'])
    return selected

def classify_result(content,call):
    value='\n'.join(texts(content))
    name=(call.get('function')or{}).get('name','');arguments=json.dumps((call.get('function')or{}).get('arguments',''))
    # Judge actual execution envelopes, not error strings printed by cat/sed.
    executing_python=bool(re.search(r'\bpython(?:3(?:\.\d+)?)?\s+(?:-[uB]\s+)*(?:-[cm]\s+|-(?=\s|$)|[\w./]+\.py\b)',arguments))
    robot_call=name.startswith(('mcp__robot.','mcp__robot__'))or bool(re.search(r'tools\.(?:mcp__robot|robot)[._]+\w+\s*\(',arguments))
    traceback=bool(re.search(r'(?m)^Traceback \(most recent call last\):\s*\n\s+File "[^"\n]+", line \d+',value)and
                   re.search(r'(?m)^[A-Za-z_][\w.]*(?:Error|Exception):',value))
    nonzero_exit=bool(re.search(r'(?im)^(?:Process exited with code|Exit code:)\s*[1-9][0-9]*\b',value))
    outer_failure=value.lstrip().startswith(('Script failed','Script error:'))
    # Python parse errors have a File/line diagnostic but no Traceback header.
    syntax_error=bool(re.search(r'(?m)^\s*File "[^"\n]+", line \d+',value)and
                      re.search(r'(?m)^(?:SyntaxError|IndentationError|TabError):',value))
    script=outer_failure or bool(executing_python and(traceback or syntax_error or nonzero_exit))
    validation=bool(robot_call or executing_python)and bool(re.search(r'(?m)^\s*(?:Input validation error:|ToolInputError:)',value))
    geometry=not script and not validation and bool(re.search(
        r'geometr(?:y|ic).{0,40}reject|reprojection.{0,40}(?:reject|exceed)|triangulation.{0,40}(?:reject|fail)|stale.observation',value,re.I))
    framework=not script and not validation and bool(re.search(r'(^|\n)(?:Internal server error|MCP transport error|Bridge connection failed)',value))
    motion_call=name.endswith('.act')or bool(re.search(r'(?:mcp__robot|robot)[._]+act\b',arguments))
    reached=bool(re.search(r'"status"\s*:\s*"(?:reached|completed)"',value))
    kind='model_script_failure'if script else'parameter_validation_rejection'if validation else'geometry_rejection'if geometry else'framework_error'if framework else'ordinary_result'
    if script and motion_call:kind='script_failure_after_possible_motion'
    return {'classification':kind,'assistant_call_loss_weight':0 if script or validation or framework else 1,
            'execution_failure_evidence':{'outer_script_failed':outer_failure,'executed_python_traceback':executing_python and traceback,'executed_python_syntax_error':executing_python and syntax_error,'nonzero_command_exit':nonzero_exit},
            'motion_execution':'observed_in_result'if motion_call and reached else'possible_observe_ACK_before_retry'if motion_call else'not_determined',
            'motion_not_executed_claimed':False,'result_sha256':sha(content),'result_excerpt':value[:400]}

def audit_rows(rows,*,terminal=None):
    """Pair across actor/compact rows, never require a pair inside each row."""
    calls={};results={};locations={};targets={}
    for index,row in enumerate(rows):
        predicted=predicted_calls(row)
        request_for_item={item:target['request_id']for target in row.get('targets',[])if 'request_id'in target for item in target.get('output_item_ids',[])}
        for message in row.get('messages',[]):
            for call_index,call in enumerate(message.get('tool_calls',[])):
                cid=call['id'];calls.setdefault(cid,call)
                if cid not in locations or cid in predicted:locations.setdefault(cid,[]).append({'row_index':index,'member':row.get('_bundle_member'),'row':row.get('_bundle_row'),'segment_id':row.get('segment_id'),'invocation_id':row.get('invocation_id')})
                targets.setdefault(cid,set()).update(request_for_item[order['id']]for order in message.get('item_order',[])if order.get('tool_call_index')==call_index and order.get('id')in request_for_item)
            if message.get('role')=='tool'and message.get('tool_call_id'):
                results.setdefault(message['tool_call_id'],[]).append(message.get('content'))
    # A yielded code cell reports its final stdout through functions.wait.
    # Attribute a proven later failure to the originating exec prediction,
    # while retaining both real replies and leaving a valid poll unmasked.
    cell_owners={};async_results={}
    for cid,values in results.items():
        if (calls.get(cid,{}).get('function')or{}).get('name')!='functions.exec':continue
        for value in values:
            for match in re.finditer(r'(?m)^Script running with cell ID ([A-Za-z0-9_-]+)\s*$', '\n'.join(texts(value))):
                cell_owners.setdefault(match[1],set()).add(cid)
    for cid,call in calls.items():
        function=call.get('function')or{}
        if function.get('name')!='functions.wait':continue
        args=function.get('arguments',{})
        if isinstance(args,str):
            try:args=json.loads(args)
            except ValueError:continue
        if not isinstance(args,dict):continue
        owners=cell_owners.get(str(args.get('cell_id')),set())
        # Cell IDs can repeat across invocations/resumes. Ambiguous evidence
        # cannot be guessed into a supervision mask.
        if len(owners)!=1:continue
        origin=next(iter(owners))
        for value in results.get(cid,[]):
            item=classify_result(value,calls[origin])
            if item['assistant_call_loss_weight']==0:
                item['async_result_via_wait_call_id']=cid
                async_results.setdefault(origin,[]).append(item)
    annotations={};unpaired=[]
    for cid,call in calls.items():
        if cid in results:
            candidates=[classify_result(value,call)for value in results[cid]]+async_results.get(cid,[])
            item=min(candidates,key=lambda v:v['assistant_call_loss_weight'])
            item.update(state='result_observed',all_result_sha256=sorted({v['result_sha256']for v in candidates}))
        else:
            item={'state':'unobserved_terminal_tail'if terminal and terminal.get('native_terminal')else'unobserved_at_capture_close',
                  'classification':'uncompleted_call','assistant_call_loss_weight':0,
                  'cancelled':False,'cancellation_receipt':None,'tool_result_fabricated':False,'motion_not_executed_claimed':False}
            unpaired.append(cid)
        item.update(call_id=cid,tool_name=(call.get('function')or{}).get('name'),locations=locations[cid],request_ids=sorted(targets[cid],key=str))
        if item['assistant_call_loss_weight']==0 or item['classification']=='geometry_rejection':annotations[cid]=item
    masked={cid for cid,a in annotations.items()if a['assistant_call_loss_weight']==0}
    row_masks=[]
    for index,row in enumerate(rows):
        predicted=predicted_calls(row)
        row_masks.append({'row_index':index,'member':row.get('_bundle_member'),'row':row.get('_bundle_row'),'segment_id':row.get('segment_id'),
                          'predicted_call_ids':sorted(predicted),'masked_call_ids':sorted(predicted&masked),
                          'source_messages_preserved':True,'image_references_preserved':True})
    return {'schema_version':VERSION,'pairing_scope':'whole_episode_actor_and_compact','calls':len(calls),'observed_result_call_ids':len(set(results)&set(calls)),
            'unpaired_call_ids':sorted(unpaired),'orphan_result_ids':sorted(set(results)-set(calls)),
            'call_annotations':annotations,'row_masks':row_masks,'terminal_evidence':terminal or{},
            'script_failure_calls':sum(v['classification']in('model_script_failure','script_failure_after_possible_motion')for v in annotations.values()),
            'parameter_rejection_calls':sum(v['classification']=='parameter_validation_rejection'for v in annotations.values()),
            'mask_policy':'zero failed/uncompleted assistant call targets; retain all context, real tool feedback and corrections',
            'cancelled_inferred_from_missing_result':False}

def rows_from_directory(root):
    rows=[]
    for kind in ('actor','compact'):
        path=Path(root)/(kind+'.jsonl')
        if path.exists():
            for number,line in enumerate(path.open(),1):
                row=json.loads(line);rows.append({**row,'_bundle_member':kind+'.jsonl','_bundle_row':number})
    return rows

def audit_bundle(path):
    path=Path(path);rows=[]
    with tarfile.open(path,'r:gz')as archive:
        manifest=json.load(archive.extractfile('dataset-manifest.json'))
        for member in archive.getmembers():
            if member.isfile()and member.name.endswith(('/training/actor.jsonl','/training/compact.jsonl')):
                for number,line in enumerate(archive.extractfile(member),1):
                    row=json.loads(line);rows.append({**row,'_bundle_member':member.name,'_bundle_row':number})
    result=audit_rows(rows,terminal={'native_terminal':bool(manifest.get('grade')),'grade':manifest.get('grade')})
    digest=hashlib.sha256()
    with path.open('rb')as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return {**result,'format':'roborsi_immutable_bundle_training_quality_v1','original_package_path':str(path),
            'original_package_sha256':digest.hexdigest(),'original_package_repacked':False,'native_run_id':manifest['native_run_id']}

def apply_label_mask(labels,supervision_spans,row_mask):
    """Actual trainer boundary: the provided tokenizer spans map to source calls."""
    output=list(labels);wanted=set(row_mask['masked_call_ids']);covered=set()
    for span in supervision_spans:
        cid=span['call_id'];start,end=span['start'],span['end']
        if type(start)is not int or type(end)is not int or not 0<=start<end<=len(output):raise ValueError('Invalid tokenizer supervision span')
        if cid in wanted:
            output[start:end]=[-100]*(end-start);covered.add(cid)
    if covered!=wanted:raise ValueError('Masked predicted calls have no tokenizer span: '+str(sorted(wanted-covered)))
    return output

def consume_tokenized(sample,quality):
    """Apply the immutable sidecar before a tokenized sample enters a trainer."""
    if sample['original_package_sha256']!=quality['original_package_sha256']:raise ValueError('Training sidecar belongs to another package')
    mask=next(v for v in quality['row_masks']if v['member']==sample['bundle_member']and v['row']==sample['bundle_row'])
    value=deepcopy(sample);value['labels']=apply_label_mask(value['labels'],value['supervision_spans'],mask)
    value['loss_mask']=[int(label!=-100)for label in value['labels']]
    value['training_quality_schema_version']=VERSION;value['training_quality_sha256']=sha(quality)
    return value
