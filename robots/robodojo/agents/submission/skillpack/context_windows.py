"""Append-only provider windows with the pinned Codex default text compact."""
from copy import deepcopy
import json
from pathlib import Path


def image_count(value):
    if isinstance(value,list):return sum(image_count(x) for x in value)
    if isinstance(value,dict):
        if value.get('type') in ['input_image','image_url']:return 1
        return sum(image_count(v) for v in value.values())
    return 0


def token_upper_bound(value):
    # Deliberately conservative; base64 image bytes are not text tokens.
    if isinstance(value,str):return len(value.encode())
    if isinstance(value,list):return sum(token_upper_bound(v) for v in value)
    if isinstance(value,dict):
        if value.get('type') in ['input_image','image_url']:return 4096
        return sum(len(k.encode())+token_upper_bound(v) for k,v in value.items())
    return 8


class ContextWindows:
    def __init__(self,directory):
        root=Path(directory)
        self.prompt=(root/'default_compact_prompt.txt').read_text()
        self.prefix=(root/'default_summary_prefix.txt').read_text()
        self.covered=[];self.replacement=[];self.last_raw=None;self.last_actual=None
        self.calls=0;self.last_input_tokens=0;self.epoch=0

    def reduce(self,items):
        if self.covered:
            if len(items)>=len(self.covered) and items[:len(self.covered)]==self.covered:
                return deepcopy(self.replacement)+deepcopy(items[len(self.covered):])
            # A native CLI compaction or explicit session refresh is a boundary.
            self.covered=[];self.replacement=[];self.last_raw=None;self.last_actual=None
            self.calls=0;self.last_input_tokens=0;self.epoch+=1
        return deepcopy(items)

    async def prepare(self,request,compact):
        raw=deepcopy(request['input']);actual=self.reduce(raw)
        if self.last_raw is not None and not (len(raw)>=len(self.last_raw) and raw[:len(self.last_raw)]==self.last_raw):
            self.last_raw=None;self.last_actual=None;self.calls=0;self.last_input_tokens=0
        added=raw[len(self.last_raw):] if self.last_raw is not None else raw
        estimate=self.last_input_tokens+token_upper_bound(added)
        reasons=[]
        if image_count(actual)>50:reasons.append('image_limit_50')
        if self.calls>=120:reasons.append('model_rounds_120')
        if estimate>=200000:reasons.append('context_200k_conservative_estimate')
        evidence={'epoch':self.epoch,'ordinary_completed_in_epoch':self.calls,'images':image_count(actual),
                  'estimated_input_tokens_upper_bound':estimate,'triggers':reasons}
        if reasons:
            if self.last_raw is None or not self.last_actual:
                raise ValueError('Initial request exceeds context limit; no past window can be compacted')
            summary=await compact(request,self.last_actual,reasons)
            base=[]
            for item in self.last_actual:
                base.append(deepcopy(item))
                if item.get('role')=='user':break
            self.covered=deepcopy(self.last_raw)
            self.replacement=base+[{'type':'message','role':'user','content':[{'type':'input_text','text':self.prefix+summary}]}]
            actual=deepcopy(self.replacement)+deepcopy(raw[len(self.covered):])
            self.calls=0;self.epoch+=1;evidence['epoch_after_compact']=self.epoch
            evidence['images_after_compact']=image_count(actual)
            if image_count(actual)>50:raise ValueError('New suffix alone exceeds image limit after compaction')
        result=deepcopy(request);result['input']=actual
        return result,evidence,raw

    def completed(self,raw,actual,usage,*,native_compact=False):
        if native_compact:
            # The CLI will emit its replacement history in its next request.
            self.covered=[];self.replacement=[];self.last_raw=None;self.last_actual=None
            self.calls=0;self.last_input_tokens=0;self.epoch+=1;return
        self.last_raw=deepcopy(raw);self.last_actual=deepcopy(actual)
        self.calls+=1;self.last_input_tokens=(usage or {}).get('input_tokens',0)
