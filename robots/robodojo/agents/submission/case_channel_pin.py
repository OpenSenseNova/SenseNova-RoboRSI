"""Persistent per-case routing; read at every native provider request boundary."""
from pathlib import Path
import hashlib,json,os

ROUTES=('claude-fable-5-2/anthropic_S/sfp','claude-fable-5-2/anthropic_L/sfo')


def resolve(owner):
    key=os.environ.get('ROBODOJO_RUN_ID');case=Path(owner.case).resolve();root=None
    if key:
        root=next((p for p in case.parents if p.name==key and p.parent.name=='cases'),None)
    if root is None:
        if os.environ.get('ROBORSI_FABLE_CASE_PIN_REQUIRED')=='1':raise ValueError('Required case pin scope missing')
        return None
    path=root/'channel-pin.json'
    if not path.exists():
        if os.environ.get('ROBORSI_FABLE_CASE_PIN_REQUIRED')=='1':raise ValueError('Required persistent case channel assignment missing')
        return None
    stat=path.stat()
    if path.is_symlink()or stat.st_uid!=os.getuid()or stat.st_mode&0o077:raise ValueError('Case pin is not private and owned')
    raw=path.read_bytes();pin=json.loads(raw)
    if pin.get('mode')!='case_pinned'or pin.get('native_run_id')!=key or pin.get('channel')not in ROUTES:raise ValueError('Case channel is held or assignment differs')
    return {**pin,'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}
