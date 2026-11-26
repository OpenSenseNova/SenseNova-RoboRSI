from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_EXCLUDED = {
    "auth.json",
    "api_keys.yaml",
    "api_keys.yml",
    "credentials.env",
    ".lifecycle.lock",
}


def archive_episode(source: Path, *, task: str, episode_id: str) -> str | None:
    root_value = os.environ.get("ROBORSI_TRACE_ROOT", "").strip()
    if not root_value or not source.is_dir():
        return None
    training = source/'agent/training'
    compact = (training/'manifest.json').is_file()
    if compact and json.loads((training/'manifest.json').read_text())['state'] != 'closed':
        # An interrupted/current writer is retained at source, not published as
        # a completed export. Legacy/raw episodes retain the original layout.
        return None
    root = Path(root_value).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_id = os.environ.get("ROBODOJO_RUN_ID", "unassigned")
    target = root / run_id / task / episode_id
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".trace-", dir=str(target.parent)))
    try:
        if compact:
            shutil.copytree(training, staging/'runtime/agent/training')
            for name in ('episode-owner.json','archive.pending.json','storage-receipt.json','resume-lineage.json'):
                path = source/name
                if path.is_file() and not path.is_symlink():
                    shutil.copyfile(path, staging/'runtime'/name)
            # These are explanations/accounting, not duplicated request bodies.
            for name in ('usage.jsonl','usage.latest.json','prompt-transforms.jsonl','dispatch_hold.json'):
                path = source/'agent'/name
                if path.is_file() and not path.is_symlink():
                    shutil.copyfile(path, staging/'runtime/agent'/name)
        else:
            shutil.copytree(
                source,
                staging / "runtime",
                ignore=shutil.ignore_patterns(*_EXCLUDED, "*.sock", "__pycache__", "*.pyc"),
                symlinks=True,
            )
        files = []
        for path in sorted((staging / "runtime").rglob("*")):
            if path.is_symlink():
                files.append(
                    {
                        "path": path.relative_to(staging).as_posix(),
                        "symlink": os.readlink(path),
                    }
                )
            elif path.is_file():
                files.append(
                    {
                        "path": path.relative_to(staging).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
        manifest = {
            "format": "xpolicylab_episode_archive_v1",
            "task": task,
            "episode_id": episode_id,
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": str(source),
            "storage_mode": "training" if compact else "raw",
            "files": files,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if target.exists():
            shutil.rmtree(target)
        os.replace(staging, target)
        return str(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
