# Copyright 2026 The Bench2Dex Evaluation Contributors.
# Licensing of this copied reference code is pending; see reference/LICENSE_NOTICE.md.

"""Original ICL inventory function, separated from native RGB encoding."""

import hashlib
import os
import stat
import time
from pathlib import Path


def content_hashes(root, *, strict=True, exclude=(), progress=None):
    """Hash every byte, directory by directory, without a whole-tree stat pass.

    Strict mode rejects links, hardlinks and special files, as knowledge trees
    require. Asset mode retains the legacy regular-file symlink semantics but
    never traverses linked directories. No metadata-only checksum cache.
    """
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Expected a real directory")
    result = {}
    total = 0
    started = last = time.monotonic()

    def walk(folder):
        with os.scandir(folder) as stream:
            entries = sorted(stream, key=lambda e: e.name)
        for entry in entries:
            if entry.name in exclude:
                continue
            info = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                yield from walk(Path(entry.path))
            elif strict:
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise ValueError(
                        f"Only independent regular files allowed: {entry.name}"
                    )
                yield Path(entry.path), info
            elif entry.is_file():
                yield Path(entry.path), entry.stat()

    for path, info in walk(root):
        h = hashlib.sha256()
        flags = os.O_RDONLY | (os.O_NOFOLLOW if strict else 0) | os.O_NONBLOCK
        fd = os.open(path, flags)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(before.st_mode)
                or (strict and before.st_nlink != 1)
                or (info.st_dev, info.st_ino) != (before.st_dev, before.st_ino)
            ):
                raise ValueError(f"File changed during verification: {path}")
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
            after = os.fstat(stream.fileno())
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise ValueError(f"File changed during verification: {path}")
        result[str(path.relative_to(root))] = h.hexdigest()
        total += after.st_size
        tick = time.monotonic()
        if progress and tick - last >= 15:
            progress({"files": len(result), "bytes": total, "seconds": tick - started})
            last = tick
    if progress:
        progress(
            {
                "files": len(result),
                "bytes": total,
                "seconds": time.monotonic() - started,
            }
        )
    return dict(sorted(result.items()))
