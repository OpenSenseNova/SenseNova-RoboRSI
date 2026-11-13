# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Read task-scoped rich demonstration records inside the model sandbox."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path


def _index(root: Path = Path("training")) -> dict:
    return json.loads((root / "index.json").read_text(encoding="utf-8"))


def _episode(index: dict, episode_id: str) -> dict:
    if not isinstance(episode_id, str) or not re.fullmatch(
        r"[A-Za-z0-9_-]+", episode_id
    ):
        raise ValueError("Invalid rich demonstration episode id")
    matches = [row for row in index["episodes"] if row.get("id") == episode_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown rich demonstration episode: {episode_id}")
    return matches[0]


def iter_records(episode_id: str, *, root: Path = Path("training")):
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Training bundle root is missing or unsafe")
    index = _index(root)
    episode = _episode(index, episode_id)
    path = (root / episode["records"]).resolve()
    if (
        (root / episode["records"]).is_symlink()
        or not path.is_file()
        or not path.is_relative_to(root.resolve())
    ):
        raise ValueError("Record path is outside the approved training bundle")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--head", type=int, default=5)
    parser.add_argument("--root", type=Path, default=Path("training"))
    args = parser.parse_args(argv)
    if args.head < 1:
        parser.error("--head must be positive")
    try:
        for number, record in enumerate(iter_records(args.episode, root=args.root)):
            if number >= args.head:
                break
            print(json.dumps(record, ensure_ascii=False, allow_nan=False))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
