# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Extract RGB frames from task-scoped rich demonstrations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

MODALITY_ALIASES = {"rgb": "rgb"}


def _safe(root: Path, relative: str) -> Path:
    candidate = root / relative
    if candidate.is_symlink():
        raise ValueError("Requested video is a symlink")
    path = candidate.resolve()
    if not path.is_file() or not path.is_relative_to(root.resolve()):
        raise ValueError("Requested video is outside the approved training bundle")
    return path


def _episode(index: dict, episode_id: str) -> dict:
    matches = [row for row in index["episodes"] if row.get("id") == episode_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown rich demonstration episode: {episode_id}")
    if (
        not isinstance(episode_id, str)
        or not episode_id.replace("-", "").replace("_", "").isalnum()
    ):
        raise ValueError("Invalid rich demonstration episode id")
    return matches[0]


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_frames(
    episode_id: str,
    camera: str,
    modality: str,
    timestamps: list[float],
    output_dir: Path,
    *,
    root: Path = Path("training"),
) -> list[dict]:
    from imageio_ffmpeg import get_ffmpeg_exe

    if root.is_symlink() or not root.is_dir():
        raise ValueError("Training bundle root is missing or unsafe")
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    episode = _episode(index, episode_id)
    modality = MODALITY_ALIASES.get(modality, modality)
    if camera not in episode["videos"] or modality != "rgb":
        raise ValueError("Only RGB videos are available during rich icl")
    relative = episode["videos"][camera][modality]
    source = _safe(root, relative)
    manifest = next((item for item in index["files"] if item["path"] == relative), None)
    if manifest is None:
        raise ValueError("Video is missing from the rich file manifest")
    digest = _digest(source)
    if digest != manifest["sha256"]:
        raise ValueError("Rich video hash mismatch")
    interval = episode["intervals"][camera][modality]
    fps = float(interval["fps"])
    duration = float(interval["frames"]) / fps
    if not timestamps:
        raise ValueError("Provide at least one timestamp")
    frames = []
    for timestamp in timestamps:
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(timestamp)
            or timestamp < 0
            or timestamp >= duration
        ):
            raise ValueError("Timestamps must be inside the selected episode interval")
        frames.append(int(math.floor(timestamp * fps + 1e-9)))
    output_dir = Path(output_dir)
    if output_dir.exists() and output_dir.is_symlink():
        raise ValueError("Output directory must not be a symlink")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for timestamp, frame in zip(timestamps, frames, strict=True):
        target = output_dir / f"{episode_id}-{camera}-{modality}-{frame:08d}.png"
        if target.is_symlink():
            raise ValueError("Output frame path must not be a symlink")
        absolute_timestamp = float(interval["start_timestamp"]) + frame / fps
        subprocess.run(
            [
                get_ffmpeg_exe(),
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                str(source),
                "-ss",
                f"{absolute_timestamp:.9f}",
                "-frames:v",
                "1",
                "-map",
                "0:v:0",
                str(target),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        results.append(
            {"path": str(target.resolve()), "timestamp": timestamp, "frame": frame}
        )
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True)
    parser.add_argument(
        "--camera", choices=("head", "left_wrist", "right_wrist"), required=True
    )
    parser.add_argument("--modality", choices=("rgb",), required=True)
    parser.add_argument("--timestamps", type=float, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("training"))
    args = parser.parse_args(argv)
    try:
        print(
            json.dumps(
                extract_frames(
                    args.episode,
                    args.camera,
                    args.modality,
                    args.timestamps,
                    args.output_dir,
                    root=args.root,
                ),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
    ) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
