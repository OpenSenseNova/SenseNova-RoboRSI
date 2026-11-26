#!/usr/bin/env python3
# Copyright 2026 The RoboRSI Contributors.
# Licensed under the Apache License, Version 2.0.

"""Freeze the task text and public scoring linked by the Agent Track rules.

The website currently renders its canonical specification from a static JS data
array. Parse that literal as data; never execute downloaded JavaScript. Only the
current task's description and scoring rows become Agent input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen

ORIGIN = "https://robodojo-benchmark.com"
CANONICAL_URL = ORIGIN + "/leaderboard/detail-description"
PROTOCOL_URL = ORIGIN + "/leaderboard/protocol?lang=en#agent-track-rules"


def sha(data):
    return hashlib.sha256(data).hexdigest()


class LiteralReader:
    """The strings, arrays and objects used by the official task data only."""

    def __init__(self, text, position):
        self.text, self.position = text, position

    def space(self):
        while self.position < len(self.text) and self.text[self.position].isspace():
            self.position += 1

    def take(self, wanted):
        self.space()
        if not self.text.startswith(wanted, self.position):
            raise ValueError("Official specification is not a static data literal")
        self.position += len(wanted)

    def read(self, depth=0):
        if depth > 12:
            raise ValueError("Official specification nesting is too deep")
        self.space()
        char = self.text[self.position : self.position + 1]
        if char == '"':
            value, end = json.JSONDecoder().raw_decode(self.text, self.position)
            self.position = end
            return value
        if char == "[":
            self.take("[")
            value = []
            self.space()
            while not self.text.startswith("]", self.position):
                value.append(self.read(depth + 1))
                self.space()
                if self.text.startswith("]", self.position):
                    break
                self.take(",")
            self.take("]")
            return value
        if char == "{":
            self.take("{")
            value = {}
            self.space()
            while not self.text.startswith("}", self.position):
                self.space()
                if self.text.startswith('"', self.position):
                    key = self.read(depth + 1)
                else:
                    match = re.match(r"[A-Za-z_$][\w$]*", self.text[self.position :])
                    if match is None:
                        raise ValueError("Invalid official specification key")
                    key = match.group()
                    self.position += len(key)
                if key in value:
                    raise ValueError("Duplicate official specification key")
                self.take(":")
                value[key] = self.read(depth + 1)
                self.space()
                if self.text.startswith("}", self.position):
                    break
                self.take(",")
            self.take("}")
            return value
        raise ValueError("Executable expressions are not official task data")


def specifications(source):
    candidates = []
    for match in re.finditer(
        r"\b[A-Za-z_$][\w$]*\s*=\s*(\[\s*\{\s*category\s*:)", source
    ):
        groups = LiteralReader(source, match.start(1)).read()
        rows = {}
        for group in groups:
            if set(group) != {"category", "tasks"}:
                raise ValueError("Official task group schema changed")
            for row in group["tasks"]:
                if (
                    set(row) != {"name", "description", "scoring"}
                    or not re.fullmatch(r"[A-Za-z0-9_]+", row["name"])
                    or not isinstance(row["description"], str)
                    or not row["description"].strip()
                    or not isinstance(row["scoring"], list)
                    or any(
                        not isinstance(text, str) or not text.strip()
                        for text in row["scoring"]
                    )
                ):
                    raise ValueError("Official task specification schema changed")
                if row["name"] in rows:
                    raise ValueError("Duplicate official task identity")
                rows[row["name"]] = row
        candidates.append(rows)
    if len(candidates) != 1:
        raise ValueError("Expected one canonical official task specification array")
    return candidates[0]


def render(row):
    text = row["description"]
    if row["scoring"]:
        text += "\n\nScoring\n" + "\n".join("- " + item for item in row["scoring"])
    return text + "\n"


def official_url(url):
    value = urlsplit(url)
    if value.scheme != "https" or value.netloc != "robodojo-benchmark.com":
        raise ValueError("Description source must remain on the official HTTPS origin")
    return url


def fetch(url):
    official_url(url)
    with urlopen(
        Request(url, headers={"User-Agent": "RoboDojo-canonical-description-snapshot"}),
        timeout=30,
    ) as response:
        official_url(response.url)
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError("Official source exceeds the snapshot size limit")
    return raw


def download_source():
    page = fetch(CANONICAL_URL).decode("utf-8")
    roots = re.findall(r'<script\b[^>]*\bsrc="([^"]+\.js)"', page)
    for root in roots:
        root_url = official_url(urljoin(CANONICAL_URL, root))
        main = fetch(root_url).decode("utf-8")
        # Inspect only same-origin modules referenced by the public page.
        modules = re.findall(r'import\("(\./[^"?#]+\.js)"\)', main)
        for url in [root_url] + [urljoin(root_url, module) for module in modules[:16]]:
            source = main if url == root_url else fetch(url).decode("utf-8")
            if "DetailDescriptionPage" in source and "category:" in source:
                try:
                    specifications(source)
                except ValueError:
                    continue
                return source, url
    raise ValueError(
        "Cannot locate the canonical Detail Description data; no Wiki fallback"
    )


def build_bundle(source, output, tasks, *, source_url, task_catalog=None):
    rows = specifications(source)
    aliases = {}
    if task_catalog is not None:
        catalog_raw = Path(task_catalog).read_bytes()
        catalog = json.loads(catalog_raw)["tasks"]
        for task in set(tasks) - set(rows):
            base = catalog.get(task, {}).get("base_task")
            # The website explicitly includes random variants within their
            # base section. A name suffix alone is not evidence of equivalence.
            if (
                not task.endswith("_random")
                or base not in rows
                or not re.search(
                    re.escape(task)
                    + r" uses (?:the same description|the nesting instructions above)\.",
                    rows[base]["description"],
                )
            ):
                raise ValueError(
                    "Requested task coverage lacks an explicit official shared description"
                )
            aliases[task] = base
    if (
        not tasks
        or len(set(tasks)) != len(tasks)
        or set(tasks) - set(rows) - set(aliases)
    ):
        raise ValueError(
            "Requested task coverage differs from the official specification"
        )
    official_url(source_url)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = {}
    for task in sorted(tasks):
        section_task = aliases.get(task, task)
        row = rows[section_task]
        payload = render(row).encode("utf-8")
        filename = task + ".txt"
        (output / filename).write_bytes(payload)
        records[task] = {
            "task": task,
            "url": CANONICAL_URL,
            "section": "Detail Description",
            "description": row["description"],
            "scoring": row["scoring"],
            "detail_file": filename,
            "detail_sha256": sha(payload),
            "includes_public_scoring": True,
            "requires_official_runtime_parameters": bool(
                re.search(
                    r"current language instruction|<target>", row["description"], re.I
                )
            ),
        }
        if task in aliases:
            records[task]["source_section_task"] = section_task
            records[task]["shared_description_explicit_in_official_text"] = True
    manifest = {
        "schema_version": 2,
        "origin": CANONICAL_URL,
        "protocol_url": PROTOCOL_URL,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "source_url": source_url,
        "source_sha256": sha(source.encode()),
        "task_count": len(records),
        "extraction": "Canonical description and all public scoring rows, without paraphrasing or task recipes",
        "tasks": records,
    }
    if aliases:
        manifest["task_catalog_sha256"] = sha(catalog_raw)
        manifest["shared_description_sections"] = aliases
    raw = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    (output / "MANIFEST.json").write_bytes(raw)
    return {
        "directory": str(output.resolve()),
        "task_count": len(records),
        "manifest_sha256": sha(raw),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="New immutable bundle directory"
    )
    parser.add_argument(
        "--source-js", type=Path, help="Previously archived official module; no network"
    )
    parser.add_argument(
        "--source-url", help="Original official URL of the archived module"
    )
    parser.add_argument(
        "--task",
        action="append",
        help="Repeat for a subset; defaults to all 42 standard tasks",
    )
    parser.add_argument(
        "--include-random",
        action="store_true",
        help="Include all 54 official catalog tasks, preserving explicit shared sections",
    )
    args = parser.parse_args()
    if args.source_js:
        if not args.source_url:
            parser.error("--source-js requires --source-url for provenance")
        source, url = args.source_js.read_text(), official_url(args.source_url)
    else:
        source, url = download_source()
    catalog = (
        Path(__file__).resolve().parents[2]
        / "robots/robodojo/agents/submission/skillpack/task_catalog.json"
    )
    tasks = args.task or [
        task
        for task in json.loads(catalog.read_text())["tasks"]
        if args.include_random or not task.endswith("_random")
    ]
    print(
        json.dumps(
            build_bundle(
                source, args.output, tasks, source_url=url, task_catalog=catalog
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
