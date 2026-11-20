# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Keep private reset states outside public Task and Observation payloads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from roborsi.evaluation.messages import Task
from robots.simple.contracts import SIMPLE_REVISION, SOURCES, validate_task


class Catalog:
    """Read an immutable catalog and hash-check each selected private reset."""

    def __init__(self, path: Path) -> None:
        if path.is_symlink():
            raise ValueError("catalog must be a regular file")
        self.path = path.resolve()
        self.document = json.loads(self.path.read_text())
        if (
            self.document.get("schema_version") != 1
            or self.document.get("simple_revision") != SIMPLE_REVISION
            or self.document.get("dataset_revision") != SOURCES["dataset_revision"]
        ):
            raise ValueError("SIMPLE catalog source revisions differ")
        self._rows = {}
        for row in self.document["cases"]:
            task = Task.from_dict(row["task"])
            validate_task(task)
            if task.task_id in self._rows:
                raise ValueError("duplicate SIMPLE case identity")
            self._rows[task.task_id] = row
            self.reset(task)
        if not self._rows:
            raise ValueError("SIMPLE catalog cannot be empty")

    def tasks(self) -> list[Task]:
        """Return only public task identities; reset geometry remains private."""
        return [Task.from_dict(r["task"]) for r in self._rows.values()]

    def reset(self, task: Task) -> dict:
        """Require exact catalog membership and unchanged reset bytes."""
        row = self._rows.get(task.task_id)
        if row is None or Task.from_dict(row["task"]) != task:
            raise ValueError("SIMPLE task differs from the published catalog")
        ref = Path(row["reset"])
        if ref.is_absolute() or ".." in ref.parts or ref.parts[:1] != ("resets",):
            raise ValueError("private reset must remain inside catalog/resets")
        path = self.path.parent / ref
        if any(p.is_symlink() for p in [path, *path.parents]):
            raise ValueError("reset paths must not contain symlinks")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != task.native["reset_sha256"]:
            raise ValueError("SIMPLE reset SHA256 differs")
        return json.loads(data)
