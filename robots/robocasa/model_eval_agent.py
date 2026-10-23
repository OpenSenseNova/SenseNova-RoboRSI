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

"""Bind the published Target50 seed-0 memory to its own evaluation task."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path, PurePosixPath

from roborsi.evaluation.messages import Action, Observation, StepResult, Task
from roborsi.evaluation.result import write_json_atomic

# The planner's own timeout sits past the Target50 limit so that the Agent ends
# the episode at the limit and the scene at that moment is scored.
PLANNER_GRACE_S = 900
PLANNER_ENDED = "Embodied Agent ended before the Benchmark episode"
CHECKSUMS_SHA256 = "88dfecd78fada13a6fc7aa2b13d1a7c61cb005451b72d53c0d3f25cc2f6a90bf"
PACKAGE_REVISION = "e977b5ad667b4ce753ee947cbb0b95d4fc0fe98c"
NO_MEMORY = frozenset(
    {
        "HeatKebabSandwich",
        "PanTransfer",
        "PortionHotDogs",
        "SeparateFreezerRack",
        "WaffleReheat",
        "WashFruitColander",
        "WeighIngredients",
    }
)


def _checksum(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("ICL payload must be a regular file")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path(root: Path, relative: str) -> Path:
    parsed = PurePosixPath(relative)
    if (
        not relative
        or "\\" in relative
        or ":" in relative
        or parsed.is_absolute()
        or ".." in parsed.parts
        or parsed.as_posix() != relative
    ):
        raise ValueError("ICL paths must be normalized relative paths")
    path = root / relative
    if any(part.is_symlink() for part in (path, *path.parents) if part != root.parent):
        raise ValueError("ICL paths cannot contain links")
    return path


def _release(root: Path, checksums_sha256: str) -> tuple[dict, dict[str, str]]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("ICL package must be a real directory")
    manifest = _path(root, "MANIFEST.sha256")
    if _checksum(manifest) != checksums_sha256:
        raise ValueError("ICL manifest checksum mismatch")
    entries = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise ValueError("Invalid ICL checksum manifest")
        digest, name = match.groups()
        path = _path(root, name)
        if name in entries:
            raise ValueError("Duplicate ICL manifest path")
        if _checksum(path) != digest:
            raise ValueError(f"ICL payload checksum mismatch: {name}")
        entries[name] = digest
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if not p.is_dir()}
    if actual - {"MANIFEST.sha256", ".download.json"} != set(entries):
        raise ValueError("ICL package inventory differs from checksum manifest")
    index = json.loads(
        _path(root, "knowledge_packages.json").read_text(encoding="utf-8")
    )
    if (
        index.get("schema_version") != 1
        or index.get("benchmark_id") != "robocasa365"
        or index.get("regime") != "target50_same_task_seed0"
    ):
        raise ValueError("Unsupported RoboCasa ICL package")
    policy = index.get("memory_policy", {})
    if (
        policy.get("scope") != "same_task_seed_0"
        or policy.get("use_global_memory") is not False
        or policy.get("use_cross_task_memory") is not False
        or set(policy.get("tasks_without_memory", [])) != NO_MEMORY
    ):
        raise ValueError("RoboCasa memory policy differs from the published release")
    return index, entries


def load_task_context(
    root: Path,
    task: Task,
    *,
    checksums_sha256: str = CHECKSUMS_SHA256,
    package_revision: str = PACKAGE_REVISION,
) -> tuple[tuple[str, ...], dict]:
    """Verify a release and return only this cell's own source-seed-0 memory."""
    split = task.native.get("split")
    limit = 10 if split == "atomic" else 5
    if (
        task.benchmark != "robocasa365"
        or task.native.get("protocol") != "robocasa-harness-vla-v1"
        or task.native.get("environment_split") != "target"
        or split not in {"atomic", "composite_seen", "composite_unseen"}
        or not 1 <= task.seed <= limit
    ):
        raise ValueError("ICL package requires a Target50 evaluation cell")
    root = Path(root)
    index, checksums = _release(root, checksums_sha256)
    key = f"{split}/{task.task_id}_s{task.seed}"
    evidence = {
        "task_id": task.task_id,
        "case_key": key,
        "source_seed": 0,
        "checksums_sha256": checksums_sha256,
        "package_revision": package_revision,
    }
    if task.task_id in NO_MEMORY:
        if task.task_id in index["packages"] or key in index["case_map"]:
            raise ValueError("Declared memory-free task unexpectedly has a package")
        return (), {**evidence, "status": "unavailable_declared"}
    if index["case_map"].get(key) != task.task_id:
        raise ValueError(f"No same-task memory is mapped for {key}")
    package = index["packages"][task.task_id]
    prefix = package["path"]
    if prefix != f"tasks/{task.task_id}" or package.get("split") != split:
        raise ValueError("ICL task package identity mismatch")
    manifest_name = f"{prefix}/manifest.json"
    if checksums.get(manifest_name) != package["manifest_sha256"]:
        raise ValueError("ICL task manifest hash mismatch")
    manifest = json.loads(_path(root, manifest_name).read_text(encoding="utf-8"))
    payloads = package["payload_files"]
    required = {
        f"robocasa/results/{task.task_id}_s0.json",
        f"robocasa/results/recipe_{task.task_id}_s0.jsonl",
    }
    if (
        manifest.get("task") != task.task_id
        or manifest.get("split") != split
        or manifest.get("regime") != "target50_same_task_seed0"
        or manifest.get("evaluation_trajectories_used") is not False
        or manifest.get("payload_files") != payloads
        or not required <= set(payloads)
        or set(payloads) - required - {f"robocasa/results/{task.task_id}.md"}
    ):
        raise ValueError("ICL payload declaration differs from the same-task policy")
    context = [
        "Reference experience for this same task, recorded at source seed 0. "
        "Use the current task, live observations and available action schema to "
        "choose actions; the examples describe a different episode."
    ]
    for name, digest in sorted(payloads.items()):
        full_name = f"{prefix}/payload/{name}"
        if checksums.get(full_name) != digest:
            raise ValueError("ICL task payload hash mismatch")
        context.append(
            f"Reference file {name}:\n{_path(root, full_name).read_text(encoding='utf-8')}"
        )
    return tuple(context), {**evidence, "status": "loaded", "payload_sha256": payloads}


class RoboCasaMemoryAgent:
    """Add verified task memory before starting a normal EmbodiedAgent.

    When the Target50 time limit or the planner's turn budget runs out, the
    Agent finishes the episode itself so the Benchmark scores the scene as it
    stands, as the Target50 protocol does.
    """

    def __init__(
        self,
        delegate,
        case_dir: Path,
        package: Path,
        *,
        checksums_sha256=CHECKSUMS_SHA256,
        package_revision=PACKAGE_REVISION,
        finish_on_budget: bool = True,
        clock=time.monotonic,
    ):
        self.delegate = delegate
        self.case_dir, self.package = Path(case_dir), Path(package)
        self.checksums_sha256 = checksums_sha256
        self.package_revision = package_revision
        self.finish_on_budget = finish_on_budget
        self._clock = clock
        self._deadline: float | None = None
        self._ended_by_budget = False
        self._started = False

    def reset(self, task: Task) -> None:
        if self._started:
            raise ValueError("RoboCasa ICL Agent owns one evaluation case")
        context, evidence = load_task_context(
            self.package,
            task,
            checksums_sha256=self.checksums_sha256,
            package_revision=self.package_revision,
        )
        self.case_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self.case_dir / "icl-evidence.json", evidence)
        self.delegate.evaluation_initial_context = (
            *self.delegate.evaluation_initial_context,
            *context,
        )
        limit = int(task.native["timeout_s"])
        if self.finish_on_budget:
            self.delegate.planner_timeout_s = limit + PLANNER_GRACE_S
            self._deadline = self._clock() + limit
        self.delegate.reset(task)
        self._started = True

    def act(self, observation: Observation) -> Action:
        if self._deadline is not None and self._clock() >= self._deadline:
            return self._finish(observation, "Target50 time limit reached")
        try:
            action = self.delegate.act(observation)
        except RuntimeError as exc:
            if not (self.finish_on_budget and PLANNER_ENDED in str(exc)):
                raise
            return self._finish(observation, "planner turn limit reached")
        if self._deadline is not None and self._clock() >= self._deadline:
            return self._finish(observation, "Target50 time limit reached")
        return action

    def _finish(self, observation: Observation, reason: str) -> Action:
        self._ended_by_budget = True
        write_json_atomic(
            self.case_dir / "budget-finish.json",
            {"step_index": observation.step_index, "reason": reason},
        )
        return Action(
            benchmark=observation.benchmark,
            episode_id=observation.episode_id,
            step_index=observation.step_index,
            data={"primitive": "finish", "status": "stuck", "summary": reason},
        )

    def observe_end(self, step: StepResult):
        if self._ended_by_budget:
            # The planner's last proposal was not executed; end its dialogue.
            return self.delegate.close()
        return self.delegate.observe_end(step)

    def close(self):
        return self.delegate.close()


def make_agent(case_dir: Path, *, config) -> RoboCasaMemoryAgent:
    """Create the task-scoped ICL Agent from an automatically prepared preset."""
    if set(config.resources) != {"icl_package"}:
        raise ValueError("RoboCasa ICL Agent requires resources.icl_package")
    parameters = config.parameters
    if set(parameters) - {"icl_manifest_sha256", "icl_revision", "finish_on_budget"}:
        raise ValueError("Unknown RoboCasa ICL Agent parameter")
    checksum = parameters.get("icl_manifest_sha256", CHECKSUMS_SHA256)
    revision = parameters.get("icl_revision", PACKAGE_REVISION)
    if (
        not isinstance(checksum, str)
        or not re.fullmatch(r"[0-9a-f]{64}", checksum)
        or not isinstance(revision, str)
        or not re.fullmatch(r"[0-9a-f]{40}", revision)
    ):
        raise ValueError("RoboCasa ICL requires a pinned manifest SHA256 and revision")
    finish_on_budget = parameters.get("finish_on_budget", True)
    if type(finish_on_budget) is not bool:
        raise ValueError("parameters.finish_on_budget must be a boolean")
    return RoboCasaMemoryAgent(
        config.embodied_agent(case_dir),
        case_dir,
        config.resources["icl_package"],
        checksums_sha256=checksum,
        package_revision=revision,
        finish_on_budget=finish_on_budget,
    )
