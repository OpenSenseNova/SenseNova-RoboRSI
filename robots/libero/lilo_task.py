# Copyright 2026 The SenseNova-RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Build LiLo instructions from the same ordered skills used by its scorer."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

INSTRUCTION_CONTRACT = "lilo-ordered-skills-v1"


def official_skill_names(
    config: dict[str, Any], suite: str, task_name: str, bddl_stem: str
) -> list[str]:
    """Locate exactly one released ordered skill sequence."""
    if suite.startswith("ultra_long"):
        family = "ultra_long"
        ordering = (
            "Reorder 2"
            if task_name.endswith("_v3")
            else "Reorder 1"
            if task_name.endswith("_v2")
            else "Original"
        )
    else:
        family = "libero_long_plus_plus"
        ordering = "Reorder 1" if task_name.endswith("_variant") else "Original"
    matches = [
        row
        for row in config.get("long_horizon_tasks", [])
        if row.get("benchmark") == family
        and row.get("bddl_file") == bddl_stem
        and row.get("name", "").endswith(f"({ordering})")
    ]
    if len(matches) != 1:
        raise ValueError(f"LiLo skill sequence is ambiguous: {suite} {task_name}")
    skills = matches[0].get("skills")
    if (
        not isinstance(skills, list)
        or not skills
        or any(not isinstance(skill, str) or not skill.strip() for skill in skills)
    ):
        raise ValueError("LiLo skill sequence must contain nonempty text")
    return list(skills)


def build_task_contract(
    config: dict[str, Any], suite: str, task: Any, bddl_path: str | Path
) -> dict[str, Any]:
    """Create the exact ordered instruction and its content identity."""
    skills = official_skill_names(config, suite, task.name, Path(task.bddl_file).stem)
    entities = scene_entities(Path(bddl_path))
    mappings = config.get("skill_mappings", {})
    for skill in skills:
        mapping = mappings.get(skill.lower().strip())
        predicates = (
            mapping.get("bddl_predicates") if isinstance(mapping, dict) else None
        )
        if not isinstance(predicates, list) or not predicates:
            raise ValueError(f"LiLo skill has no released predicate mapping: {skill}")
        references = [mapping.get("target_object")]
        for predicate in predicates:
            if not isinstance(predicate, list) or len(predicate) < 2:
                raise ValueError(f"LiLo skill predicate is invalid: {skill}")
            references.extend(predicate[1:])
        for reference in references:
            if not isinstance(reference, str) or not any(
                reference == name or reference.startswith(name + "_")
                for name in entities
            ):
                raise ValueError(
                    f"LiLo skill references an object absent from the scene: "
                    f"{skill} -> {reference}"
                )
    instruction = (
        "Complete these subtasks in this exact order. Instance numbers identify "
        "specific scene objects and must be preserved:\n"
        + "\n".join(f"{index}. {skill}" for index, skill in enumerate(skills, 1))
    )
    fingerprint = {
        "version": INSTRUCTION_CONTRACT,
        "suite": suite,
        "task": task.name,
        "bddl_file": Path(bddl_path).name,
        "skills": skills,
        "entities": sorted(entities),
    }
    return {
        "version": INSTRUCTION_CONTRACT,
        "skills": skills,
        "instruction": instruction,
        "source_language": task.language,
        "sha256": hashlib.sha256(
            json.dumps(fingerprint, sort_keys=True).encode()
        ).hexdigest(),
    }


def scene_entities(bddl: Path) -> set[str]:
    """Read declared object/fixture names without reading poses or goal state."""
    text = re.sub(r";[^\n]*", "", bddl.read_text(encoding="utf-8"))
    entities: set[str] = set()
    for section in re.findall(r"\(:(?:objects|fixtures)\s+([^()]*)\)", text):
        tokens = section.split()
        skip_type = False
        for token in tokens:
            if skip_type:
                skip_type = False
            elif token == "-":
                skip_type = True
            else:
                entities.add(token)
    if not entities:
        raise ValueError(f"LiLo BDDL declares no objects or fixtures: {bddl}")
    return entities


class OrderedSkillTracker:
    """Credit only the uninterrupted ordered prefix of released predicates."""

    def __init__(self, env: Any, skills: list[str], checker: Any) -> None:
        if not skills:
            raise ValueError("ordered task requires at least one skill")
        self.env, self.skills, self.checker = env, list(skills), checker
        self.completed = 0
        self._prime_pick()

    def _prime_pick(self) -> None:
        if self.completed >= len(self.skills):
            return
        skill = self.skills[self.completed]
        if "pick" in skill.lower():
            self.checker.reset_predicate_baselines(skill)
            self.checker.check_skill_success_by_language(self.env, skill)

    def observe(self) -> bool:
        """Advance by at most one skill after one physical environment step."""
        if self.completed < len(self.skills):
            skill = self.skills[self.completed]
            if self.checker.check_skill_success_by_language(self.env, skill):
                self.completed += 1
                self._prime_pick()
        return self.success

    @property
    def success(self) -> bool:
        return self.completed == len(self.skills)

    @property
    def progress(self) -> float:
        return self.completed / len(self.skills)
