# Copyright 2026 The RoboRSI Contributors.
# Licensed under the Apache License, Version 2.0.

"""Bind the submitted General Track controller to the model evaluation entry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from roborsi.agents.config import AgentConfig


def validate_configuration(config) -> dict:
    agent = AgentConfig.from_file(config.agent)
    parameters = {**agent.parameters, **config.agent_overrides.get("parameters", {})}
    if (
        agent.factory != "robots.robodojo.agents.factory:make_agent"
        or agent.parameters.get("skill_library_mode") != "general_only"
        or agent.parameters.get("eef_execution_backend_by_task") != {}
        or parameters.get("skill_library_mode") != "general_only"
        or parameters.get("eef_execution_backend_by_task") != {}
        or parameters.get("public_icl_root")
        or parameters.get("provider_kind") not in {"responses", "cci"}
    ):
        raise ValueError(
            "RoboDojo reproduction requires the general_only Agent Track preset without case/ICL or task routing"
        )
    return parameters


def description_parameters(config, work_dir: Path) -> dict:
    """Freeze canonical descriptions and public scoring; never prepare ICL."""
    from benchmarks.robodojo.official_descriptions import (
        CANONICAL_URL,
        build_bundle,
        download_source,
    )
    from robots.robodojo.agents.submission.agent_track import _downloaded_detail

    parameters = validate_configuration(config)
    selected = parameters.get("official_description_bundle")
    digest = parameters.get("official_description_manifest_sha256")
    if bool(selected) != bool(digest):
        raise ValueError(
            "Provide both the official description bundle and its manifest SHA256"
        )
    bundle = (
        Path(selected).expanduser().resolve()
        if selected
        else work_dir / "assets/robodojo-official-descriptions"
    )
    if not selected and not bundle.exists():
        source, url = download_source()
        catalog = (
            Path(__file__).resolve().parents[2]
            / "robots/robodojo/agents/submission/skillpack/task_catalog.json"
        )
        build_bundle(
            source,
            bundle,
            list(json.loads(catalog.read_text())["tasks"]),
            source_url=url,
            task_catalog=catalog,
        )
    manifest = bundle / "MANIFEST.json"
    raw = manifest.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if digest and digest != actual:
        raise ValueError("Official description manifest SHA256 differs")
    document = json.loads(raw)
    if (
        document.get("schema_version") != 2
        or document.get("origin") != CANONICAL_URL
        or not document.get("tasks")
    ):
        raise ValueError(
            "RoboDojo reproduction requires complete canonical descriptions and public Scoring"
        )
    for task in document["tasks"]:
        _downloaded_detail(task, bundle, actual)
    return {
        "official_description_bundle": str(bundle),
        "official_description_manifest_sha256": actual,
    }
