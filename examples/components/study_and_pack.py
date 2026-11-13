# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Study is user policy; packaging starts only after Agent writes have stopped.

Run ``python -m examples.components.study_and_pack --dry-run --output <new-dir>``
to exercise the file lifecycle offline. For a model-driven study, construct an
EmbodiedAgent with your own file/program tools and READY_TOOL, then call
study_and_pack. No simulator is required during study.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from roborsi.components.icl import (
    ContextExports,
    ContextPackage,
    build_package,
    materialize_package,
    snapshot_workspace,
)
from roborsi.components.icl.adapters import context_from, make_icl_tools
from roborsi.embodied_agent import EmbodiedAgent, LocalToolSpec
from roborsi.tools.toolkit import ToolResult

READY_TOOL = LocalToolSpec(
    "ready",
    "Finish studying after saving the selected materials.",
    {"type": "object", "properties": {}, "additionalProperties": False},
)


def study_and_pack(
    agent: EmbodiedAgent,
    workspace: Path,
    destination: Path,
    *,
    package_id: str,
    exports: ContextExports,
    timeout_s: float = 120,
) -> ContextPackage:
    """Use a deferred ready boundary, then seal the stopped study directory.

    The caller supplies study tools and owns the Agent. A timeout propagates;
    no package is produced until all admitted file handlers finish cleanup.
    ``ready`` records completion of study, not benchmark success or skill quality.
    """
    with agent.start_episode(
        "Study the selected training materials. Save notes, examples and evidence, then call ready.",
        system_prompt="Use only the supplied study tools. Record uncertainty and source information. Call ready after saving your materials.",
        deferred_tools=("ready",),
        include_finish=False,
    ) as episode:
        call = episode.next_call(timeout=timeout_s)
        if call is None:
            episode.wait(timeout=timeout_s)
            raise RuntimeError("study ended without ready")
        if call.name != "ready":
            raise RuntimeError("unexpected deferred study tool")
        episode.complete(
            call, ToolResult(call.name, {"_finish": True, "status": "ready"})
        )
        result = episode.wait(timeout=timeout_s)
        if result.error:
            raise RuntimeError("study failed after ready")
    return build_package(
        workspace,
        destination,
        package_id=package_id,
        exports=exports,
        metadata={
            "icl": {
                "generation": {"method": "user_agent", "completion": "ready"},
                "original_provenance": None,
            }
        },
    )


def dry_run(output: Path) -> dict[str, str]:
    """Exercise file operations with clearly synthetic user-generated materials."""
    output.mkdir(parents=True, exist_ok=False)
    source = output / "study"
    source.mkdir()
    (source / "notes.txt").write_text(
        "Synthetic reference: inspect before moving.\n", encoding="utf-8"
    )
    (source / "analyze.py").write_text(
        "# Stored reference; never executed by ICL.\n", encoding="utf-8"
    )
    package = build_package(
        source,
        output / "package",
        package_id="dry-run",
        exports=ContextExports(memory=(("Notes", "notes.txt"),)),
        metadata={
            "icl": {
                "generation": {"method": "offline_fixture"},
                "original_provenance": None,
            }
        },
    )
    workspace = materialize_package(package, output / "case" / "materials")
    context = context_from(workspace, planner="api")
    assert context.memory[0].text.startswith("Synthetic")
    assert (
        make_icl_tools(workspace)[1]
        .handler({"path": "analyze.py"})
        .result["text"]
        .startswith("#")
    )
    (workspace.root / "notes.txt").write_text(
        "Synthetic updated experience.\n", encoding="utf-8"
    )
    snapshot = snapshot_workspace(
        workspace, output / "snapshot", package_id="dry-run-revised"
    )
    return {
        "package": str(package.root),
        "manifest_sha256": package.manifest_sha256,
        "snapshot_sha256": snapshot.manifest_sha256,
    }


def main() -> None:
    """Run the explicit offline example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", required=True, action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(dry_run(args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
