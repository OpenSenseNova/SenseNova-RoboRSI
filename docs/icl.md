# Build and use ICL materials

[简体中文](icl.zh-CN.md) · [All guides](README.md)

ICL packages preserve selected notes, skills, images, programs and supporting
files. Building or loading a package does not execute its programs or call a
model. The caller chooses the material and its context exports.

## Try the file lifecycle offline

From the installed repository, use a new output directory:

```bash
python -m examples.components.study_and_pack --dry-run --output work/icl-demo
python -m roborsi.cli.icl verify work/icl-demo/package
python -m roborsi.cli.icl inspect work/icl-demo/snapshot
```

The example uses synthetic notes and an unexecuted program. It builds a sealed
package, edits an independent working copy and seals a new snapshot. It needs
neither a simulator nor model credentials. Real study uses caller-supplied
Agent tools; `study_and_pack.py` shows the deferred `ready` completion boundary.

## Build, copy and snapshot

```python
from pathlib import Path
from roborsi.components.icl import (
    ContextExports, ContextPackage, build_package,
    materialize_package, snapshot_workspace,
)
from roborsi.components.icl.adapters import context_from, make_icl_tools

package = build_package(
    Path("selected-materials"), Path("reference-v1"), package_id="reference-v1",
    exports=ContextExports(memory=(("Task", "TASK.md"),)),
    metadata={"dataset": "caller-selected"},
)
verified = ContextPackage.load(package.root)
workspace = materialize_package(verified, Path("case-001/workspace"))
context = context_from(workspace, planner="api")
tools = make_icl_tools(workspace, name_prefix="reference")
(workspace.root / "TASK.md").write_text("New experience", encoding="utf-8")
revision = snapshot_workspace(workspace, Path("reference-v2"), package_id="reference-v2")
```

The complete input tree is copied; exports select only what becomes context.
Declare `skills`, `skill_paths` and typed text/image `initial_context` explicitly.
Supply program dependencies yourself. Source and destination must be disjoint;
destinations must be new. Empty trees, missing exports, links, special files and
nonportable name collisions fail. Equal files, exports, identity and metadata
produce equal manifest bytes without a generated timestamp.

The sealed layout remains `manifest.json` plus `payload/`, using
`rpent.icl.context`, schema version 1. Loading verifies all payload SHA-256
hashes and rejects extra files. Legacy imports from
`roborsi.agents.context_package` remain supported. Workspaces are independent
ordinary files. Stop all writers before snapshotting; snapshots preserve exports
and record `metadata.icl.parent_manifest_sha256`. Deleted exports require new
declarations. Keep Python bytecode outside sealed payloads.

## Connect to an Agent

Pass `context.memory`, `skills`, `skill_paths` and `initial_context` to
`EmbodiedAgent.run` or `start_episode`, and register `tools` explicitly. For
Runner evaluation, supply `evaluation_memory`, `evaluation_skills`,
`evaluation_skill_paths` and `evaluation_initial_context` at construction; see
[icl_build_and_load.py](../examples/components/icl_build_and_load.py).

`context_from` accepts `api`, `codex`, `codex_cli` and `claude_code` planners.
Skill handling is explicit:

| `skill_mode` | Behavior |
| --- | --- |
| `declared` | Preserve declarations; directory skills require the API planner. |
| `preload` | Discover declared skill collections and preload their `SKILL.md` files. |
| `on_demand` | Use declared skill directories with the API planner. |
| `none` | Omit skills; preserve memory and initial context. |

For Codex CLI directory exports select `preload` or `none`. Configure its
`expected_version` and use null model turn/token limits as required by the
[Agent guide](agents.md). The read-only tools expose `<prefix>_list`,
`<prefix>_read_text` and `<prefix>_read_image`, with pagination, truncation and
rechecked paths. Sealed reads verify hashes; workspace reads see current files.
Writing and execution tools remain the caller's responsibility.

## CLI and pinned sources

Use `roborsi-icl` or `python -m roborsi.cli.icl`. Success emits JSON to stdout;
errors go to stderr with a nonzero exit status. A build spec resolves source
and destination relative to its own directory:

```json
{
  "source": "selected-materials", "destination": "reference-v1",
  "package_id": "reference-v1",
  "exports": {"memory": [{"title": "Task", "path": "TASK.md"}]},
  "metadata": {"dataset": "caller-selected"}
}
```

```bash
roborsi-icl build --spec build.json
roborsi-icl verify reference-v1
roborsi-icl inspect reference-v1
roborsi-icl fetch --source-json source.json --cache-dir icl-cache
roborsi-icl fetch --source-json source.json --cache-dir icl-cache --offline
roborsi-icl materialize reference-v1 case-001/workspace --workspace-record case-001/workspace.json
roborsi-icl snapshot --workspace-record case-001/workspace.json reference-v2 --package-id reference-v2
roborsi-icl import-policylab shared-task imported-package --package-id imported-v1
```

`source.json` selects one source:

```text
{"type": "local", "path": "reference-v1"}
{"type": "http", "url": "https://host/package.zip", "archive_sha256": "<64 lowercase hex>"}
{"type": "hf", "repo_id": "owner/repository", "revision": "<40 lowercase commit hex>", "subpath": "packages/reference", "repo_type": "dataset"}
```

HTTP/HF sources also accept `manifest_sha256`. ZIP roots must contain exactly
the package layout. Downloads default to 1 GiB and unpacked content to 4 GiB;
override using `--max-download-bytes` and `--max-unpacked-bytes`. Cache receipts
record source identity outside the package. Every cache hit is verified;
`--offline` requires the exact pinned source already cached.

The workspace receipt must be new and outside both the workspace and source
package. After moving a workspace, update its recorded root. Snapshot supports
`--exports` and `--metadata` JSON files. PolicyLab import accepts one task with
`workspace/` and `provenance.json`, preserves unknown provenance and copies
original attachments under `payload/__policylab_source__/`; it does not rewrite
historical paths or independently verify source training claims.

Declare packages in Agent preset `resources` for fingerprinting and wheel export.
Keep generated data and traces outside the checkout. For the complete task
learning workflow see [BEHAVIOR-1K](../benchmarks/behavior_1k/README.md#icl-and-evaluation).
The offline example and tests establish file and interface behavior, not native
model or simulator performance.
