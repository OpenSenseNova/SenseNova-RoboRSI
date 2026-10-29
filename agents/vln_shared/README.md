# Shared navigation Agent installation

This directory owns the shared Python dependency lock and keeps `agent.yaml`
as a compatibility alias. Use the standard Benchmark entries:

- [VLNCE](../../robots/vlnce/agent.yaml) for R2R and RxR.
- [LHPR-VLN](../../robots/lhpr_vln/agent.yaml) for LHPR.

Both inherit the [shared baseline](../../robots/vln_shared/agents/baseline/README.md).
The factory, prompts and tool policy live there once; the Benchmark entries
contain only `extends` layers. The one-click launcher uses these entries to
generate self-contained dataset presets with frozen training memory.

The dependency lock targets Python 3.11 and RoboRSI base
`67fe9bc6daf985e438be0396ef4206a7f98b4d6b`. The external CLI executable has its
own exact version preflight; the Python Codex SDK package is not its launcher.

Usage: [English](../../docs/vln_shared_evaluation.md) ·
[中文](../../docs/vln_shared_evaluation.zh-CN.md).
