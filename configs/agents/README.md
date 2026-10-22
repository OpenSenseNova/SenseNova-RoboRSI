# Agent presets

`embodied/` contains generic Agent configurations; `libero/` holds the specialized
LIBERO presets and context maps, and `robocasa365/icl.yaml` selects the same-task
ICL factory. Choose prompts, context and budgets here.

For `evaluate_model.py --benchmark NAME`, select the environment and Agent in
`../benchmarks/NAME.yaml` and the shared model in `../models/default.yaml`.
Model connection overrides and benchmark `agent_overrides` are composed over the
Agent YAML. See the [configuration layout](../README.md). Direct `bench eval`
and installed bundles continue to use the selected Agent's own model settings.

| Preset | Purpose |
| --- | --- |
| `embodied/default.yaml` | Model API, eight recent feedback turns |
| `embodied/short_context.yaml` | Same model, four recent feedback turns |
| `embodied/codex.yaml` | Codex SDK |
| `embodied/codex_cli.yaml` | Separately installed, pinned Codex CLI |
| `embodied/claude_code.yaml` | Claude SDK |
| `libero/standard.yaml` | Standard LIBERO, optional pi05 VLA |
| `libero/pro_vla.yaml` | PRO with automatic pi05 VLA assistance enabled |

LIBERO presets use `parameters.enable_vla: true` to opt into automatic download
and serving. The selected model YAML's `vla` section controls the execution model;
Agent YAML does not reference a separate model profile. See [LIBERO VLA](../../docs/libero-vla.md)
/ [VLA 中文指南](../../docs/libero-vla.zh-CN.md). An explicit `--agent-config` skips
benchmark policy overrides while retaining automatic local resource bindings.

Generic model names are placeholders; specialized presets retain their published
model policy for direct reproduction. Credentials come from the Agent environment;
do not put them in YAML. `source_root` is relative to the preset file.
[`../evaluations/embodied.yaml`](../evaluations/embodied.yaml) compares the two
API context choices through `--agent-matrix`.

See [Agents](../../docs/agents.md) / [Agent 配置](../../docs/agents.zh-CN.md)
for overrides and portable package export, and
[Evaluation](../../docs/evaluation.md) / [评测](../../docs/evaluation.zh-CN.md)
for execution and results.
