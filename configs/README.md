# Evaluation configuration / 评测配置

`evaluate_model.py --benchmark NAME` reads `benchmarks/NAME.yaml` and follows
its model and Agent references. All reference paths are relative to the YAML
that declares them. The command works from any current directory.

`evaluate_model.py --benchmark NAME` 自动读取 `benchmarks/NAME.yaml`，组合其中引用的
模型与 Agent 配置。引用路径以声明它的 YAML 所在目录为基准。

```text
configs/
  benchmarks/       # Environment, downloads, catalog and model/Agent selection
  models/           # Shared model connection profiles
  agents/
    embodied/       # Generic Agent presets
    libero/         # LIBERO presets and context_maps/
    robocasa365/    # RoboCasa same-task ICL Agent
  evaluations/      # Existing multi-Agent comparison matrices
```

| Change / 修改内容 | File / 文件 |
| --- | --- |
| Model endpoint, name and API format / 模型地址、名称、API 格式 | [models/default.yaml](models/default.yaml) |
| Native image, downloads, default catalog / 原生镜像、下载源、默认 catalog | `benchmarks/<benchmark>.yaml` |
| Prompt, budgets, context and factory / 提示词、预算、上下文与 factory | `agents/` 中对应 YAML |
| Per-benchmark Agent adjustments / benchmark 对 Agent 的覆盖 | benchmark YAML 的 `agent_overrides` |
| Comparison matrix / 多 Agent 对比 | [evaluations/embodied.yaml](evaluations/embodied.yaml) |

Model priority is **CLI > named environment variable > model YAML**. Set
`base_url` and `model_name` in the model file, or use `OPENAI_BASE_URL` and
`MODEL_NAME`. `api_key_env` names the variable containing the key (default:
`OPENAI_API_KEY`); inline keys are rejected. `openai_format: null` preserves
the Agent's format; set `chat` or `responses` to select one for that model.

模型优先级为 **命令行 > 指定环境变量 > 模型 YAML**。可以直接填写 `base_url`、
`model_name`，或继续使用 `OPENAI_BASE_URL`、`MODEL_NAME`。
`api_key_env` 只填写密钥环境变量名，默认 `OPENAI_API_KEY`；不接受 YAML 明文密钥。
`openai_format: null` 保留 Agent 格式，填写 `chat` 或 `responses` 可为模型指定格式。

For managed environments, `ROBORSI_LIBERO_SUITE` and `ROBORSI_ROBOCASA_CELLS`
override `environment.catalog.suite` and `.cells`. With external HTTP/registry
sources, their catalogs remain authoritative and automatic ICL setup is skipped;
use `--agent-config` for a prepared custom Agent.

自动环境中，`ROBORSI_LIBERO_SUITE` 与 `ROBORSI_ROBOCASA_CELLS` 分别覆盖
`environment.catalog.suite`、`.cells`。已有 HTTP/registry 模式以外部 catalog 为准，
跳过自动 ICL 准备，可用 `--agent-config` 指定已经准备好的 Agent。

The automatic Agent is composed in this order: referenced Agent YAML (including
`extends`), benchmark `agent_overrides`, generated local resource bindings, then
model/CLI overrides. Generated files live under `<work-dir>/presets/` and are
rewritten on preparation; edit source YAML instead. The released LIBERO presets
retain their original policy; the benchmark YAML applies the automatic API
compatibility settings explicitly.

自动 Agent 按以下顺序组合：引用的 Agent YAML（含 `extends`）→ benchmark
`agent_overrides` → 自动生成的本地资源路径 → 模型/命令行覆盖。
`<work-dir>/presets/` 中的文件会在准备时重新生成，请修改源 YAML。
LIBERO 发布预设保留原始策略，一键模式的 API 兼容配置显式放在 benchmark YAML 中。

Dockerfiles, manifest generators, asset locks and scoring code remain under
`benchmarks/`; implementation modules remain under `robots/`. Model deployment
remains a separate step in [`scripts/deploy/`](../scripts/deploy/README.md).

See [model evaluation](../docs/model-evaluation.md) /
[模型一键评测](../docs/model-evaluation.zh-CN.md) for commands and task coverage.
