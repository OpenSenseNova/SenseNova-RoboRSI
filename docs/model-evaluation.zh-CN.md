# 一键评测已部署模型

[English](model-evaluation.md) · [全部指南](README.md)

[`scripts/evaluation/evaluate_model.py`](../scripts/evaluation/evaluate_model.py)
将已部署的模型 API 接入指定 benchmark：自动下载或构建环境、准备并校验可用的 ICL 包、
启动 Benchmark 服务、执行评测、保存结果，最后清理本次启动的容器。下载资源会缓存复用。

```text
部署脚本 → 模型 API 已启动
    → 配置 base_url、api_key、model_name
    → evaluate_model.py --benchmark NAME
    → 下载/构建环境和 ICL → 评测 → 保存结果
```

部署命令放在 [`scripts/deploy/`](../scripts/deploy/README.md)，评测入口与下载逻辑放在
`scripts/evaluation/`，各 benchmark 的原生环境与协议保留在 `benchmarks/`。

## 安装框架并启动模型

使用 **Linux/WSL、Python 3.10–3.12，以及已启动的 Docker**。LIBERO 增强榜单与
RoboCasa365 还需要 NVIDIA GPU 和 Docker GPU 支持；标准 LIBERO 使用 CPU 渲染。
在本仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[runtime]'
```

将已验证的部署命令填入
[`scripts/deploy/deploy_model.sh`](../scripts/deploy/deploy_model.sh)，在另一终端启动：

```bash
bash scripts/deploy/deploy_model.sh
```

此文件是预留入口，未填写时会报错退出。服务已运行时可以直接使用。
模型需通过 OpenAI 兼容接口支持**图像输入与工具调用**。自动预设使用 Chat Completions，
关闭推理参数与供应商专有的 prompt-cache 扩展；支持 Responses 的服务可传
`--openai-format responses`。

## 统一配置环境、模型和 Agent

入口自动读取 `configs/benchmarks/<benchmark>.yaml`，从中选择环境并引用模型与 Agent
预设，无需额外配置参数。目录说明见 [`configs/README.md`](../configs/README.md)。

```yaml
# configs/benchmarks/libero_pro.yaml（引用部分节选）
schema_version: 1
benchmark: libero_pro
model: ../models/default.yaml
agent: ../agents/libero/pro.yaml
# 同一文件内还定义 environment 和 agent_overrides
```

共享模型连接在 [`configs/models/default.yaml`](../configs/models/default.yaml) 配置：

```yaml
schema_version: 1
provider: openai
base_url: http://127.0.0.1:8000/v1
model_name: YOUR_SERVED_MODEL
api_key_env: OPENAI_API_KEY
openai_format: chat
```

模型配置优先级为 **命令行 > 模型配置指定的环境变量 > YAML**。仓库默认文件的地址、
名称留空，方便继续使用环境变量；`openai_format: null` 保留 Agent 格式。
不同 benchmark 可以引用不同模型配置，评测子进程分别接收各自的密钥，统一映射为
`OPENAI_API_KEY`，不会把密钥写入生成配置。

镜像、构建参数、资源下载版本和 catalog 默认值放在 benchmark YAML。
Agent 策略统一放在 `configs/agents/`，LIBERO 预设迁至
`libero/{base,v13,pro,long_plus_plus,ultra_long}.yaml`，context map 也放在该目录。
benchmark 的 `agent_overrides` 显式覆盖一键模式所需的 API 兼容设置，再自动绑定本地
ICL/tokenizer 路径，生成 `<work-dir>/presets/<benchmark>.json`。
请修改源 YAML，生成文件会被重新写入；命令行 Agent 参数优先。
`configs/evaluations/embodied.yaml` 继续用于已有的多 Agent 对比。

也可以直接在启动命令中分别选择三个配置：

```bash
python scripts/evaluation/evaluate_model.py --benchmark libero_pro \
  --benchmark-config configs/benchmarks/examples/libero_pro_smoke.yaml \
  --agent-config configs/agents/libero/pro_vla.yaml \
  --model-config configs/models/default.yaml --openai-format chat
```

单个 benchmark 支持直接路径；多个时重复使用 `NAME=PATH`。精简 benchmark YAML
自动继承现有环境，`evaluation.tasks`、`evaluation.seeds` 选择测试集，
`evaluation.selector: path.py:function` 支持自己的 Python 筛选代码。
CLI 的 `--task`、`--seed` 优先于对应 YAML 列表，然后运行 selector，返回已有 catalog
中的非空、无重复 cell。每个任务可以选择不同种子，不会扩展成其他组合。
详见 [示例配置](../configs/benchmarks/examples/libero_pro_smoke.yaml)和
[Python 示例](../configs/benchmarks/examples/select_first.py)。

LIBERO Agent 设置 `parameters.enable_vla: true` 后，还会自动构建独立 OpenPI
环境、下载 π0.5 权重、启动并预热服务；`--prepare-only` 只准备资源。
配置、GPU 选择与已有服务接法见 [LIBERO VLA 指南](libero-vla.zh-CN.md)。

## 一键评测

```bash
export OPENAI_BASE_URL='http://127.0.0.1:8000/v1'
export OPENAI_API_KEY='YOUR_API_KEY'
export MODEL_NAME='YOUR_SERVED_MODEL'

python scripts/evaluation/evaluate_model.py --benchmark libero_pro
```

**必须显式指定 `--benchmark`，不会默认运行全部榜单。** 多项评测可重复传入。
`--base-url`、`--api-key`、`--model-name` 可覆盖对应环境变量。
建议通过环境变量传密钥，避免写入命令行历史。无鉴权服务设置 `OPENAI_API_KEY=EMPTY`。
base URL 通常包含 `/v1`，model name 使用服务实际提供的模型 ID。

| `--benchmark` | 自动准备内容与任务范围 |
| --- | --- |
| `libero` | 下载固定版本资产，按当前 checkout 构建服务。测试下方指定的一个 suite；暂无已发布的标准 LIBERO ICL 包。 |
| `libero_pro` | 拉取固定 digest 镜像，提取并校验封存 ICL。`migration_subset`：80 个 seed-0 case，40 个 ICL 包。 |
| `libero_long_plus_plus` | 拉取固定 digest 镜像，提取并校验封存 ICL。`migration_subset`：12 个 seed-0 case/包。 |
| `libero_ultra_long` | 拉取固定 digest 镜像，提取并校验封存 ICL。`migration_subset`：9 个 seed-0 case/包。 |
| `robocasa365` | 基于固定基础镜像按当前 checkout 构建服务，含厨房资产和 RLDX-1 权重；下载固定版本 Target50 ICL。默认 340 个 cell。 |

三项增强 LIBERO 使用本 checkout 的[合并交付入口](../benchmarks/libero_three_combined/README.md)。
自动准备会在固定的原生镜像上覆盖构建当前 checkout 的 Benchmark 服务代码，避免依赖发布镜像的旧入口布局。
自动 catalog 暴露的是 **migration 子集，并非完整官方榜单任务集**，封存 context map 只覆盖
这些子集。需要其他或完整协议时，按 benchmark 指南准备对应 catalog 和兼容 Agent 预设，
再使用下方已有环境模式。仅替换模型 API 不代表复现已发布 Harness 成绩。

RoboCasa ICL 只注入**同一任务 source seed 0 的经验**，覆盖 50 个任务中的 43 个，
对应 340 个评测 cell 中的 305 个；明确声明无记忆的 7 个任务以空 ICL 运行。
每个 case 写入 `icl-evidence.json`，不使用全局或跨任务记忆。
RLDX-1 是原生动作执行模型，与被评测模型 API 独立，由环境自动启动。

首次运行会下载较大的镜像并安装构建依赖。Docker 缓存镜像；资产、ICL 和生成的预设默认放在
`work/model_eval/`，可用 `--work-dir /path/to/cache` 修改。复用缓存时会校验，
改动或不完整的缓存会报错，请更换 work directory 或移走对应缓存后重试。
LIBERO ICL 从固定镜像提取，按 `assets.lock.json` 校验；RoboCasa 使用固定 Hugging Face
commit 和 SHA-256 manifest。增强 LIBERO 所需的 `tokenizers==0.22.2` 会按需安装到当前虚拟环境。

Docker、GPU 驱动需要事先安装。镜像使用 Docker 登录凭据；固定的 RoboCasa 基础镜像
当前需要 GHCR 访问权限，出现 `denied` 时请使用有读取权限的账号登录。
Hugging Face 需要鉴权时使用 `HF_TOKEN` 或已有 Hub 登录状态，不使用模型 API 密钥。
参见 [RoboCasa 构建说明](../benchmarks/robocasa365/BUILD.zh-CN.md)。

## 用环境变量选择 catalog

以下环境变量覆盖 benchmark YAML 的 `environment.catalog`，只作用于本脚本自动启动的 Benchmark 服务：

| 环境变量 | 默认值 | 可选值 |
| --- | --- | --- |
| `ROBORSI_LIBERO_SUITE` | `libero_spatial` | `libero_spatial`、`libero_object`、`libero_goal`、`libero_10`、`libero_90` |
| `ROBORSI_ROBOCASA_CELLS` | `all` | `all`：340 个 cell；`seed1`：50 个 seed-1 cell |

```bash
ROBORSI_LIBERO_SUITE=libero_90 python scripts/evaluation/evaluate_model.py --benchmark libero
ROBORSI_ROBOCASA_CELLS=seed1 python scripts/evaluation/evaluate_model.py --benchmark robocasa365
```

这些变量不修改外部 registry 或已有服务的 catalog。入口不提供 `--libero-suite`
和 `--robocasa-cells` 参数。

## 只准备、查看计划、单任务检查

只下载/构建指定 benchmark 所需环境与 ICL，不请求模型、不启动 episode，无需模型连接配置：

```bash
python scripts/evaluation/evaluate_model.py --benchmark robocasa365 --prepare-only
```

不传外部 source 时，`--dry-run` 只打印准备计划，不下载、不启动容器、不请求 catalog
或模型，也不验证模型是否支持图像与工具调用：

```bash
python scripts/evaluation/evaluate_model.py --benchmark libero_pro --dry-run
```

实际运行一个 catalog 中存在的任务与种子：

```bash
python scripts/evaluation/evaluate_model.py --benchmark robocasa365 \
  --task robocasa365:OpenDrawer --seed 1 --run-id smoke-001
```

省略 `--task` 和 `--seed` 时，运行所选 catalog 的任务与种子。
单任务检查通过后，使用新的 run ID 运行 RoboCasa 全部 340 个 cell：

```bash
ROBORSI_ROBOCASA_CELLS=all python scripts/evaluation/evaluate_model.py \
  --benchmark robocasa365 --run-id robocasa-001
```

命令前的 `all` 也会覆盖之前设置的 `ROBORSI_ROBOCASA_CELLS=seed1`。
原生 Windows 仅支持 `--help` 及自动/HTTP 模式的 `--dry-run`，准备与执行评测请使用 Linux/WSL。

## 已有环境和自定义 Agent

传入一种外部 source 后，跳过自动环境与 ICL 准备：

```bash
python scripts/evaluation/evaluate_model.py --benchmark libero \
  --registry-root /absolute/path/to/registry

python scripts/evaluation/evaluate_model.py --benchmark robocasa365 \
  --benchmark-url robocasa365=http://127.0.0.1:18800
```

多项 HTTP benchmark 需为每个所选名称提供 `--benchmark-url NAME=URL`。
这是**模拟器服务地址**，与模型的 `OPENAI_BASE_URL` 不同。Registry 模式启动已准备的服务；
HTTP 模式连接自行管理的服务。这两种外部 source 模式的 `--dry-run` 会读取、校验 catalog
与计划，不启动 episode、不请求模型。HTTP 服务需要鉴权时，用
`--benchmark-token-env ENV_NAME` 指定 token 所在环境变量。

`--agent-config NAME=PATH` 可替换自动 ICL 预设。外部 source 模式默认使用
`configs/agents/embodied/default.yaml`，不会自动配置 ICL。指定 `--benchmark-config`
时会采用该配置选择的 Agent；启用 VLA 的显式 Agent 在已有环境模式下也会自动准备 VLA。
自定义预设需已具备代码、依赖与上下文资源；脚本覆盖模型连接，其余格式、上下文与预算保留，
除非显式传参覆盖。

完整参数见 `python scripts/evaluation/evaluate_model.py --help`。
`--max-turns`、`--max-tokens`、`--keep-turns` 可覆盖 Agent 预算。增强 LIBERO 保留对应预设
的上下文和较大输出预算，请按服务模型的能力配置。标准 LIBERO 通用预设默认 100 轮、
2048 输出 token、保留 8 轮。RoboCasa 保留 catalog 的原生 planner 预算：atomic 1800 秒、
composite 3600 秒。`--timeout-s` 为单次传输调用限制，默认 1800 秒。
完整 case 的 `--case-timeout-s` 默认增强 LIBERO 43200 秒，其余 4500 秒。
自动/HTTP 模式要求 `--max-concurrent 1`。

## 查看结果

结果默认写入 `<work-dir>/results/`，可用 `--results-root` 修改。
例如 `--run-id my-model-001` 仅创建所选且非空的组：

```text
work/model_eval/results/
  my-model-001-libero_pro/
  my-model-001-robocasa365-atomic/
  my-model-001-robocasa365-composite/
```

每组保留标准 batch plan、逐 case 结果、原生评分产物与 trace。每次请使用新的 `--run-id`。
使用 `bench status` 查看具体组；RoboCasa 可将两个结果目录传给
`python -m benchmarks.robocasa365.score` 评分，完整 50-cell seed-1 运行需追加
`--cells seed1`。单任务检查不代表完整榜单分数。
详见[评测指南](evaluation.zh-CN.md)与 [RoboCasa365](../benchmarks/robocasa365/README.zh-CN.md)。
