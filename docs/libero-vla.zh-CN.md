# LIBERO VLA 自动下载与部署

[English](libero-vla.md) · [模型一键评测](model-evaluation.zh-CN.md)

保持主模型 API 事先部署好的流程。LIBERO Agent 启用 VLA 后，一键入口会额外准备
π0.5 执行模型：**构建 OpenPI 环境 → 下载权重 → 启动并预热 VLA → 评测 → 清理服务**。
主模型负责观察、规划和选择子任务，VLA 只根据当前图像与状态预测动作。

## 使用

先按[评测指南](model-evaluation.zh-CN.md)安装框架，配置主模型连接和密钥。可以在
[`configs/models/default.yaml`](../configs/models/default.yaml) 中填写主模型的地址、名称，
密钥继续通过 `OPENAI_API_KEY` 传递。然后：

```bash
python scripts/evaluation/evaluate_model.py --benchmark libero_pro \
  --agent-config configs/agents/libero/pro_vla.yaml --openai-format chat
```

首次会下载镜像依赖和模型权重。运行环境为 Linux/WSL、Docker 与 NVIDIA Container
Toolkit；VLA 推理需要 NVIDIA GPU。标准 LIBERO 的模拟器仍可使用 CPU，VLA 单独使用 GPU。
已有的 VLA-disabled 预设保持关闭，不会触发 OpenPI 构建或下载。

[`pro_vla.yaml`](../configs/agents/libero/pro_vla.yaml) 继承 PRO 预设并设置：

```yaml
parameters:
  enable_vla: true
```

标准 LIBERO 使用 [`standard.yaml`](../configs/agents/libero/standard.yaml)，PRO、Long++、
Ultra-Long 分别使用 `pro.yaml`、`long_plus_plus.yaml`、`ultra_long.yaml`。
这些 Agent 同样可以将 `enable_vla` 改为 `true`；Agent 不再引用单独的模型文件。
主模型和 VLA 统一配置在 [`configs/models/default.yaml`](../configs/models/default.yaml)，
或使用 `--model-config` 选择自己的 `model.yaml`：

```yaml
schema_version: 1
provider: openai
base_url: http://127.0.0.1:8000/v1
model_name: YOUR_MODEL
api_key_env: OPENAI_API_KEY
openai_format: chat
vla:
  backend: openpi
  endpoint: null  # 已有兼容 HTTP 地址；null 表示自动部署
  openpi_revision: 215abfb217dbac7d5f1273282331b9b1866c0479
  policy_config: pi05_libero
  checkpoint: gs://openpi-assets/checkpoints/pi05_libero
  gpus: all
  startup_timeout_s: 1800
  request_timeout_s: 120
  action_steps: 5
```

也可通过 benchmark 的 `agent_overrides.parameters`
启用；显式指定 `--agent-config` 时，以选择的 Agent 为准。

模型文件中的 `vla` 段设置 OpenPI 提交版本、`policy_config`、`checkpoint`、GPU、启动/请求超时
和 `action_steps`。默认使用官方 `pi05_libero` 检查点，OpenPI 代码固定到提交，
依赖使用其 `uv.lock`。官方 GCS 权重地址本身不是不可变版本；缓存使用上游 OpenPI
机制，记录的来源不能视为权重 SHA-256 锁定。需要固定自己的权重时提供受控本地目录。

```bash
# 只构建环境和下载模型/分词器，不加载 GPU 模型、不启动评测
python scripts/evaluation/evaluate_model.py --benchmark libero_pro \
  --agent-config configs/agents/libero/pro_vla.yaml --prepare-only

# 只查看准备计划，不下载、不启动容器
python scripts/evaluation/evaluate_model.py --benchmark libero_pro \
  --agent-config configs/agents/libero/pro_vla.yaml --dry-run \
  --base-url http://127.0.0.1:8000/v1 --model-name YOUR_MODEL

# VLA 选择 GPU 1；主模型 API 可部署在另一张卡或远端
ROBORSI_VLA_GPUS=device=1 python scripts/evaluation/evaluate_model.py \
  --benchmark libero_pro --agent-config configs/agents/libero/pro_vla.yaml --openai-format chat
```

`ROBORSI_LIBERO_VLA_CHECKPOINT=/absolute/path/to/checkpoint` 可覆盖 profile 中的权重。
本地路径以只读卷挂载；远端 GCS 地址自动下载。缓存默认位于
`work/model_eval/vla/openpi-cache/`，可通过 `--work-dir` 改变。
不向 Runner 安装 JAX、Torch 或 OpenPI；VLA 容器也不接收主模型 API 密钥。
本地相对权重路径以模型 YAML 所在目录为基准。VLA 权重和 tokenizer 下载统一由
[`scripts/evaluation/downloads.py`](../scripts/evaluation/downloads.py) 完成；部署入口
先检查本地缓存，再加载模型。Agent 启用 VLA 时，所选模型文件必须含 `vla` 段；
未启用 VLA 或使用原有 MCP 接法时可以省略。

## 独立指定三个配置

```bash
python scripts/evaluation/evaluate_model.py --benchmark libero_pro \
  --benchmark-config configs/benchmarks/examples/libero_pro_smoke.yaml \
  --agent-config configs/agents/libero/pro_vla.yaml \
  --model-config configs/models/default.yaml --openai-format chat
```

单个 benchmark 接受直接路径；多个 benchmark 使用 `--model-config libero_pro=PATH`
等写法，三个选项均可重复。命令行路径相对当前目录，YAML 内的引用相对 YAML 所在目录。
显式 Agent 保留自己的 API 格式、预算与策略；上例为 Chat Completions 服务显式选 `chat`。

测试集只改 `evaluation` 就会复用原有环境。例如：

```yaml
schema_version: 1
benchmark: libero_pro
evaluation:
  seeds: [0]
  selector: select_first.py:select_cases
  parameters:
    task_count: 1
```

也可填写 `evaluation.tasks: [TASK_ID]`，或在 Python 函数中定义筛选逻辑：

```python
def select_cases(cases, *, benchmark, parameters):
    # cases 是当前 catalog 经 tasks/seeds 筛选后的 task_id/seed 字典序列。
    tasks = list(dict.fromkeys(c['task_id'] for c in cases))[:parameters['task_count']]
    return [dict(c) for c in cases if c['task_id'] in tasks]
```

返回非空、无重复的已有 cell 列表。支持每个任务选择不同种子；不会重新展开成交叉组合。
函数在 Runner 中执行，是用户提供的 Python 代码；新增原生任务仍需 benchmark 自己提供
兼容 catalog。CLI 的 `--task`、`--seed` 优先于 YAML 同名筛选，随后执行 selector。

## 执行与已有服务

主模型会获得 `roborsi_action` 的 `primitive: vla` 动作，可给出短子任务指令。
Agent 读取当前 case 校验过的两路图像和 8 维状态，请求 VLA 后提交原生动作。
增强 LIBERO 每次最多执行 `action_steps` 个预测动作；标准 LIBERO 保持每次一个动作。
所有动作继续走 Runner 和 Benchmark 的原生步数、停止与评分，不存在 VLA 直接改环境的通道。
每个 case 的 `vla.jsonl` 记录指令、来源和实际动作。

同一配置在一次调用中共享 VLA 服务；退出、失败或中断时只清理本次启动的容器，缓存保留。
已有兼容 HTTP 服务可在模型 YAML 设置 `vla.endpoint: http://host:port`，
或用 `ROBORSI_LIBERO_VLA_URL=http://host:port` 覆盖，跳过 VLA 下载和部署。
这是本仓库的 HTTP `/v1/health`、`/v1/infer` 协议，**不是上游 WebSocket 地址**。
已有 `ROBORSI_VLA_MCP_URL`（或 `vla_url_env` 指定变量）时，继续走原有 MCP 接法，
不自动部署 OpenPI。两种服务地址均与主模型 `OPENAI_BASE_URL` 分开。
benchmark 使用已有 HTTP 服务时仍会准备本地 ICL、tokenizer 和所选 VLA。
增强 LIBERO 首次提取 ICL 可能需要拉取包含资源的固定镜像并使用 Docker，但不会启动
仿真环境。自动 VLA 地址通过运行时环境变量绑定，端口变化不会改变 Agent 标识，支持续跑。
清理容器前会保存服务日志。

上游参考：[OpenPI 权重与安装](https://github.com/Physical-Intelligence/openpi)、
[LIBERO 输入与执行示例](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/main.py)。
