# 一键评测已部署模型

[English](model-evaluation.md) · [全部指南](README.md)

[`scripts/evaluation/evaluate_model.py`](../scripts/evaluation/evaluate_model.py)
将已部署的模型 API 接入现有评测器，流程如下：

```text
一次性准备框架与 benchmark 环境
    → 使用部署脚本启动模型 API 服务
    → 配置 base_url、api_key、model_name
    → 执行 evaluate_model.py 并查看结果
```

模型部署入口预留在 [`scripts/deploy/`](../scripts/deploy/README.md)，评测入口放在
`scripts/evaluation/`。各 benchmark 的镜像、资产和原生协议继续由 `benchmarks/` 管理。

## 1. 准备环境与 benchmark

使用 Python 3.10–3.12。Registry 和 HTTP 两种模式的实际评测都需要 **Linux/WSL Runner**。
原生 Windows Python 可以查看 `--help` 和执行 HTTP `--dry-run`，不能执行评测 case。
在 Linux/WSL 的仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[runtime,test]'
```

然后按对应 benchmark 指南准备镜像、资产和 registry，或启动独立的 Benchmark HTTP
服务。脚本支持下列五项；不传 `--benchmark` 时默认全部运行。

| Benchmark | `--benchmark` 参数值 | 准备指南 |
| --- | --- | --- |
| LIBERO | `libero` | [LIBERO](../benchmarks/libero/README.md) |
| LIBERO-PRO | `libero_pro` | [LIBERO-PRO](../benchmarks/libero_pro/README.md) |
| LIBERO-Long++ | `libero_long_plus_plus` | [Long++ / Ultra-Long](../benchmarks/libero_lilo/README.md) |
| LIBERO-Ultra-Long | `libero_ultra_long` | [Long++ / Ultra-Long](../benchmarks/libero_lilo/README.md) |
| RoboCasa365 Target50 | `robocasa365` | [RoboCasa365](../benchmarks/robocasa365/README.zh-CN.md) |

分支专用接入以当前本地 `LIBERO` 分支为准；RoboCasa365 以
[远端 `main` 指南](https://github.com/OpenSenseNova/SenseNova-RoboRSI/blob/main/benchmarks/robocasa365/README.zh-CN.md)
为准。本地 LIBERO 入口仍可能随正在进行的修改而更新。

两种运行方式选其一：

- **Registry 模式：** `--registry-root /absolute/path/to/registry`。registry 必须包含
  所有选中的 benchmark，评测器负责启动对应 Docker 服务。此模式（包括 `--dry-run`）
  需要在 Linux/WSL 中运行。原生 benchmark 镜像面向 Linux，硬件及 Docker 要求见各自指南。
- **已有 HTTP 服务：** 为每个选中的 benchmark 重复传入 `--benchmark-url NAME=URL`。
  评测器连接已有服务，本机无需安装模拟器，实际执行仍需 Linux/WSL。原生 Windows 使用
  `.venv\Scripts\Activate.ps1` 激活环境后，只能在此模式下运行 `--dry-run`。

实际任务、种子、profile 和物理动作预算以 manifest 或 HTTP catalog 为准。catalog
只包含子集时，脚本也只评测该子集。需要完整榜单协议时，应按对应 benchmark 指南准备
完整 catalog 和 official profile。该通用 API 接入流程本身不等于复现已发布的模型或
Harness 成绩。

RoboCasa365 还需要原生资产和独立的 **RLDX-1 执行服务**。其镜像可以自动启动这个进程，
也可以连接已有进程；待评测模型的 API 不会替代 RLDX-1。详见
[RoboCasa365 指南](../benchmarks/robocasa365/README.zh-CN.md)。

## 2. 启动模型 API

已有模型服务可跳过此步。否则，先在
[`scripts/deploy/deploy_model.sh`](../scripts/deploy/deploy_model.sh)
中填入已经验证的部署命令，再启动：

```bash
bash scripts/deploy/deploy_model.sh
```

该文件是明确的占位入口，未填写时会报错退出。仓库没有假设模型权重、部署框架和 GPU
配置；部署命令需要提供的接口见[部署说明](../scripts/deploy/README.md)。

模型必须通过 OpenAI 兼容 API 支持**图像输入和工具调用**。通用预设使用 Chat Completions；
服务实现 Responses API 时可传 `--openai-format responses`。模型名称必须与部署服务实际
提供的 model ID 一致。

## 3. 配置模型并执行

在评测终端设置三个连接参数。建议通过环境变量提供 API key，避免它出现在命令行参数中：

```bash
export OPENAI_BASE_URL='http://127.0.0.1:8000/v1'
export OPENAI_API_KEY='YOUR_API_KEY'
export MODEL_NAME='YOUR_SERVED_MODEL'
```

如果服务关闭了鉴权，可以设置非空占位值，例如 `OPENAI_API_KEY=EMPTY`。模型 base URL
通常包含 `/v1`，不要填写完整的 `/chat/completions` 路径，具体以部署服务为准。

| 地址 | 用途 | 示例 |
| --- | --- | --- |
| `--base-url` / `OPENAI_BASE_URL` | 待评测模型的 API | `http://127.0.0.1:8000/v1` |
| `--benchmark-url NAME=URL` | 模拟器的 Benchmark 服务 | `robocasa365=http://127.0.0.1:18800` |

先用已准备好的 catalog 检查单个 LIBERO 任务：

```bash
python scripts/evaluation/evaluate_model.py \
  --registry-root /absolute/path/to/registry \
  --benchmark libero --task libero:0 --seed 0 --dry-run
```

`--dry-run` 会验证并打印评测计划，不请求模型、不启动 Docker、不生成 case 结果。HTTP 模式仍
需要访问远端 catalog。它不能确认模型的 API、图像输入或工具调用是否兼容。去掉
`--dry-run` 即可执行该单任务连通性检查。

运行全部五项 benchmark 在 catalog 中声明的所有任务和种子：

```bash
python scripts/evaluation/evaluate_model.py \
  --registry-root /absolute/path/to/registry --run-id my-model-001
```

只运行 LIBERO-PRO 时添加 `--benchmark libero_pro`，选择多项时重复传入 `--benchmark`。
省略 `--task` 和 `--seed` 会采用各自 catalog 的默认值；LIBERO 与 RoboCasa365 暴露的种子
范围可能不同。

如果使用已启动的 HTTP 服务，明确选择所提供的 benchmark：

```bash
python scripts/evaluation/evaluate_model.py \
  --benchmark libero --benchmark robocasa365 \
  --benchmark-url libero=http://127.0.0.1:18000 \
  --benchmark-url robocasa365=http://127.0.0.1:18800
```

上面的端口仅为示例，请填写实际服务地址。RoboCasa365 的单任务检查可选择
`--benchmark robocasa365`，添加 `--task robocasa365:OpenDrawer --seed 1`，并提供对应的
registry 或 HTTP 地址。

原生 Windows PowerShell 中可按下面的方式检查已有 HTTP catalog，实际评测请在 Linux/WSL
中运行：

```powershell
$env:OPENAI_BASE_URL = 'http://127.0.0.1:8000/v1'
$env:OPENAI_API_KEY = 'YOUR_API_KEY'
$env:MODEL_NAME = 'YOUR_SERVED_MODEL'
python scripts/evaluation/evaluate_model.py --benchmark robocasa365 --benchmark-url robocasa365=http://127.0.0.1:18800 --task robocasa365:OpenDrawer --seed 1 --dry-run
```

## 常用参数

完整帮助可运行 `python scripts/evaluation/evaluate_model.py --help`。

| 参数 | 默认值 / 行为 |
| --- | --- |
| `--base-url`、`--api-key`、`--model-name` | 分别覆盖 `OPENAI_BASE_URL`、`OPENAI_API_KEY`、`MODEL_NAME` 环境变量 |
| `--openai-format` | 通用预设为 `chat`，也支持 `responses`；显式传入时覆盖自定义预设 |
| `--registry-root` / `--benchmark-url NAME=URL` | 必须选择一种运行方式，两者互斥 |
| `--benchmark NAME` | 可重复；默认运行全部五项 |
| `--agent-config NAME=PATH` | 可重复；为指定 benchmark 选择专用 Agent 预设 |
| `--task NAME:ID`、`--seed N` | 可重复筛选；不传时使用 catalog |
| `--benchmark-token-env ENV` | 保存已有 Benchmark 服务鉴权令牌的环境变量名称 |
| `--max-concurrent` | `1`；HTTP 模式必须为 `1` |
| `--max-turns`、`--max-tokens`、`--keep-turns` | 通用预设为 `100`、`2048`、`8`；显式传入时覆盖自定义预设 |
| `--timeout-s` | 单次传输调用 `1800` 秒 |
| `--case-timeout-s` | LIBERO-PRO / Long++ / Ultra-Long 为 `43200` 秒，LIBERO / RoboCasa365 为 `4500` 秒；显式传入时覆盖默认值 |
| `--results-root` | 仓库下的 `work/model_eval/results` |
| `--run-id` | 不传时自动生成；每次评测使用新 ID |
| `--dry-run` | 验证并打印计划，不执行 case |

RoboCasa365 按 catalog 中的 planner 预算自动分组：**atomic 任务 1800 秒、composite
任务 3600 秒**。传输超时与完整 case 超时是另外两个限制；脚本会在执行两组任务时保留
这些原生 planner 预算。

## 使用 benchmark 专用 Agent 预设

需要特定 Harness 时，可重复传入 `--agent-config NAME=PATH` 为对应 benchmark 选择预设。
脚本将模型连接指向你的 OpenAI 兼容服务；API 格式、上下文和预算保留预设配置，只有显式
传入对应 CLI 参数时才覆盖。RoboCasa365 的 planner 预算仍按原生 catalog 设置。

本地 `LIBERO` 分支提供了 PRO、Long++、Ultra-Long 的专用预设和 sealed ICL 配置，请结合
该分支当前评测入口阅读其中的 `docs/libero-three-benchmarks.zh-CN.md`。这些预设要求 checkout
或已安装的 Agent 环境包含其**完整运行时代码、依赖和上下文资产**；仅把当前分支指向外部
YAML 文件不会安装缺失模块。选择其预设前须按对应分支完成环境与协议准备，这里的通用
默认配置不等于该分支的 sealed ICL 复现配置。

## 查看结果与评分

使用 `--run-id my-model-001` 时，各组结果分别保存到：

```text
work/model_eval/results/
  my-model-001-libero/
  my-model-001-libero_pro/
  my-model-001-libero_long_plus_plus/
  my-model-001-libero_ultra_long/
  my-model-001-robocasa365-atomic/
  my-model-001-robocasa365-composite/
```

只会创建选中且非空的组。各组沿用 `bench eval` 的输出结构，包括 `batch_plan.json`、
各 case 的 `result.json`、原生评分产物和轨迹。查询某一组状态：

```bash
bench status --results-root work/model_eval/results --run-id my-model-001-libero
```

RoboCasa365 评分时，把两个已经生成的分组目录一起传给 scorer：

```bash
python -m benchmarks.robocasa365.score \
  work/model_eval/results/my-model-001-robocasa365-atomic \
  work/model_eval/results/my-model-001-robocasa365-composite
```

默认评分协议包含 **340 个 cell**，总分是 50 个任务各自成功率的平均值。如果准备并运行的
是 seed-1 协议，添加 `--cells seed1`；其分母仍是**全部 50 个 seed-1 cell**。缺失或无效
cell 计为失败，因此单任务检查不代表完整 benchmark 成绩。case 状态、轨迹和结果比较见
[评测指南](evaluation.zh-CN.md)。
