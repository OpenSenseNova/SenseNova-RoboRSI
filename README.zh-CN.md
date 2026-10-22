<h1 align="center">SenseNova-RoboRSI：具身智能体的递归自我改进</h1>

<p align="center"><strong>通过统一运行框架，构建、评测并迭代具身 Agent。</strong></p>

<p align="center">
  <img src="assets/badges/python.svg" alt="Python 3.10–3.12" />
  <img src="assets/badges/license.svg" alt="License: Apache 2.0" />
  <img src="assets/badges/status.svg" alt="Status: Pre-Alpha" />
</p>

<p align="center">
  <a href="https://www.sensenova.ai/has-gpt-6-astra-achieved-embodied-intelligence"><img src="assets/badges/report.svg" alt="研究报告：Has GPT-6 Achieved Embodied Intelligence?" /></a>
  <a href="https://huggingface.co/sensenova/sensenova-roborsi-27b-preview"><img src="assets/badges/model.svg" alt="Model：SenseNova-RoboRSI-27B-Preview" /></a>
</p>

<p align="center">
  <a href="README.md">English</a> · <strong>简体中文</strong> ·
  <a href="#quick-start">快速开始</a> · <a href="docs/README.md">使用文档</a> ·
  <a href="https://www.sensenova.ai/has-gpt-6-astra-achieved-embodied-intelligence">研究报告</a> ·
  <a href="#benchzoo">BenchZoo</a> ·
  <a href="#harness-results">Harness 结果</a> ·
  <a href="#sensenova-roborsi-27b-preview">RoboRSI-27B-Preview</a> ·
  <a href="#release-plan--roadmap">发布计划</a>
</p>

## News

- **2026-10-22** — **SenseNova-RoboRSI-27B-Preview** 已在包括真机与仿真在内的不同环境中运行！[🤗 模型权重](https://huggingface.co/sensenova/sensenova-roborsi-27b-preview)与[使用指南](docs/sensenova-roborsi-27b-preview.zh-CN.md)现已发布，技术报告即将发布。

- **2026-10-22** — 本分支已提供 [**RoboCasa365 Target50**](benchmarks/robocasa365/README.zh-CN.md) 的首批代码与使用指南。

- **2026-10-21** — 已在 [BenchZoo](#benchzoo) 中补充 **benchmark 镜像与 ICL 经验包**，提供镜像地址和资源包下载链接。

- **2026-10-20** — 本分支已提供 [**BEHAVIOR-1K**](benchmarks/behavior_1k/README.zh-CN.md)、[**Bench2Dex**](benchmarks/bench2dex/README.zh-CN.md) 和 [**DexJoCo**](benchmarks/dexjoco/README.zh-CN.md) 的首批代码与使用指南。

- **2026-10-16** — 本分支已提供 [**LIBERO**](benchmarks/libero/README.md)、[**LIBERO-PRO**](benchmarks/libero_pro/README.md)、[**LIBERO-Long++**](benchmarks/libero_lilo/README.md) 和 [**LIBERO-Ultra-Long**](benchmarks/libero_lilo/README.md) 的首批代码与使用指南，详见[快速开始](docs/getting-started.zh-CN.md)。

- **2026-10-16** — SenseNova-RoboRSI 作为实验框架支持了实证研究 [**Has GPT-6 Achieved Embodied Intelligence?**](https://www.sensenova.ai/has-gpt-6-astra-achieved-embodied-intelligence)。

- **2026-10-09** — SenseNova-RoboRSI 在 RoboDojo 上取得 **60.08 的平均得分**和 **54.17% 的成功率**，平均得分较 GPT-6 Astra 基线（**28.97**）提升 **+31.11 分**。结果正在提交至 RoboDojo 官方榜单。更多评测对比见 [Harness 结果](#harness-results)。

## Contributors

- **Project leads:** Hanming Deng, Zhongang Cai, Haiwen Diao, Zhi Hou, Zhonghua Wu, Qingyi Tao
- **Model team:** Silei Wu\*, Weiye Xu, Bo Liu, Pengfei Liu, Tailai Chen, Chenjing Ding, Jiahao Wang, Changxu Wei, Zongpu Zhang
- **Harness team:** Jiapeng Luo\*, Yuyao Shao, Linjun Dai, Quanzhou Li, Ziyi Shang, Wanqi Yin, Xiaowan Li, Donghao Wu, Tianbo Zhong
- **Advisor:** Dahua Lin, Xiaogang Wang, Ziwei Liu, Lewei Lu, Lei Yang

\* 表示对应团队的 Leads。

## Overview

通过统一运行框架接入模型、技能与机器人工具，配置提示词和上下文，并在保留原生执行与
评分逻辑的环境中评测策略。

在此基础上，**递归自我改进（RSI）** 利用执行轨迹与任务反馈，在保持模型参数不变的前提下
优化 Harness。候选改进经过复测验证后，有效变化被继承到下一 Agent 版本，
将任务执行与持续改进连接起来。

🎉 **[SenseNova-RoboRSI-27B-Preview](#sensenova-roborsi-27b-preview)** 将这项工作进一步推进到模型训练。我们将经过验证的
交互经验转化为多模态监督，让目标理解、动作决策与反馈纠错逐步沉淀到模型参数中。
更新后的模型回到同一运行框架中复测并继续探索，让模型训练与 Harness 改进通过共享经验
和联合验证形成持续演进的闭环。

![SenseNova-RoboRSI 用户接口、Agent 执行闭环、Benchmark 接入与 RSI 迭代流程](assets/sensenova-roborsi-overview-zh.webp)

### 核心能力

#### Harness：通用运行框架

- **灵活的 Agent 接口**：接入模型、提示词、技能与工具/MCP，按任务配置上下文和缓存策略。
- **统一执行与评测**：跨基准调用 VLA 技能、脚本动作或末端控制工具，保留原生反馈与评分。
- **实验记录与训练数据**：记录完整轨迹、导出训练 episode，支持并发批量评测与中断恢复。

#### RSI：持续改进 Harness

- **从执行反馈中改进**：保持模型参数不变，优化动作预测、任务规划、执行反馈与记忆组织。
- **复测并保留有效变化**：记录 Agent 版本，在相同任务条件下评测候选改进，并将验证有效的变化继承到后续版本。

## Quick Start

Agent 和评测器使用 Python 3.10–3.12。在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[runtime,test]'
bench --help
```

Windows 使用 `.venv\Scripts\Activate.ps1` 激活环境。

按照[安装与首次评测指南](docs/getting-started.zh-CN.md)准备 benchmark，运行固定动作
或模型 Agent。原生 benchmark 镜像面向 Linux；连接已有 HTTP benchmark 服务时，
无需在本机安装模拟器。

- **Benchmark 配置：** [LIBERO](benchmarks/libero/README.md)、
  [LIBERO-PRO](benchmarks/libero_pro/README.md)、
  [LIBERO-Long++ / LIBERO-Ultra-Long](benchmarks/libero_lilo/README.md)、
  [Bench2Dex](benchmarks/bench2dex/README.zh-CN.md)、
  [DexJoCo](benchmarks/dexjoco/README.zh-CN.md)、
  [BEHAVIOR-1K](benchmarks/behavior_1k/README.zh-CN.md) 和
  [RoboCasa365](benchmarks/robocasa365/README.zh-CN.md)。
- **Agent 配置：** [模型、提示词、上下文与预算](docs/agents.zh-CN.md)。
- **评测：** [批量执行、结果查看与中断恢复](docs/evaluation.zh-CN.md)。

SenseNova-RoboRSI-27B-Preview 的部署方法见
[模型部署说明](docs/sensenova-roborsi-27b-preview.zh-CN.md#1-部署模型)。

### 一键评测已部署模型

完成框架安装后，即可接入支持图像输入与工具调用的已部署模型 API。
请使用已启动 Docker 的 Linux/WSL；增强 LIBERO 和 RoboCasa365 还需要 Docker GPU 支持。
先运行一个 RoboCasa 任务，检查下载、环境准备与评测的完整流程：

```bash
export OPENAI_BASE_URL='http://127.0.0.1:8000/v1'
export OPENAI_API_KEY='YOUR_API_KEY'
export MODEL_NAME='YOUR_SERVED_MODEL'
python scripts/evaluation/evaluate_model.py --benchmark robocasa365 \
  --task robocasa365:OpenDrawer --seed 1 --run-id robocasa-smoke-001
```

必须显式传入 `--benchmark`，可选 `libero`、`libero_pro`、`libero_long_plus_plus`、
`libero_ultra_long`、`robocasa365`，需要多项时重复传入。脚本自动下载/构建对应环境，
并准备可用的 ICL 包。增强 LIBERO 使用已发布的 `migration_subset` catalog。

配置统一放在 [`configs/`](configs/README.md)：每个 `configs/benchmarks/<name>.yaml`
选择环境、模型和 Agent。主模型连接和 `vla` 模型统一修改 [`configs/models/default.yaml`](configs/models/default.yaml)，
Agent 策略修改 `configs/agents/` 中的预设。上面的环境变量仍会覆盖模型 YAML，密钥保留在环境变量中。

可选 `--benchmark-config`、`--agent-config`、`--model-config` 直接指定配置地址；
同时选择多个 benchmark 时使用 `NAME=PATH`。benchmark 配置支持任务、种子列表及
[用户 Python 筛选器](configs/benchmarks/examples/select_first.py)。LIBERO Agent 设置
`parameters.enable_vla: true` 后会[自动下载、部署 π0.5 VLA](docs/libero-vla.zh-CN.md)，
仍可保持只传 `--benchmark NAME` 的入口。
benchmark YAML 的 `endpoint: http://host:18800` 可连接已有镜像的 HTTP 服务，
默认 `endpoint: null` 自动准备环境。HTTP 模式仍会准备 Agent 所需的 ICL、tokenizer 和 VLA。
benchmark 的 `token_env` 配置 HTTP 鉴权变量名；模型的 `vla.endpoint` 可连接已有 VLA 服务。

正式运行会先检查主模型图像/工具接口和已有 HTTP 服务，再下载资源。
任务、seed、完成数量和耗时实时显示，运行日志与脱敏配置会保存到结果目录。
可先离线查看计划，或在中断后用新的运行 ID 续跑：

```bash
python scripts/evaluation/evaluate_model.py --benchmark robocasa365 --dry-run
python scripts/evaluation/evaluate_model.py --benchmark robocasa365 \
  --run-id robocasa-retry --resume-from robocasa-001
```

结果概览见 `<results-root>/<run-id>/summary.json`；完整用法见[评测指南](docs/model-evaluation.zh-CN.md)。

单任务检查通过后，去掉任务和种子过滤，运行 RoboCasa 全部 340 个 cell：

```bash
ROBORSI_ROBOCASA_CELLS=all python scripts/evaluation/evaluate_model.py \
  --benchmark robocasa365 --run-id robocasa-001
```

通过环境变量选择标准 LIBERO suite，或 RoboCasa 的 50-cell seed-1 catalog：

```bash
ROBORSI_LIBERO_SUITE=libero_90 python scripts/evaluation/evaluate_model.py --benchmark libero
ROBORSI_ROBOCASA_CELLS=seed1 python scripts/evaluation/evaluate_model.py --benchmark robocasa365
```

默认 catalog 在 benchmark YAML 中配置：LIBERO 为 `libero_spatial`，RoboCasa 为 `all`；
`ROBORSI_LIBERO_SUITE`、`ROBORSI_ROBOCASA_CELLS` 可覆盖这些默认值。
结果保存在 `work/model_eval/results/`，每次运行请使用新的 `--run-id`。
只下载/构建环境与 ICL、不执行评测：

```bash
python scripts/evaluation/evaluate_model.py --benchmark robocasa365 --prepare-only
```

GPU/Docker 要求、镜像访问权限、任务范围、
已有服务接入与结果查看见[模型一键评测指南](docs/model-evaluation.zh-CN.md)。
需要先部署时，将已验证的命令填入预留的[模型部署入口](scripts/deploy/README.md)，
启动 API 服务后再评测。

## BenchZoo

列出已提供的 benchmark 接入与即将开源的内容。已支持的服务共用可配置的 EmbodiedAgent
预设；选择模型后，按对应评测目录完成环境准备与评测。

ICL 经验包统一托管在
[SenseNova-RoboRSI-Packages](https://huggingface.co/datasets/sensenova/SenseNova-RoboRSI-Packages)。
下表链接指向各已开源 benchmark 的版本化资源目录；下载、校验与使用方式见各资源包说明。

预构建镜像托管在 GHCR，使用 `docker pull <image>` 拉取表中对应镜像。

| Benchmark | 状态 | ICL 经验包 | Docker 镜像 |
| --- | --- | --- | --- |
| RoboDojo | 即将开源 | — | — |
| [RoboCasa365](benchmarks/robocasa365/) | 已开源 | [v1.0.0](https://huggingface.co/datasets/sensenova/SenseNova-RoboRSI-Packages/tree/main/robocasa365/v1.0.0) | [构建指南](benchmarks/robocasa365/BUILD.zh-CN.md) |
| [LIBERO-Pro](benchmarks/libero_pro/) | 已开源 | [v1.0.0](https://huggingface.co/datasets/sensenova/SenseNova-RoboRSI-Packages/tree/main/libero-pro/v1.0.0) | `ghcr.io/opensensenova/sensenova-roborsi-libero-pro@sha256:a83417f1459a4c555c457caa82ef00a5de50191e3d8deb943894fa9285cff172` |
| [LIBERO-Long++](benchmarks/libero_lilo/) | 已开源 | [v1.0.0](https://huggingface.co/datasets/sensenova/SenseNova-RoboRSI-Packages/tree/main/libero-long-plus-plus/v1.0.0) | `ghcr.io/opensensenova/sensenova-roborsi-libero-long-plus-plus@sha256:814139261e914b114cb89a359c6a39d4f409a556f79928e986b6ade3dfe7dc7a` |
| [LIBERO-Ultra-Long](benchmarks/libero_lilo/) | 已开源 | [v1.0.0](https://huggingface.co/datasets/sensenova/SenseNova-RoboRSI-Packages/tree/main/libero-ultra-long/v1.0.0) | `ghcr.io/opensensenova/sensenova-roborsi-libero-ultra-long@sha256:814139261e914b114cb89a359c6a39d4f409a556f79928e986b6ade3dfe7dc7a` |
| RoboLab | 即将开源 | — | — |
| [Bench2Dex](benchmarks/bench2dex/) | 已开源 | [v1.0.0](https://huggingface.co/datasets/sensenova/SenseNova-RoboRSI-Packages/tree/main/bench2dex/v1.0.0) | `ghcr.io/opensensenova/sensenova-roborsi-bench2dex@sha256:ba5253aa91e8e6888f0f0543626b9ace9c76dbe8559d7092caaa160e9b634a76` |
| [DexJoCo](benchmarks/dexjoco/) | 已开源 | [v1.0.0](https://huggingface.co/datasets/sensenova/SenseNova-RoboRSI-Packages/tree/main/dexjoco/v1.0.0) | `ghcr.io/opensensenova/sensenova-roborsi-dexjoco@sha256:19fa30540e3956ae460efa0a4b473586637cb330d863ea65ba0fd6318c376d4d` |
| LHVLN | 即将开源 | — | — |
| VLN-CE | 即将开源 | — | — |
| SIMPLE | 即将开源 | — | — |
| [BEHAVIOR-1K](benchmarks/behavior_1k/) | 已开源 | [v1.0.0](https://huggingface.co/datasets/sensenova/SenseNova-RoboRSI-Packages/tree/main/behavior/v1.0.0) | `ghcr.io/opensensenova/sensenova-roborsi-behavior-1k:v1.0.0-cu128` |

## Harness Results

每张表列出 SenseNova-RoboRSI 与部分参考结果，分数越高越好。

### [RoboDojo](https://robodojo-benchmark.com/)

SenseNova-RoboRSI 在 RoboDojo-Sim 上取得 **60.08 的平均得分**和 **54.17% 的成功率**。
相比 **GPT-6-Astra** 基线，平均得分提升 **31.11 分**，成功率提升 **31.69 个百分点**。

**榜单提交：** 我们的结果正在提交至 RoboDojo 官方榜单。官方结果可能略有差异；待官方结果公布后，我们将更新此处的结果。

| 方法 | 平均得分 ↑ | 成功率 ↑ |
| --- | ---: | ---: |
| SenseNova-RoboRSI / GPT-6 Astra（我们） | **60.08** | **54.17%** |
| EmbodiedEvo | 45.08 | 38.95% |
| VPP2 | 39.26 | 32.26% |
| PhysicalRSI | 36.27 | 31.38% |
| Awomo-0.5 | 35.34 | 29.64% |
| Simate-beta | 33.95 | 27.96% |
| GPT-6-Astra（基线） | 28.97 | 22.48% |

参考结果：[RoboDojo 仿真榜单](https://robodojo-benchmark.com/leaderboard)
（更新于 **2026-10-14**，访问于 **2026-10-15**）。

### [LIBERO-PRO](https://zxy-mllab.github.io/LIBERO-PRO-Webpage/)

| 方法 | 成功率 ↑ |
| --- | ---: |
| SenseNova-RoboRSI / GPT-6 Astra（我们） | **94.50%** |
| RPent / GPT-6 Astra | 92.63% |
| RPent / Opus-4.7 | 82.40% |
| RPent / GPT-5.5 | 75.13% |
| RPent Flash Mode / Molmo2-8B | 72.63% |

上表参考成绩由 [RPent](https://rpent.readthedocs.io/zh-cn/latest/rst_source/leaderboard/performance.html) 在 Task/Swap 套件上测得并公布，其中 GPT-6 Astra 为 92.63%。[LIBERO-PRO 官方榜单](https://github.com/Zxy-MLlab/LIBERO-PRO#-libero-pro-model-leaderboard) 汇总五类扰动，统计口径不同。

### [RoboCasa365](https://robocasa.ai/)

| 方法 | 评测设置 | 成功率 ↑ |
| --- | --- | ---: |
| SenseNova-RoboRSI / GPT-6 Astra（我们） | Target 厨房 + 任务记忆 | **64.60%** |
| Paimon-0 | 预训练厨房 | 58.10% |
| Xiaomi-Robotics-1 | 预训练厨房 | 57.40% |
| Phasor-m7 | 预训练厨房 | 56.80% |
| ProWAM | 预训练厨房 | 48.10% |

参考结果：[RoboCasa 官方榜单](https://robocasa.ai/leaderboard.html)。SenseNova-RoboRSI 使用 Target50 设置，官方榜单条目在预训练厨房中评测。以上为不同评测设置下的结果参考，不构成同一协议下的排名。

## SenseNova-RoboRSI-27B-Preview

**SenseNova-RoboRSI-27B-Preview** 是我们开源的 **27B 具身视觉语言模型**，将 RoboRSI
从 Harness 改进进一步拓展到模型训练。通过**模型与 Harness 协同 RSI**，模型从经过验证的
交互经验中学习，让目标理解、动作决策与反馈纠错逐步成为模型自身的能力。

模型支持 **VLA 配合执行与 VLM 直接控制**两种架构，已在 RoboDojo、LIBERO-Pro 和
RoboCasa365 上开展评测，并展示了仿真与真机任务能力。开放的模型权重为社区部署、复现和
继续训练提供起点。

![SenseNova-RoboRSI-27B-Preview 在 RoboDojo、LIBERO-Pro 与 RoboCasa365 上的评测结果](assets/roborsi-27b-preview-benchmarks.webp)

模型权重已在 [🤗 Hugging Face](https://huggingface.co/sensenova/sensenova-roborsi-27b-preview) 开放。
详细评测结果、演示与使用说明请参阅[模型指南](docs/sensenova-roborsi-27b-preview.zh-CN.md)。

## Release Plan & Roadmap

### 发布计划

- [x] 项目介绍
- [x] [GPT-6 Astra 具身能力实证调研](https://www.sensenova.ai/has-gpt-6-astra-achieved-embodied-intelligence)
- [x] 首批代码：Agent 运行时与 LIBERO 系列评测
- [x] [使用文档](docs/README.md)
- [x] [SenseNova-RoboRSI-27B-Preview 发布](docs/sensenova-roborsi-27b-preview.zh-CN.md)
- [ ] Harness demo
- [ ] RoboRSI 技术报告

### 后续研究方向

- **经验检索：** 计划将成功策略、失败案例与执行轨迹整理为可检索的经验库。
- **跨任务和跨环境迁移：** 将探索已验证的工具组合、规划策略与反馈机制，
  能否为其他任务与环境提供有效起点。
- **更多机器人平台与真实场景：** 希望将接入与评测扩展到更多机器人本体和物理场景，
  检验反馈驱动的持续改进在这些环境中的可靠性。
- **模型与 Harness 协同 RSI：** 持续将验证有效的交互经验转化为训练监督，联合复测模型与
  系统改进，并跟踪多轮迭代中的能力与效率收益。

## Community & Acknowledgements

欢迎基础模型、VLA、仿真平台与真实机器人相关团队参与合作。

- **参与贡献：** 欢迎新的接口接入、评测任务、可复现的失败案例，
  以及 Harness 设计建议。当前发布范围与检查要求见[贡献指南](CONTRIBUTING_zh.md)。
- **问题与合作：** 通过
  [Issues](https://github.com/OpenSenseNova/SenseNova-RoboRSI/issues)
  说明所用平台、应用场景与希望参与的方向。
- **发布进展：** 关注
  [SenseNova-RoboRSI 仓库](https://github.com/OpenSenseNova/SenseNova-RoboRSI)。
- **致谢：** 感谢
  [RoboDojo](https://robodojo-benchmark.com/)、
  [LIBERO-PRO](https://zxy-mllab.github.io/LIBERO-PRO-Webpage/)、
  [RoboCasa](https://robocasa.ai/)、
  [RPent](https://rpent.readthedocs.io/zh-cn/latest/) 和相关开源社区
  对具身 Agent 研究的支持。
- **许可证：** 外部模型、模拟器与数据集遵循各自的许可证。

**这一成果凝聚了联合团队在具身智能与 Agent 系统方向上的持续探索和紧密协作。SenseNova-RoboRSI 已用于现已发布的实证研究 [Has GPT-6 Achieved Embodied Intelligence?](https://www.sensenova.ai/has-gpt-6-astra-achieved-embodied-intelligence)。未来，我们将继续推动相关技术的迭代与验证。**
