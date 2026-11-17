# SenseNova-RoboRSI-27B

**学会 Code as Policy，多轮 RSI 驱动 Agent 与 Harness 协同进化。**

[English](sensenova-roborsi-27b.md) · [🤗 Hugging Face](https://huggingface.co/sensenova/SenseNova-RoboRSI-27B) · [演示](#演示) · [评测结果](#评测结果) · [快速开始](#快速开始)

**发布草稿 · 模型权重：TBD · 技术报告：TBD**

SenseNova-RoboRSI-27B 是面向具身 Agent 的 **27B 视觉语言模型**。在 RoboRSI Harness 与 RoboRSI-27B-Preview 的基础上，模型从调用 VLA 技能进一步学习生成控制代码，计算目标位姿、组织机器人动作，并在统一的“观察—动作—反馈”闭环中结合 VLA 与代码控制。

模型通过**多轮递归自我改进（RSI）**持续学习：先改进 Harness，再采集、筛选交互经验，训练 Agent 适配更新后的执行系统。训练结合长程 Agent 轨迹与从已有 VLA 训练数据构造的局部动作监督。在具身后训练不使用真机数据的条件下，同一份模型权重也已用于真实机械臂的抓取与放置任务。

> **总览图 — TBD**<br>
> Agent 与 Harness 协同进化、VLA × Code as Policy，以及从仿真到真机的迁移。

## 核心亮点

- **VLA × Code as Policy。** 学习生成控制代码，并根据任务与执行状态，组织 VLA 技能与显式 TCP（工具中心点）控制。
- **多轮 RSI。** 迭代 Harness 的工具与执行策略，再通过监督微调和拒绝采样筛选的交互轨迹，使 Agent 适配持续演进的执行系统。
- **更准确的动作目标。** 将已有机器人轨迹构造成局部 TCP 监督，与长程决策轨迹联合训练，连接视觉观察、机器人状态与可执行的动作目标。
- **同权重 Sim-to-Real。** 具身后训练使用零真机数据，以同一份权重将仿真中学习到的能力迁移至真实机器人操作。

## 演示

### 真机操作

抓取鸭子并放入容器，或从容器内取出鸭子。这里将继续补充其他任务，以及与 Preview 的对比。

> **真机演示 — TBD**<br>
> 机器人配置、最终 checkpoint、任务指令与试验结果：TBD。

### 仿真中的 VLA × Code as Policy

通过一个完整任务展示 Agent 如何观察场景、选择执行方式、生成控制代码或调用 VLA 技能，并依据执行后的观察决定下一步动作。

> **RoboDojo 演示 — TBD**<br>
> 代表性的代码控制与 VLA 使用轨迹，配合关键决策和工具调用。

### 学习抓取、放置与插入

对比加入与未加入 TCP 监督时的局部操作行为，包括抓住并抬起目标、稳定放置，以及插入后松爪撤手。

> **TCP 监督对比演示 — TBD**<br>
> 抓取、放置与插入的并排片段，配合对应任务说明。

## 评测结果

### RoboDojo

正式版对比将采用统一评测协议，报告平均任务得分与任务成功率。

| 模型 | 平均得分 ↑ | 成功率 ↑ |
| :--- | ---: | ---: |
| **SenseNova-RoboRSI-27B** | **TBD** | **TBD** |
| SenseNova-RoboRSI-27B-Preview | TBD | TBD |
| Qwen3.8-27B | TBD | TBD |
| DeepSeek-V4.1-Flash-1017 | TBD | TBD |
| PEAR-27B | TBD | TBD |

任务集合、seed/layout、Harness 版本、执行后端、推理预算与聚合方式：**TBD**。

> **主结果图 — TBD**<br>
> 正式版最终结果与同协议下已完成的模型对比。

### TCP 目标预测

当前 TCP 监督研究的诊断集覆盖 **35 个任务、374 个局部动作**，包含 179 条单臂动作和 195 条双臂动作。当预测 TCP 终点与参考终点的位置误差落入阈值时，记为命中；双臂动作要求两臂均满足阈值。

| 训练设置 | 5 mm 阈值下 TCP 命中率 ↑ |
| :--- | ---: |
| 加入局部 TCP 监督 | **15.5%** |
| 未加入局部 TCP 监督 | 4.3% |

该研究中，5 mm 阈值下的命中率提升约 **3.6 倍**。对应模型版本与完整阈值结果：**TBD**。最终发布 checkpoint 的评测结果：**TBD**。

> **TCP 诊断图 — TBD**<br>
> 不同位置误差阈值下的命中率、训练消融与模型对比。

### 真机评测

| 任务 | 成功次数 / 总次数 | 执行配置 |
| :--- | :---: | :--- |
| 抓取鸭子并放入容器 | TBD | TBD |
| 从容器中取出鸭子 | TBD | TBD |
| 其他任务 | TBD | TBD |

机器人平台、相机与控制接口、试验条件、最终 checkpoint：**TBD**。

### 其他基准

最终发布 checkpoint 的 LIBERO-Pro 与 RoboCasa365 结果：**TBD**。

## 快速开始

### 1. 下载与部署

模型仓库：[`sensenova/SenseNova-RoboRSI-27B`](https://huggingface.co/sensenova/SenseNova-RoboRSI-27B)。

模型权重与经过验证的部署方案正在准备。发布包将包含权重，以及配套的模型配置、tokenizer、视觉 processor 与 chat template。

```bash
# 模型下载命令：TBD
# 推理环境与服务启动命令：TBD
# 视觉输入与工具调用请求示例：TBD
```

支持的推理后端及版本、硬件要求、上下文长度、生成参数与推荐推理强度：**TBD**。

### 2. 接入 RoboRSI Harness 并评测

通过 [RoboRSI Harness](https://github.com/OpenSenseNova/SenseNova-RoboRSI) 将模型连接到评测环境与机器人执行工具。正式版将提供各执行方式对应的配置与命令。

```bash
# RoboDojo — VLA + Code as Policy：TBD
# RoboDojo — 直接代码/TCP 控制：TBD
# LIBERO-Pro：TBD
# RoboCasa365：TBD
```

真机接入示例：**TBD**。

## 模型如何学习

### 多轮 Agent–Harness RSI

每轮首先改进 Harness，包括工具、执行策略、上下文与反馈处理。更新后的 Harness 产生交互轨迹，经拒绝采样筛选后用于监督微调。更新后的 Agent 再回到 Harness 中执行任务与评测。

**改进 Harness → 采集并筛选轨迹 → 训练 Agent → 联合评测 → 下一轮迭代。**

> **RSI 迭代曲线 — TBD**<br>
> 各轮成绩、模型与 Harness 版本，以及区分两者贡献的交叉对照。

### 学习何时行动、如何执行

VLA 技能提供习得的动作执行能力；Code as Policy 则允许 Agent 生成代码，计算目标位姿、组织运动，并调用 TCP 控制工具。两类方式通过同一交互闭环获得反馈。我们分析它们在不同任务与操作阶段中的使用情况，以及从基座模型、Preview 到正式版的行为变化。

> **执行方式分析 — TBD**<br>
> 按任务与阶段统计的 VLA/TCP 使用分布、代表性切换案例，以及组合执行与固定执行方式的对照。

### 长程轨迹与局部动作监督

长程 Agent 轨迹让模型学习连接观察、决策、工具调用与执行反馈；局部动作样本让模型学习预测单次运动的目标。

我们从已有 VLA 训练轨迹中识别局部操作片段，将动作指令、起始观察与片段终点 TCP 对齐。每条样本以三视角图像、双臂状态和局部指令为输入。后续画面用于核对片段标注，不进入模型输入。

当前数据构造覆盖 **35 个任务、3,493 个 episode**，共得到 **77,202 条局部样本**，其中单臂样本 66,359 条、双臂样本 10,843 条。最终入训子集、混合配方与数据划分将随正式版说明。

> **数据构造图 — TBD**<br>
> 一条真实的“观察＋机器人状态＋局部指令 → TCP 目标”样本，以及它与长程 Agent 训练的关系。

### 从仿真走向真实机器人

仿真与真机验证使用同一份模型权重，具身后训练不使用真机数据。视觉推理与代码控制将模型习得的任务决策连接到机器人的执行接口。

相机标定、动作坐标映射、机器人适配与完整执行样例：**TBD**。

## 复现材料

| 资源 | 发布版本 / 链接 |
| :--- | :--- |
| 模型 checkpoint / revision | TBD |
| 配套 Harness commit / tag | TBD |
| 任务清单与评测配置 | TBD |
| 逐任务结果与执行轨迹 | TBD |
| 技术报告 | TBD |

此前的版本见 [SenseNova-RoboRSI-27B-Preview](https://huggingface.co/sensenova/SenseNova-RoboRSI-27B-Preview)。

## Contributors

## 许可证

**Apache License 2.0。** 模型基于 [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)。权重发布时将附带上游许可证与署名文件。外部模型、仿真器和数据集遵循各自的许可证。
