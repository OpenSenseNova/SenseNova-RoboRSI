# SenseNova-RoboRSI-27B｜通过模型与 Harness 的协同演进融合 VLA 与 Code as Policy

[English](./sensenova-roborsi-27b.md) · [模型权重](https://huggingface.co/sensenova/SenseNova-RoboRSI-27B) · [GitHub](../README.zh-CN.md) · [评测与演示](#评测与演示) · [快速开始](#快速开始)

在我们此前的真机插花实验中，GPT-6 Astra 看不清瓶口，便移动另一条机械臂，借它的相机找准位置，把花插了进去。Claude Fable 5.2 也展现出了跨任务的操作能力，同一个模型可以控制机器人拧开瓶盖、还原魔方。这些进展让我们看到，从图像和语言中学到的知识，到了新的操作任务里，依然能帮机器人想出办法。 [Astra 实证研究](https://www.sensenova.ai/has-gpt-6-astra-achieved-embodied-intelligence) · [Fable 5.2 真机演示](https://www.anthropic.com/claude-fable-and-mythos-5-2)

这样的泛化能力，能否也在一个更轻量的开源模型上实现，让机器人以更低的成本、更快地作出判断？我们从桌面操作做起。在 SenseNova-RoboRSI-27B-Preview 中，我们为具体任务配备相应的技能，让模型根据眼前的情况规划步骤，再调用视觉语言动作模型（VLA）操控机械臂。

到了正式版，不同任务共用同一套工具。模型自己判断何时调用 VLA、它给出的动作是否合适，以及何时该自己写代码，算出夹爪要去的位置（Code as Policy）。

<a name="评测与演示"></a>

在新版 RoboDojo Agent 评测中，我们使用通用 Harness，所有任务共用这套工具，不为单项任务提供示例或预设操作流程。按这一设置重新评测，Preview 平均 **10.0 分**，正式版达到 **34.7 分**。作为参照，RoboDojo 团队公布的 GPT-6 Astra 平均得分为 **27.6 分**，两项结果相差 **7.1 分**。我们还把同一份权重接到两款真实机械臂上。Piper 抓放物体成功 **6/10**，Piper X 堆叠积木成功 **9/10**，具身后训练全程使用仿真数据。

![27B 模型的多轮训练、执行方式、夹爪定位与真机任务总览][teaser]

*图 1　从仿真中的轨迹选择和代码控制，到真机上的抓放、叠积木和双臂还原二阶魔方。*

从工具迭代到真机验证，我们围绕四个问题做了实验。

1. **工具改进后，27B 怎样学会用？** 我们用三轮 RSI 迭代 Harness，再用新版工具下的交互轨迹训练模型。
2. **什么时候用 VLA，什么时候写代码？** 同一次操作中，模型可以先采用 VLA 的动作，再用代码修正落点。
3. **夹爪的目标位置怎样学得更准？** VLA2Agent 将已有的开源机器人操作示范转成单步空间监督，与长程决策轨迹联合训练。
4. **换到真机后能带走哪些能力？** 我们在两种平台上测试同一份权重，并回看模型如何定位、执行和纠错。

## 1. 多轮 RSI 让 27B 学会使用新工具

RoboRSI Harness 是连接模型、工具和机器人的运行框架。它决定模型能读到哪些相机画面和状态，能检查哪些候选动作，以及动作完成后能得到哪些反馈。模型负责依据这些信息作决定。改进其中一方，另一方能用好这些变化吗？我们从同一个 Preview 起点做了三组对照，在相同条件下逐轮评测。图 2 也给出了其他模型使用同一通用 Harness 的结果，其中 GPT-6.1 Sol 得 33.7 分，Qwen3.8-27B 基座得 9.1 分。

![RoboDojo 模型得分及仅改 Harness、仅训练模型、协同更新三种方式的逐轮对照][results]

*图 2　左图中带星号的七个模型使用同一通用 Harness，在 seed 0、layout 0 下自测。GPT-6 Astra 的 27.6 分来自 RoboDojo 团队。右图追踪同一起点后的三轮迭代。* [RoboDojo Agent 榜单](https://robodojo-benchmark.com/leaderboard#agent)

每轮先让教师模型 GPT-6 Astra 使用当前 Harness 做任务。仿真环境按任务条件给出回合得分和完成状态，交互记录则保留执行过程中模型看到的画面、选择的工具和动作结果。我们把两类信息放在一起看。例如，一次放置没有完成，可以沿轨迹查到模型采用了哪段动作、执行后物体落在哪里，以及当时的画面能否让模型发现偏差。

这些记录决定了工具的迭代顺序。我们先完善 VLA 候选轨迹的审查与片段选择，随后加入 Code as Policy 控制，最后补足空间定位工具和执行结果反馈。修改工具的阶段保持模型参数固定。每轮工具更新后，教师使用新接口采集轨迹。我们结合任务完成情况、回合得分和中间动作的执行质量，通过拒绝采样选出训练轨迹，再对 27B 做监督微调（SFT）。保留下来的样本同时包含工具选择前的观测与动作执行后的反馈。训练后的 27B 独立使用这版 Harness 做任务，它的执行记录进入下一轮分析。

![每轮先分析教师的执行反馈、改进 Harness，再采集筛选轨迹并微调 27B][loop]

*图 3　教师使用改进后的工具重新做任务，27B 因而有机会学到新版接口下的选择和执行过程。*

三轮结束后，仅改 Harness 得到 **24.8 分**，仅更新模型得到 **32.2 分**，两者协同更新达到 **34.7 分**。协同组在三轮评测中均取得最高分。在这组实验里，增加工具拓展了模型可执行的动作，而针对新版工具训练，进一步提高了实际任务得分。

## 2. 根据任务组合 VLA 和 Code as Policy

在此前的 RoboDojo 对照实验中，我们比较了 Astra 与 VLA 的两种分工。由 Astra 给出子目标后，VLA 可以直接执行，也可以先把候选动作交给 Astra，让它查看夹爪将如何移动，再决定采用、修改还是放弃。每次审查四步候选动作时，按任务类别平均的成功率比直接执行高出 **20 个百分点**。这也是我们让 27B 学习审查 VLA 轨迹的依据。 [实证研究，第 3.1 节](https://arxiv.org/html/2610.22471v2#S3.SS1)

VLA 能根据画面、指令和机器人状态给出一段连续动作，适合难以逐步指定轨迹的操作。Code as Policy 则让模型把已知的空间关系写成程序，用来计算目标、安排动作顺序和调用工具。两者的分工甚至可以发生在一次按键的几步之间。

在**教师 Astra** 的一条数字按键轨迹中，VLA 先把夹爪带到按键附近。接下来，Astra 用代码完成按下和抬起，并拒绝了一段可能重复按压的 VLA 候选轨迹。类似地，传送带上的动态接近、倾倒液体时的连续倾斜，都可以由 VLA 提出动作。当朝向、位移或按压次数已经明确时，模型可以用代码表达这些约束，再调用控制器执行。

在**正式版 27B** 的执行记录中，我们也看到模型针对局部目标编排动作。装试管时，它先对准孔位，再分步下探。操作衣物时，它给两只夹爪分别指定需要对准的衣角。按正式版实际执行的**环境步**统计，VLA 占 **41.9%**，Code as Policy 占 **58.1%**。一个任务可以为两边都贡献执行步数。

![27B 通过代码按朝向抓取、侧推 T 形块、双臂对准衣角和分步插入试管][control]

*图 4　四段仿真操作中的空间约束。侧推要控制方向，双臂要对准两个不同位置，试管插入则要求先对齐再下降。*

叠碗提供了一个能从头看到尾的例子。27B 使用 VLA 抓起一只碗，下一帧画面里，手中的碗比下方那只稍稍偏左。模型保持夹爪闭合，用代码把目标位置**向右平移 2 厘米**，再下降、松爪。先前的抓取仍然有效，模型只需修正放置这一段。

![仿真叠碗时的三个相机视角和控制时间线][bowl]

*演示 1　沿着时间线可以看到 VLA 抓取、模型检查相对位置，以及 Code as Policy 修正放置的切换点。*

## 3. 用长程轨迹学决策，用 VLA2Agent 学准动作目标

叠碗时，知道“还需要调整”只是第一步，模型还得给控制器一个合适的落点。长程教师轨迹能示范什么时候该重新观察、什么时候该动手，但要为每个局部动作都单独采集一段 Agent 交互，代价很高。已有的开源 VLA 示范里，执行动作时的画面、机械臂和夹爪状态、最终到达的位姿，本来就记录在一起。

我们把这些记录转成 Agent 的训练样本，称为 **VLA2Agent**。以一次放置为例，我们根据机械臂运动、夹爪的开合和多视角画面确定片段边界，辨认操作对象与执行手臂，用一句局部指令描述这段动作的意图，再取终点的夹爪位姿作为空间目标。这类单轮样本与长程轨迹一起训练，既帮助模型审查 VLA 候选轨迹是否会到达预期位置，也为它用代码指定控制目标提供监督。按构造**一个可比的动作目标样本**估算，复用 VLA 示范约为 **0.02 美元**，通过长程 Agent 交互采集约为 **0.50 美元**，单样本数据成本降低约 **96%**。

我们从 RoboDojo 官方仿真数据中取出 **374 个局部动作**，覆盖本次评测的全部 RoboDojo 任务，用来单独检验终点预测。给定已有观测，模型预测活动手臂的终点坐标，再与轨迹中记录的三维位置比较。179 个动作只涉及单臂，另有 195 个涉及双臂。对双臂动作，只有两只夹爪都落入误差阈值才计为命中。

| 开环末端位置预测 | 未加入 VLA2Agent | 加入 VLA2Agent |
| --- | ---: | ---: |
| 误差在 5 mm 内 | 4.3% | **15.5%** |
| 误差在 20 mm 内 | 9.4% | **29.7%** |
| 误差在 50 mm 内 | 24.9% | **49.2%** |
| 三维位置误差中位数 | 119.7 mm | **51.6 mm** |

![不同模型在三个距离阈值下的夹爪终点命中率及位置误差中位数][accuracy]

*图 5　在最严格的 5 mm 阈值下，加入 VLA2Agent 前后的命中率分别为 4.3% 和 15.5%。位置误差中位数按能够解析出坐标的预测计算。*

开环预测给出了毫米数，闭环执行可以看到偏差会导致什么后果。在下面选取的试管插入片段里，加入 VLA2Agent 的模型对准孔位，将试管插入，对照模型则停在孔缘。在积木放置片段中，两种配置的差异则体现在积木放下时是否对准了支撑面。

![两种训练配置在积木放置、试管插入和手表抓取中的执行对照][comparison]

*演示 2　左侧加入 VLA2Agent，右侧为对照组。三个片段分别展示放置、插入和抓取时的局部动作。*

## 4. 同一份权重，从仿真走到真实机械臂

具身后训练全程使用仿真数据。迁移到真机时，我们保留仿真评测的模型权重，分别适配 Piper 和 Piper X 的相机观测、机器人状态和控制接口。每个模型在每项任务上各做 10 次测试。

| 平台与任务 | 正式版的执行方式 | Qwen3.8-27B | Preview | 正式版 |
| --- | --- | ---: | ---: | ---: |
| Piper 抓放物体 | Code as Policy | 0/10 | 2/10 | **6/10（60%）** |
| Piper X 堆叠积木 | VLA × Code as Policy | 0/10 | 3/10 | **9/10（90%）** |

![Piper 抓放物体和 Piper X 堆叠积木各十次真机测试][trials]

*演示 3　每格对应正式版的一次测试。上面两行是 Piper 抓放，下方两行是 Piper X 叠积木。*

从单条真机轨迹中，还能看出哪些行为从仿真迁移了过来。一次抓放中，Piper 借右腕相机确定目标位置，再用左臂抓取。另一次，积木落在了盘前。新的相机画面显示放置尚未完成，模型重新定位积木，把它抓起、放回盘内。在支持 VLA 的配置中，我们还观察到，当候选轨迹与模型自己的定位结果不一致时，它会拒绝提案，重新生成或修正轨迹再执行。

![通过右腕视角辅助左臂定位，以及积木首次放偏后的再次抓取][recovery]

*图 6　上排展示跨视角定位，下排展示首次放偏后的纠错。判断是否需要再抓一次，要看物体最终落在了哪里。*

再往前一步，我们让模型尝试一个未经专项训练的任务。它用 Code as Policy 协调双臂持握、翻转二阶魔方，逐面查看六个面，再完成还原。下面记录了这次真机执行。

![双臂翻转、观察并还原二阶魔方，以及同步的两个腕部相机视角][cube]

*演示 4　从腕部画面可以跟踪翻面后的朝向变化。模型未针对还原魔方做专项训练。*

## 展望

### 可持续的 RSI

在已完成的三轮 RSI 中，教师模型 GPT-6 Astra 先使用 Harness 执行任务。每轮根据执行反馈改进 Harness，再由教师模型指导 27B 使用新版工具生成交互轨迹。这些轨迹经拒绝采样筛选后用于监督微调，使 27B 逐步学会审查 VLA 提出的轨迹、选取合适的片段执行，并在需要时编写代码调整夹爪的目标位姿。沿着 Self-Taught Optimizer（STOP）探索的方向，我们希望让模型根据自身的执行记录提出工具改进，再通过复测决定是否采纳。

要让这个过程持续下去，首先要判断**何时结束本轮对 Harness 和模型的优化**。我们计划在相同的验证任务和交互预算下，固定模型比较不同版本的 Harness，再固定 Harness 比较不同的模型权重，最后复测两者的组合。如果模型训练的收益已经趋于平稳，而改进工具仍能解决反复出现的问题，就先固定模型，继续优化 Harness。当两者分别更新和联合更新都难以带来可复现的提升时，就结束本轮优化，选取表现稳定、保住已有能力的组合作为下一轮的起点。

另一个是 **哪些上下文值得保留**。轨迹越来越长，把每帧画面和每次工具返回都送回模型，推理开销会随之增长。我们在实验中还观察到，当上下文已经很长时，继续加入交互历史对任务表现的提升很有限。模型越小，这一瓶颈越明显。但如果压缩掉松爪后的画面，下一步就可能把“夹爪已打开”误当成“物体已放稳”。我们计划保留关键动作前后的观测、候选轨迹被否决的依据，以及已经尝试过的纠错方式，并记录对应的模型和 Harness 版本。每轮用这些记录定位问题，在留出的任务上复测。

### 面向跨环境泛化的强化学习

通过多轮 RSI，SenseNova-RoboRSI-27B 已经学会调用 VLA、审查候选轨迹，并用代码调整夹爪位置，**为后续强化学习打下了冷启动基础**。有了这些基本能力，模型在探索初期就更有机会完成任务，获得成功奖励。后续 RL 可以利用执行结果，进一步优化模型选择工具、切换执行方式和纠正偏差的时机。我们认为，这样的冷启动有望减少无效试错，显著提高 RL 的样本效率。

有了冷启动基础，下一步还要考虑模型从哪些环境中积累经验。现有具身 RL 研究常在单一仿真平台内采集交互、优化策略。在一个平台上学会的操作和纠错方法，换个环境还能用吗？我们希望把后续训练扩展到**跨仿真平台强化学习（Cross-Simulation RL）**。我们计划用**同一套通用 RoboRSI Harness** 接入多个仿真平台，包括 LIBERO、RoboDojo 以及根据真实场景重建的仿真环境，覆盖单臂操作、双臂协作等不同任务。模型通过这套 Harness 获取观测、调用工具并接收执行反馈，训练和评测沿用相同的交互方式。我们将从同一份 27B 权重出发，在相同的环境步数预算下比较单平台训练与跨平台混合训练，并在训练时留出的平台和任务上检验迁移能力。我们关心的是，换了仿真环境或机械臂配置，模型能否仍然用好这套工具，自主选择动作，并在执行出现偏差时调整策略。

### 在仿真中还原真实场景，再回到真机验证（Real2Sim2Real）

GPT-6 Astra 展现了用代码设计和重建三维场景的能力。研究者已经用它根据机器人操作录像，在仿真中还原桌面布局和物体的位置。Piper X 叠积木演示中使用的 VLA 就是按这条路线训练的。我们在仿真中还原真实桌面，生成积木操作示范，再与其他任务的真机示范混合训练。在真机测试中，RoboRSI-27B 结合这套 VLA 与 Code as Policy 完成积木堆叠，成功率为 **90%（9/10）**。

RoboRSI-27B 的真机实验也表明，视觉语言模型（VLM）在仿真中学到的桌面操作与纠错能力，可以迁移到真实机械臂上。结合这两项进展，我们希望建立一条更高效的 Real2Sim2Real 训练流程。先根据真实场景搭建仿真环境，改变物体摆放和相机视角，批量采集轨迹用于训练，再回到真机验证。遇到物体放偏或滑落时，可以把当时的条件带回仿真，复现问题、改进策略。我们会用达到相同真机成功率所需的真机数据量和测试时间来衡量效率，也会检验更小的开源模型能否在缩短响应时间的同时保留操作能力。

## 开源

[GitHub](../README.zh-CN.md) · [RoboRSI Harness 使用指南](./getting-started.zh-CN.md)

| 模型 | 模型权重 |
| --- | --- |
| SenseNova-RoboRSI-27B | [🤗 Hugging Face](https://huggingface.co/sensenova/SenseNova-RoboRSI-27B) |
| SenseNova-RoboRSI-9B | Coming Soon |

[评测与演示](#评测与演示) · [快速开始](#快速开始)

## 快速开始

### 安装 vLLM

以下示例在 Linux x86_64 上使用四张支持 BF16 的 NVIDIA GPU。请在推理环境中安装 vLLM 0.21.0 的 CUDA 12.9 版本，并确保驱动与该版本兼容。其他 CUDA 版本见 [vLLM 安装指南](https://docs.vllm.ai/en/v0.21.0/getting_started/installation/gpu/)。

```bash
python -m pip install 'vllm==0.21.0+cu129' huggingface_hub \
  --extra-index-url https://wheels.vllm.ai/0.21.0/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129
```

### 下载模型

下载正式版的权重、分词器、视觉处理器和对话模板。访问私有仓库时，先用有权限的账号运行 `hf auth login`。如果已有完整的本地模型，可将 `MODEL_DIR` 设为其绝对路径，直接启动服务。[🤗 模型权重](https://huggingface.co/sensenova/SenseNova-RoboRSI-27B/tree/main)

```bash
export MODEL_DIR="$PWD/SenseNova-RoboRSI-27B"
hf download sensenova/SenseNova-RoboRSI-27B \
  --local-dir "$MODEL_DIR" --exclude 'assets/*'
```

### 启动服务

在四张 GPU 上以张量并行方式启动模型。服务使用模型自带的对话模板，并启用推理和工具调用解析。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

vllm serve "$MODEL_DIR" \
  --served-model-name sensenova-roborsi-27b \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --tensor-parallel-size 4 \
  --max-model-len 262144 \
  --generation-config auto \
  --chat-template "$MODEL_DIR/chat_template.jinja" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}' \
  --enable-prefix-caching \
  --mamba-cache-mode align
```

### 验证接口

在同一台机器上打开另一个终端，检查服务状态，再发送一条机器人规划请求。服务就绪后，健康检查会返回 HTTP 200，模型回复位于 `choices[0].message`。

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "sensenova-roborsi-27b",
    "messages": [{
      "role": "user",
      "content": "A robot needs to place a mug on a coaster. Give a short plan and explain how to verify that the mug is stable before releasing it."
    }],
    "max_tokens": 8192,
    "chat_template_kwargs": {"reasoning_effort": "medium"},
    "stream": false
  }'
```

在运行 Harness 的终端中设置以下环境变量，然后按照[已部署模型评测指南](./model-evaluation.zh-CN.md)运行评测。

```bash
export OPENAI_BASE_URL='http://127.0.0.1:8000/v1'
export OPENAI_API_KEY='EMPTY'
export MODEL_NAME='sensenova-roborsi-27b'
```

### RoboDojo 一键复现

按 [RoboDojo 环境准备指南](../benchmarks/robodojo/README.zh-CN.md#环境准备)准备并启动 Benchmark 服务。配置好上面的模型连接后，在仓库根目录运行：

```bash
python scripts/evaluation/evaluate_model.py --benchmark robodojo \
  --benchmark-url robodojo=http://127.0.0.1:18800 \
  --run-id roborsi-27b-robodojo-001
```

将 Benchmark 地址替换为实际服务地址，结果保存在
`work/model_eval/results/roborsi-27b-robodojo-001/`。

## 许可证

SenseNova-RoboRSI-27B 采用 Apache License 2.0。

[teaser]: ../assets/roborsi-27b-teaser.zh-CN.png
[results]: ../assets/roborsi-27b/robodojo-results.png
[loop]: ../assets/roborsi-27b/rsi-loop.png
[control]: ../assets/roborsi-27b/code-as-policy.png
[bowl]: ../assets/roborsi-27b/bowl-stacking.webp
[accuracy]: ../assets/roborsi-27b/eef-accuracy.png
[comparison]: https://huggingface.co/sensenova/SenseNova-RoboRSI-27B/resolve/main/assets/roborsi-27b/vla2agent-comparison.webp?v=eb8ef2fb7a70
[trials]: ../assets/roborsi-27b/real-robots.webp
[recovery]: ../assets/roborsi-27b/real-robot-recovery.png
[cube]: ../assets/roborsi-27b/dual-arm-cube.webp
