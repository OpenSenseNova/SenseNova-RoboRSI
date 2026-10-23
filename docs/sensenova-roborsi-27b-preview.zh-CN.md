# SenseNova-RoboRSI-27B-Preview

[English](sensenova-roborsi-27b-preview.md) · [🤗 Hugging Face](https://huggingface.co/sensenova/sensenova-roborsi-27b-preview) · [演示视频](#演示视频) · [快速开始](#快速开始)

**SenseNova-RoboRSI-27B-Preview** 是面向具身智能体的开源 **27B 视觉语言模型**。
通过模型与 Harness 的协同 RSI，模型从经过验证的交互经验中学习，形成支持仿真与真机任务的
规划、动作决策与反馈纠错能力。

模型权重已在 Hugging Face 开放，支持社区部署、复现和继续训练。本文介绍模型部署方法，
以及使用 **RoboRSI Harness** 在 **LIBERO-Pro 和 RoboCasa365** 上复现评测结果的流程。
RoboDojo 的复现流程将随后续 RoboRSI 对该 benchmark 的支持一同开放。

![SenseNova-RoboRSI-27B-Preview 在 RoboDojo、LIBERO-Pro 与 RoboCasa365 上的评测结果](../assets/roborsi-27b-preview-benchmarks.webp)

- **RoboDojo：** 在配合 VLA 执行的设置下取得 **26.13 的平均得分**和 **20.83% 的任务成功率**，平均得分仅比 RoboDojo 团队评测的 GPT-6 Astra 低 **2.84 分**。模型能够主动观察视觉细节、推理空间关系，并根据执行反馈纠正动作，展现了解决复杂操作任务的能力。
- **LIBERO-Pro：** 在 VLA 配合执行（System-2 + System-1）与 VLM 直接控制（System-2）两种架构下，成功率分别达到 **92.5%** 和 **95.0%**，在图中两种设置的对比模型中均仅次于 GPT-6 Astra，展现了跨执行架构的具身规划与决策能力。
- **RoboCasa365：** 相较于 Qwen3.8-27B 基座模型，RSI 训练后成功率由 **26.0% 提升至 46.5%**，平均输出 Token 由 **59.5K 降至 15.9K**，减少 **73.3%**。模型将交互经验转化为更强任务能力，同时提升推理效率。

## 演示视频

**真机演示**

![真机演示](../assets/real-robot-27b-preview.webp)

**仿真演示 · RoboDojo**

![仿真演示 · RoboDojo](../assets/robodojo-27b-preview.webp)

**仿真演示 · LIBERO**

![仿真演示 · LIBERO](../assets/libero-27b-preview.webp)

**仿真演示 · RoboCasa365**

![仿真演示 · RoboCasa365](../assets/robocasa-27b-preview.webp)

## 快速开始

### 1. 部署模型

#### 安装 vLLM

在推理环境中安装 CUDA 12.9 版本，服务器需已有兼容的 NVIDIA 驱动：

```bash
pip install "vllm==0.21.0" huggingface_hub \
  --extra-index-url https://download.pytorch.org/whl/cu129
```

其他 CUDA 版本见 [vLLM 安装指南](https://docs.vllm.ai/en/v0.21.0/getting_started/installation/gpu/)。

#### 下载模型

下载完整 checkpoint，包括 tokenizer、processor、生成配置和 chat template：

```bash
export MODEL_DIR="$PWD/models/sensenova-roborsi-27b-preview"
hf download sensenova/sensenova-roborsi-27b-preview \
  --local-dir "$MODEL_DIR"
```

如已有本地 checkpoint，将 `MODEL_DIR` 设置为其绝对路径。

#### 启动服务

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

vllm serve "$MODEL_DIR" \
  --served-model-name sensenova-roborsi-27b-preview \
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

#### 验证接口

在同一台机器的另一个终端中检查服务状态，并发送一个简短的机器人规划请求：
将杯子放到杯垫上，并说明松爪前如何确认杯子放稳。

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "sensenova-roborsi-27b-preview",
    "messages": [{
      "role": "user",
      "content": "A robot needs to place a mug on a coaster. Give a short plan and explain how to verify that the mug is stable before releasing it."
    }],
    "max_tokens": 8192,
    "chat_template_kwargs": {"reasoning_effort": "medium"},
    "stream": false
  }'
```

### 2. 接入 RoboRSI Harness 并运行评测

按照[模型评测指南](model-evaluation.zh-CN.md)安装 RoboRSI Harness 和配套评测环境，使用下面指令运行评测：

```bash
export OPENAI_BASE_URL='http://127.0.0.1:8000/v1'
export OPENAI_API_KEY='EMPTY'
export MODEL_NAME='sensenova-roborsi-27b-preview'

# LIBERO-Pro：VLM + VLA（System 2 + System 1）
# 准备工具环境、下载权重并在后台部署 VLA/SAM3；不自动启动评测
python scripts/deploy/libero_tools.py serve \
  --model-config configs/models/roborsi_27b.yaml \
  --work-dir work/model_eval/libero_tools &

# 等待输出 Services ready 后，再执行下面的评测命令（地址为默认值）
ROBORSI_VLA_URL=http://host.docker.internal:24001 \
ROBORSI_SAM3_URL=http://host.docker.internal:24002 \
python scripts/evaluation/evaluate_model.py --benchmark libero_pro \
  --benchmark-config configs/benchmarks/libero_pro_80.yaml \
  --model-config configs/models/roborsi_27b.yaml \
  --agent-config configs/agents/libero/sensenova_roborsi_27b_vla_sam3.yaml

# LIBERO-Pro：VLM 直接控制（System 2）；无需执行上面的 VLA/SAM3 部署命令
python scripts/evaluation/evaluate_model.py --benchmark libero_pro \
  --benchmark-config configs/benchmarks/libero_pro_80.yaml \
  --model-config configs/models/roborsi_27b.yaml \
  --agent-config configs/agents/libero/sensenova_roborsi_27b_no_tools.yaml

# RoboCasa365
python scripts/evaluation/evaluate_model.py --benchmark robocasa365

# RoboDojo：评测支持将随后续 RoboRSI Harness 更新开放
```
