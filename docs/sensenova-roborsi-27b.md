# SenseNova-RoboRSI-27B

[简体中文](sensenova-roborsi-27b.zh-CN.md) · [🤗 Hugging Face](https://huggingface.co/sensenova/SenseNova-RoboRSI-27B) · [Results & Demos](#results--demos) · [Quickstart](#quickstart)

SenseNova-RoboRSI-27B is a vision-language model for robot manipulation. It can request motion from a vision-language-action (VLA) policy, review a proposed trajectory, or write code to compute and execute a gripper target pose (Code as Policy). After each action, the agent uses new camera observations and robot state to decide what to do next.

![Training workflow, control modes, evaluation results and start/end frames from two physical robots](../assets/roborsi-27b-teaser.png)

*Figure 1. The same checkpoint runs in RoboDojo and on two physical robot platforms.*

## Results & Demos

### RoboDojo simulation

The GPT-6 Astra result comes from the RoboDojo Team’s published leaderboard. The release scores 7.1 points higher. We evaluated the other five entries at **seed 0, layout 0**, using the same general-purpose Harness with no task-specific skills. [RoboDojo Agent leaderboard](https://robodojo-benchmark.com/leaderboard#agent)

![RoboDojo agent comparison and scores across three rounds of RSI](../assets/roborsi-27b/en/robodojo-results.png)

*Figure 2. Starting from a Preview score of 10.0, three rounds of updates reached 34.7 when both the model and Harness changed, 32.2 with model updates alone and 24.8 with Harness updates alone.*

We measured the distance in millimeters between the predicted and recorded gripper endpoint coordinates (x, y, z) on **374 local actions from all RoboDojo tasks**. Of these, 179 use one arm and 195 use both. A dual-arm prediction counts as a hit only when both arms meet the threshold. The median uses predictions with parseable coordinates.

VLA2Agent supplies end-effector target poses from open VLA demonstrations.

| Training configuration | Within 5 mm ↑ | Within 20 mm ↑ | Within 50 mm ↑ | Median XYZ error ↓ |
| --- | ---: | ---: | ---: | ---: |
| Without VLA2Agent | 4.3% | 9.4% | 24.9% | 119.7 mm |
| With VLA2Agent | **15.5%** | **29.7%** | **49.2%** | **51.6 mm** |

![Position accuracy at three distance thresholds and median position error](../assets/roborsi-27b/en/eef-accuracy.png)

*Figure 3. Open-loop end-effector predictions, with and without VLA2Agent supervision and across reference models.*

The open-loop test measures how accurately the agent predicts a gripper target. In the RoboDojo trial below, the bowl is slightly left of the stack. The agent keeps its grip and uses Code as Policy to shift the target 2 cm to the right before lowering the bowl.

![Three camera views of the robot stacking bowls](../assets/roborsi-27b/en/bowl-stacking.webp)

*Demo 1. Bowl stacking in simulation. Follow the three camera views and the control timeline as the agent chooses a VLA motion, checks the grasp and switches to code to correct a placement.*

![Camera views and control code from four RoboDojo manipulation tasks](../assets/roborsi-27b/en/code-as-policy.png)

*Figure 4. Code as Policy in simulation. The agent uses object orientation to plan a grasp, nudges a T-block sideways, aligns both grippers with garment corners, and inserts a test tube in two steps.*

### Physical robots (Piper & Piper X)

With **no real-robot data in embodied post-training**, SenseNova-RoboRSI-27B used the same checkpoint from simulation to pick and place objects on Piper and stack blocks on Piper X. We adapted only the camera, state and control interfaces to each platform. Each model ran ten trials per task.

| Platform | Task | Release execution mode | RoboRSI-27B | Preview | Qwen3.8-27B |
| --- | --- | --- | ---: | ---: | ---: |
| Piper | Object pick-and-place | Code as Policy only | **6/10 (60%)** | 2/10 | 0/10 |
| Piper X | Block stacking | VLA × Code as Policy | **9/10 (90%)** | 3/10 | 0/10 |

![Piper X stacking blocks on the left and Piper placing an object in a dish on the right](../assets/roborsi-27b/en/real-robots.webp)

*Demo 2. Both robots use the same checkpoint. On the left, Piper X stacks blocks using VLA and Code as Policy. On the right, Piper places an object in a dish using Code as Policy alone.*

Two Piper trials show how the agent uses camera feedback to decide where to grasp and when to try again after a missed placement.

![Right-wrist localization for a left-arm grasp and recovery from a missed placement](../assets/roborsi-27b/en/real-robot-recovery.png)

*Figure 5. Two Piper trials. In the first, the right wrist camera locates the target for a left-arm grasp. In the second, the block lands in front of the dish. The agent locates it again, picks it up and places it inside.*

## Quickstart

### Install vLLM

This example uses four NVIDIA GPUs with BF16 support on x86-64 Linux. In your inference environment, install the CUDA 12.9 build of vLLM 0.21.0 with a compatible NVIDIA driver. For other CUDA builds, see the [vLLM installation guide](https://docs.vllm.ai/en/v0.21.0/getting_started/installation/gpu/).

```bash
python -m pip install 'vllm==0.21.0+cu129' huggingface_hub \
  --extra-index-url https://wheels.vllm.ai/0.21.0/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129
```

### Download the model

Download the release weights, tokenizer, visual processor and chat template. For a private repository, first run `hf auth login` with an account that has access. If you already have the complete checkpoint, set `MODEL_DIR` to its absolute path and continue to the next step. [🤗 Model Weights](https://huggingface.co/sensenova/SenseNova-RoboRSI-27B/tree/52ac9fdd07cbad42fab74217e595db12b4f1a000)

```bash
export MODEL_DIR="$PWD/SenseNova-RoboRSI-27B"
hf download sensenova/SenseNova-RoboRSI-27B \
  --local-dir "$MODEL_DIR" --exclude 'assets/*'
```

### Start the service

Serve the model across four GPUs using tensor parallelism. The server uses the checkpoint’s chat template and enables reasoning and tool-call parsing.

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

### Check the endpoint

In a second terminal on the same machine, check the service health and send a short robot-planning request. A ready server returns HTTP 200 for the health check and a reply in `choices[0].message` for the chat request.

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

To run benchmarks, set these variables in the Harness terminal and follow the [evaluation guide for deployed models](./model-evaluation.md).

```bash
export OPENAI_BASE_URL='http://127.0.0.1:8000/v1'
export OPENAI_API_KEY='EMPTY'
export MODEL_NAME='sensenova-roborsi-27b'
```

**RoboDojo reproduction.** We will release the reproduction script after we submit our results to the RoboDojo leaderboard.

## License

SenseNova-RoboRSI-27B is released under the Apache License 2.0.
