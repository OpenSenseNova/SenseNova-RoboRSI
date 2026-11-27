# SenseNova-RoboRSI-27B: Integrating VLA and Code as Policy Through Model–Harness Co-Evolution

[简体中文](./sensenova-roborsi-27b.zh-CN.md) · [Model weights](https://huggingface.co/sensenova/SenseNova-RoboRSI-27B) · [GitHub](../README.md) · [Results & Demos](#results--demos) · [Quickstart](#quickstart)

In a flower-arranging trial from our earlier empirical study, GPT-6 Astra could not see the mouth of the vase. It moved the other arm to get a better view through its wrist camera, then guided the flower into place. Claude Fable 5.2 has also shown how one model can handle different physical tasks, from unscrewing a bottle cap to solving a puzzle cube. These examples suggest that knowledge learned from images and language can help a robot work out what to do in a new task. [Our empirical study of Astra](https://www.sensenova.ai/has-gpt-6-astra-achieved-embodied-intelligence) · [Fable 5.2 robot demonstrations](https://www.anthropic.com/claude-fable-and-mythos-5-2)

Could a smaller, open model learn to generalize in this way and make decisions faster and at lower cost? We started with tabletop manipulation. For SenseNova-RoboRSI-27B-Preview, we supplied task-specific skills. The model planned its next steps from the scene in front of it and called a vision-language-action (VLA) policy to move the arm.

In the full release, tasks share a common set of tools. The model decides when to call the VLA, whether its proposed motion fits the task, and when to write code to calculate a target pose for the gripper (Code as Policy).

<a name="results--demos"></a>

On the new RoboDojo Agent evaluation, with no task-specific examples, skills or preset procedures supplied to the agent, **SenseNova-RoboRSI-27B averaged 34.7**, up from **10.0** for Preview. The RoboDojo Team separately reports **27.6** for GPT-6 Astra, a difference of 7.1 points. We also took the same weights from simulation to two physical robots. Piper placed objects in a dish in **6 of 10 trials**, and Piper X stacked blocks in **9 of 10**. [RoboDojo Agent leaderboard](https://robodojo-benchmark.com/leaderboard#agent)

![The training loop, examples of VLA and code control, and the two physical robot tasks][teaser]

*Figure 1. SenseNova-RoboRSI-27B learns to use VLA proposals and write control code in simulation. The same checkpoint runs on Piper and Piper X.*

1. **Train the model as its tools improve.** Each round starts with a teacher using the current tools. Its actions and their outcomes guide revisions, then selected trajectories teach the 27B model how to use the new tools.
2. **Let the agent choose how to move.** A VLA can supply a grasp or a continuous tilt. Code can set a precise target or correct part of a motion. A single task can use both.
3. **Teach the gripper where to finish.** We pair long agent trajectories with individual action targets extracted from existing VLA demonstrations, at an estimated 4% of the data-construction cost per comparable target.
4. **Test what transfers to hardware.** With only simulation data in embodied post-training, the agent carried behaviors such as finding a target from another camera and retrying a missed placement onto physical arms.

## 1. Improving the tools and training the model to use them

Preview relied mainly on the VLA to generate movement. The release gives the agent a choice over how far to follow a proposed trajectory and where to send the gripper next. The software that makes those choices executable is the **RoboRSI Harness**. It exposes camera views and robot state, executes the requested motion and returns the resulting observation. Figure 2 places the release alongside other models evaluated with those same tools, including GPT-6.1 Sol at 33.7 and the Qwen3.8-27B base model at 9.1.

![RoboDojo model scores and three ways to update the model and its Harness over three rounds][results]

*Figure 2. The stars mark seven models we evaluated with the same general-purpose Harness at seed 0, layout 0. The Astra score comes from the RoboDojo Team. The right panel follows three update strategies from the same Preview baseline.*

We developed the Harness and the 27B model over three rounds of recursive self-improvement (RSI). At the start of each round, GPT-6 Astra, our teacher, attempted tasks with the current tools. We compared each tool request with what the robot actually did. The simulator’s own task score recorded progress, while the camera images and robot state after each action let us inspect an inefficient move or a missed grasp. The early revisions let the agent review a VLA candidate and select a segment. Next came direct control through code, followed by better tools for spatial localization and better feedback after execution.

Once the revised Harness was ready, Astra ran tasks with it. We used rejection sampling to select trajectories for supervised fine-tuning (SFT), considering task completion, the episode score and the quality of intermediate actions. Each selected trace kept the observation before a tool choice and the feedback after the motion. The 27B student then attempted tasks independently with the same tools. Its records fed into the next round.

![The RSI cycle from teacher rollouts and tool revisions through trajectory selection, fine-tuning and student evaluation][loop]

*Figure 3. Each change to the Harness produces a new data-collection setting. The teacher demonstrates the revised tools, then the student uses them in its own rollouts.*

Which part of the improvement came from the tools, and which from training? Starting at **10.0**, changing only the Harness reached **24.8** after three rounds. Updating only the model reached **32.2**. Updating the Harness and then training the model to use it reached **34.7**, with this combination ahead at the end of every round.

## 2. Choosing when to use VLA and when to write code

In our earlier RoboDojo study, we compared two ways for Astra and a VLA to work together. Once Astra set a subgoal, the VLA could execute its motions directly or submit them for review. Astra could inspect where those motions would take the gripper, then accept, revise or reject them. Reviewing four proposed actions at a time raised success by **20 percentage points** over direct execution, averaged across task categories. This finding motivated us to train 27B to review VLA trajectories. [Empirical study, Section 3.1](https://arxiv.org/html/2610.22471v2#S3.SS1)

A VLA predicts motion from images, language and robot state. In Astra’s teacher traces, it generated a moving approach to an object on a conveyor and a continuous tilt for pouring. Other actions had a geometric target the agent could express in code, such as the height of a stacked block or the down-and-up motion of a keypad press.

The keypad required both kinds of control. Astra used the VLA to approach the target, then wrote code to press the key and lift the gripper. When the VLA proposed a motion that risked pressing the key again, Astra rejected it. A plausible motion in isolation would have been the wrong motion at that point in the sequence.

The trained 27B model made a similar choice while stacking bowls. It used the VLA to grasp and lift the upper bowl. A fresh camera view showed the bowl sitting slightly left of the one underneath. The model kept its grip and used code to shift the target **2 cm to the right** before lowering and releasing the bowl. Across the release model’s recorded environment steps, **41.9%** used VLA actions and **58.1%** used Code as Policy. Both kinds of step can contribute to the same completed task.

![Four examples of code controlling orientation, sideways motion, two-arm alignment and insertion][control]

*Figure 4. The agent uses an object’s orientation to plan a grasp, pushes a T-block sideways, aligns the grippers with two corners of a garment and lowers a test tube into an opening in stages.*

![Three camera views and a timeline of the simulated bowl-stacking episode][bowl]

*Demo 1. Watch the interval between the grasp and release. The agent retains the VLA grasp while correcting the placement target in code.*

## 3. Learning the next decision and the next gripper target

A long teacher trajectory shows when to intervene. A placement can still fail if the target is a few centimeters off. In testing, the 27B model lagged behind Astra on fine spatial localization, even after training on teacher trajectories. We wanted to give it more examples of exactly where a successful motion had ended.

Our **VLA2Agent** pipeline turns existing robot demonstrations into single-turn agent examples. For a local action such as a grasp or placement, we use arm motion, gripper-state changes and camera views to identify the action boundaries. The starting observations and a short instruction describe the object and action. The recorded gripper pose at the end supplies the target. We mix these local examples with the longer trajectories that teach tool selection and recovery. Endpoint supervision is useful for both judging where a VLA trajectory would take the gripper and specifying a target in code. We estimate that extracting a comparable action target this way costs about **$0.02**, versus about **$0.50** to collect one through long agent interactions. That is a roughly **96% reduction** in data-construction cost per target.

We tested endpoint prediction on **374 local actions spanning all RoboDojo tasks**, drawn from the benchmark’s simulation data. There were 179 single-arm and 195 dual-arm actions. For each active arm, we compared the predicted XYZ endpoint with the recorded endpoint in millimeters. A dual-arm action counted as a hit only if both arms met the threshold. We computed the median position error over predictions with parseable coordinates.

| Training configuration | Within 5 mm | Within 20 mm | Within 50 mm | Median position error |
| --- | ---: | ---: | ---: | ---: |
| Without VLA2Agent | 4.3% | 9.4% | 24.9% | 119.7 mm |
| With VLA2Agent | **15.5%** | **29.7%** | **49.2%** | **51.6 mm** |

![End-effector prediction accuracy at three thresholds and median position error][accuracy]

*Figure 5. In this open-loop test, the model predicts a target from recorded observations. At the tightest threshold of 5 mm, 4.3% of actions were hits without VLA2Agent supervision, rising to 15.5% with it.*

The closed-loop clips below show what a difference in target position can look like once the robot moves. In the block-placement clip, the model trained with VLA2Agent aligns the block with its support and sets it down. The comparison model misses the support and the block falls. In the insertion clip, one gripper aligns the test tube with the opening while the other stops at the rim.

![Side-by-side executions of block placement, test-tube insertion and watch grasping with and without VLA2Agent][comparison]

*Demo 2. Three selected executions with VLA2Agent supervision on the left and without it on the right.*

## 4. Taking the same weights from simulation to physical robots

Different camera views, contacts and control interfaces can all change what happens after an arm moves. To test transfer, we took the same checkpoint used in RoboDojo and adapted the camera observations, state inputs and control interface for each robot. All embodied post-training used simulation data. Each model received ten attempts at each physical task.

| Platform and task | Release execution mode | Qwen3.8-27B base | RoboRSI-27B-Preview | RoboRSI-27B |
| --- | --- | ---: | ---: | ---: |
| Piper, placing objects in a dish | Code as Policy only | 0/10 | 2/10 | **6/10 (60%)** |
| Piper X, stacking blocks | VLA and Code as Policy | 0/10 | 3/10 | **9/10 (90%)** |

![Twenty physical trials, with Piper placing objects and Piper X stacking blocks][trials]

*Demo 3. Each tile shows one attempt with the release model. The top two rows show Piper, and the bottom two show Piper X.*

On Piper, the model moved the right wrist camera into position to find an object that the left arm needed to grasp. In another trial, a block landed just in front of the dish. The gripper had opened, but the next view showed where the block had actually landed. The agent located it again, picked it up and placed it inside the dish. On the VLA-equipped setup, we also observed the agent reject a candidate trajectory when it disagreed with the target position seen through the cameras, then revise the motion before execution.

![Piper using a wrist camera to guide the opposite arm and recovering from a missed placement][recovery]

*Figure 6. The upper sequence shows the right wrist camera guiding a left-arm grasp. The lower sequence follows the block from the missed placement through the second grasp.*

We also tried a two-arm task for which the model had received no task-specific training. Using Code as Policy, it coordinated the arms to inspect all six faces of a 2×2 puzzle cube, turn it and solve it. The clip follows one complete solve.

![Two robot arms inspecting and solving a 2×2 cube, with views from both wrist cameras][cube]

*Demo 4. The wrist views show how the cube changes orientation as the arms take turns holding and turning it.*

## Future Directions

### Sustainable RSI

In each of the three RSI rounds, the teacher model, GPT-6 Astra, ran tasks with the current Harness. Feedback from those runs informed revisions to the Harness. The teacher then guided the 27B model as it generated interaction traces with the revised tools. Supervised fine-tuning on traces selected through rejection sampling taught 27B to review VLA trajectories, choose which segments to execute and write code to adjust the target gripper pose when needed. Inspired by the Self-Taught Optimizer (STOP), we would like the model to use its own traces to propose tool improvements and test them before adoption. [STOP paper](https://arxiv.org/abs/2310.02304)

A first challenge is deciding **when to end the current round of Harness and model optimization**. On the same validation tasks and interaction budget, we plan to compare Harness revisions with the model held fixed, compare model checkpoints with the Harness held fixed, then test the two updates together. If further model training yields little improvement while a tool change still resolves a recurring failure, we can freeze the model and keep working on the Harness. Once neither update alone nor the two together yields a reproducible gain, we would close the round and carry forward a combination that performs consistently and retains earlier capabilities.

The second is **which context to retain**. Feeding every frame and tool result back to the model becomes expensive as trajectories grow. In our experiments, once the context was long, adding more interaction history yielded little further improvement in task performance. This plateau was more pronounced in smaller models. Yet dropping the view immediately after release can leave the agent treating an open gripper as evidence of a stable placement. We plan to retain observations around consequential actions, the evidence for rejecting a proposed trajectory and the corrections already attempted, together with the model and Harness versions that produced the trace. Each round can use those records to diagnose failures and test changes on held-out tasks.

### Reinforcement Learning for Cross-Environment Generalization

Through multiple rounds of RSI, SenseNova-RoboRSI-27B has learned to call a VLA, inspect candidate trajectories and adjust the gripper position in code. **These skills give the model a starting policy for reinforcement learning.** With these skills in place, the model has a better chance of completing tasks and receiving a success reward early in exploration. RL could then use execution outcomes to refine when the model calls a tool, switches control modes or corrects a motion. We expect this starting policy to reduce unproductive trials and substantially improve sample efficiency.

With that starting policy in place, the next question is where the model gains experience. Embodied RL studies often collect rollouts and optimize policies within a single simulation platform. Will the same manipulation and recovery strategies work in another environment? We plan to explore **cross-simulation reinforcement learning (Cross-Simulation RL)** with **one general-purpose RoboRSI Harness** across platforms, including LIBERO, RoboDojo and simulations reconstructed from physical scenes. Tasks would span single-arm manipulation and coordinated two-arm work. The model would use the Harness in the same way during training and evaluation to observe the scene, call tools and receive execution feedback. Starting from the same 27B checkpoint and using the same environment-step budget, we would compare single-platform training with training on a mixture of platforms. We would reserve entire platforms and tasks for transfer tests. When the simulator or arm configuration changes, can the model still use the same tools to choose a motion and adjust its strategy when execution goes off course?

### Reconstructing real scenes in simulation and testing on hardware (Real2Sim2Real)

GPT-6 Astra has shown it can design and reconstruct 3D scenes in code. Researchers have used it to recreate tabletop layouts and object positions in simulation from recordings of robots at work. We used this approach to train the VLA in our Piper X block-stacking demo. We reconstructed the physical tabletop in simulation, generated block-handling demonstrations there and combined them with real demonstrations from another task. On the physical robot, RoboRSI-27B used this VLA together with Code as Policy to stack the blocks in **9 of 10 trials (90%)**.

Our hardware trials with RoboRSI-27B also show that a vision-language model (VLM) can transfer tabletop manipulation and error recovery skills learned in simulation to physical robot arms. Together, these results suggest a more efficient Real2Sim2Real training process. We would reconstruct a physical scene, vary object placement and camera views to collect training trajectories at scale, then test the trained model on the robot. If an object lands off target or slips, we could recreate those conditions in simulation, reproduce the failure and refine the policy. We would measure efficiency by how much real-world data and robot time it takes to reach the same success rate, and test whether smaller open models can respond faster while retaining their manipulation skills.

## Open source

[GitHub](../README.md) · [RoboRSI Harness guide](./getting-started.md)

| Model | Model weights |
| --- | --- |
| SenseNova-RoboRSI-27B | [🤗 Hugging Face](https://huggingface.co/sensenova/SenseNova-RoboRSI-27B) |
| SenseNova-RoboRSI-27B-GGUF | [🤗 GGUF weights & llama.cpp guide](https://huggingface.co/sensenova/SenseNova-RoboRSI-27B-GGUF) |
| SenseNova-RoboRSI-9B | Coming Soon |

[Results & Demos](#results--demos) · [Quickstart](#quickstart)

## Quickstart

### Install vLLM

This example uses four NVIDIA GPUs with BF16 support on x86-64 Linux. In your inference environment, install the CUDA 12.9 build of vLLM 0.21.0 with a compatible NVIDIA driver. For other CUDA builds, see the [vLLM installation guide](https://docs.vllm.ai/en/v0.21.0/getting_started/installation/gpu/).

```bash
python -m pip install 'vllm==0.21.0+cu129' huggingface_hub \
  --extra-index-url https://wheels.vllm.ai/0.21.0/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129
```

### Download the model

Download the release weights, tokenizer, visual processor and chat template. For a private repository, first run `hf auth login` with an account that has access. If you already have the complete checkpoint, set `MODEL_DIR` to its absolute path and continue to the next step. [🤗 Model Weights](https://huggingface.co/sensenova/SenseNova-RoboRSI-27B/tree/main)

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

### RoboDojo reproduction

Prepare and start the Benchmark service following the [RoboDojo setup guide](../benchmarks/robodojo/README.md#setup). With the model connection configured above, run from the repository root:

```bash
python scripts/evaluation/evaluate_model.py --benchmark robodojo \
  --benchmark-url robodojo=http://127.0.0.1:18800 \
  --run-id roborsi-27b-robodojo-001
```

Replace the Benchmark URL with your service address. Results are saved under
`work/model_eval/results/roborsi-27b-robodojo-001/`.

## License

SenseNova-RoboRSI-27B is released under the Apache License 2.0.

[teaser]: ../assets/roborsi-27b-teaser.png
[results]: ../assets/roborsi-27b/en/robodojo-results.png
[loop]: ../assets/roborsi-27b/en/rsi-loop.png
[control]: ../assets/roborsi-27b/en/code-as-policy.png
[bowl]: ../assets/roborsi-27b/en/bowl-stacking.webp
[accuracy]: ../assets/roborsi-27b/en/eef-accuracy.png
[comparison]: https://huggingface.co/sensenova/SenseNova-RoboRSI-27B/resolve/main/assets/roborsi-27b/en/vla2agent-comparison.webp?v=6414a12b1bed
[trials]: ../assets/roborsi-27b/en/real-robots.webp
[recovery]: ../assets/roborsi-27b/en/real-robot-recovery.png
[cube]: ../assets/roborsi-27b/en/dual-arm-cube.webp
