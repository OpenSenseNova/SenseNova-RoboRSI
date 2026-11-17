# SenseNova-RoboRSI-27B

**Learning Code as Policy through multi-round Agent–Harness co-evolution.**

[简体中文](sensenova-roborsi-27b.zh-CN.md) · [🤗 Hugging Face](https://huggingface.co/sensenova/SenseNova-RoboRSI-27B) · [Demos](#demonstrations) · [Results](#evaluation-results) · [Quickstart](#quickstart)

**Release draft · Model weights: TBD · Technical report: TBD**

SenseNova-RoboRSI-27B is a **27B vision-language model for embodied agents**. Building on RoboRSI Harness and RoboRSI-27B-Preview, it extends robot execution from invoking VLA skills to generating code that computes target poses and controls robot motion. The Agent brings VLA skills and code-driven control into a shared observation–action–feedback loop.

The model learns through **multi-round recursive self-improvement (RSI)**: improve the Harness, collect and select interaction experience, and train the Agent to adapt to the updated execution system. Long-horizon Agent trajectories are combined with local action supervision constructed from existing VLA training data. With no real-robot data used in embodied post-training, the same model weights have also been used for real-robot pick-and-place tasks.

> **Overview figure — TBD**<br>
> Agent–Harness co-evolution, VLA × Code as Policy, and transfer from simulation to real robots.

## Highlights

- **VLA × Code as Policy.** Learn to generate control code and organize VLA skills alongside explicit tool-center-point (TCP) control according to the task and execution state.
- **Multi-round RSI.** Iterate the Harness, then use supervised fine-tuning and rejection-sampled interaction trajectories to adapt the Agent to its evolving tools and execution strategies.
- **More accurate action targets.** Combine long-horizon decision-making with local TCP supervision constructed from existing robot trajectories, connecting visual observations and robot states to executable motion targets.
- **Sim-to-Real with shared weights.** Transfer simulation-trained capabilities to real-robot manipulation using the same weights, with zero real-robot data in embodied post-training.

## Demonstrations

### Real robots

Pick up a toy duck and place it into a container, or retrieve it from the container. Additional tasks and comparisons with Preview will be added here.

> **Real-robot demo — TBD**<br>
> Robot setup, final checkpoint, task instructions, and trial results: TBD.

### VLA × Code as Policy in simulation

Follow an Agent through a complete task: observe the scene, select an execution method, generate control code or invoke a VLA skill, and use the resulting observations to decide the next action.

> **RoboDojo demo — TBD**<br>
> Representative code-control and VLA-use episodes, with selected decisions and tool calls.

### Learning to grasp, place, and insert

Compare local manipulation behavior with and without TCP supervision, including a grasp that lifts the target, stable placement, and insertion followed by gripper release.

> **TCP-supervision comparison — TBD**<br>
> Side-by-side grasping, placement, and insertion clips with matching task descriptions.

## Evaluation Results

### RoboDojo

The release comparison will report average task score and task success rate under a shared evaluation protocol.

| Model | Average score ↑ | Success rate ↑ |
| :--- | ---: | ---: |
| **SenseNova-RoboRSI-27B** | **TBD** | **TBD** |
| SenseNova-RoboRSI-27B-Preview | TBD | TBD |
| Qwen3.8-27B | TBD | TBD |
| DeepSeek-V4.1-Flash-1017 | TBD | TBD |
| PEAR-27B | TBD | TBD |

Task set, seeds/layouts, Harness version, execution backends, inference budgets, and aggregation: **TBD**.

> **Benchmark figure — TBD**<br>
> Final release results and completed comparisons under the same protocol.

### TCP target prediction

The current TCP-supervision study uses **374 local actions across 35 tasks**, including 179 single-arm and 195 dual-arm actions. A prediction is a hit when its TCP endpoint is within the positional-error threshold of the reference endpoint; for dual-arm actions, both arms must satisfy the threshold.

| Training configuration | TCP hit rate at 5 mm ↑ |
| :--- | ---: |
| With local TCP supervision | **15.5%** |
| Without local TCP supervision | 4.3% |

The study reports a **3.6×** hit-rate improvement at 5 mm. Model revisions and the full threshold breakdown: **TBD**. Evaluation of the final release checkpoint: **TBD**.

> **TCP diagnostic figure — TBD**<br>
> Hit rates across positional-error thresholds, with training ablations and model comparisons.

### Real-robot evaluation

| Task | Successful trials / total | Execution setup |
| :--- | :---: | :--- |
| Pick up a duck and place it into a container | TBD | TBD |
| Retrieve a duck from a container | TBD | TBD |
| Additional tasks | TBD | TBD |

Robot platform, camera and control interfaces, trial conditions, and final checkpoint: **TBD**.

### Additional benchmarks

LIBERO-Pro and RoboCasa365 results for the release checkpoint: **TBD**.

## Quickstart

### 1. Download and deploy

Model repository: [`sensenova/SenseNova-RoboRSI-27B`](https://huggingface.co/sensenova/SenseNova-RoboRSI-27B).

Weights and the validated deployment recipe are being prepared. The release will include the model configuration, tokenizer, visual processor, and chat template alongside the weights.

```bash
# Model download command: TBD
# Inference environment and service launch command: TBD
# Vision and tool-call request example: TBD
```

Supported inference backend/version, hardware requirements, context length, generation settings, and recommended reasoning effort: **TBD**.

### 2. Connect RoboRSI Harness and evaluate

Use [RoboRSI Harness](https://github.com/OpenSenseNova/SenseNova-RoboRSI) to connect the model to benchmark environments and robot execution tools. Release-specific configurations and commands will be provided for the available execution modes.

```bash
# RoboDojo — VLA + Code as Policy: TBD
# RoboDojo — direct code/TCP control: TBD
# LIBERO-Pro: TBD
# RoboCasa365: TBD
```

Real-robot integration example: **TBD**.

## How It Learns

### Multi-round Agent–Harness RSI

Each round begins by improving the Harness: its tools, execution strategies, context, and feedback handling. Interaction with the updated Harness produces trajectories that are selected through rejection sampling and used for supervised fine-tuning. The updated Agent then returns to the Harness for further execution and evaluation.

**Improve the Harness → collect and select trajectories → train the Agent → evaluate the joint version → repeat.**

> **RSI learning curves — TBD**<br>
> Round-by-round results, model and Harness versions, and controls that isolate their contributions.

### Learning when and how to act

VLA skills provide learned action execution, while Code as Policy lets the Agent generate code to compute target poses, organize motion, and call TCP control tools. Both receive feedback through the same interaction loop. We study how their use changes across tasks and manipulation stages, and how the learned behavior evolves from the base model and Preview to the release model.

> **Execution analysis — TBD**<br>
> VLA/TCP usage by task and stage, representative switching cases, and comparisons of combined execution with fixed execution modes.

### Long-horizon trajectories and local action supervision

Long-horizon Agent trajectories teach the model to connect observations, decisions, tool calls, and execution feedback. Local action examples teach it to predict the target of an individual movement.

We construct local supervision from existing VLA training trajectories by identifying manipulation segments and pairing their instructions and initial observations with endpoint TCP targets. Each example uses three camera views, the dual-arm state, and a local instruction as input. Later frames help verify the segment annotation; they are excluded from the model input.

The current data construction covers **35 tasks and 3,493 episodes**, producing **77,202 local examples**: 66,359 single-arm and 10,843 dual-arm examples. The final training subset, mixture, and data split will be specified with the release.

> **Data construction figure — TBD**<br>
> A real example of observations + robot state + local instruction → TCP target, and its relationship to long-horizon Agent training.

### From simulation to real robots

The same model weights are used in simulation and real-robot validation, with no real-robot data in embodied post-training. Visual reasoning and code-based control connect the learned task decisions to the robot's execution interface.

Camera calibration, action-coordinate mapping, robot adapters, and complete execution examples: **TBD**.

## Reproduction

| Resource | Release reference |
| :--- | :--- |
| Model checkpoint / revision | TBD |
| Matching Harness commit / tag | TBD |
| Task manifests and evaluation configurations | TBD |
| Per-task results and execution traces | TBD |
| Technical report | TBD |

The earlier release remains available at [SenseNova-RoboRSI-27B-Preview](https://huggingface.co/sensenova/SenseNova-RoboRSI-27B-Preview).

## Contributors

## License

**Apache License 2.0.** The model is based on [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). The weight release will include the upstream license and attribution files. External models, simulators, and datasets retain their respective licenses.
