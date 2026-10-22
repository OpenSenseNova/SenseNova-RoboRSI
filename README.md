<h1 align="center">SenseNova-RoboRSI: Recursive Self-Improvement for Embodied Agents</h1>

<p align="center"><strong>Build, evaluate, and improve embodied agents with a shared runtime.</strong></p>

<p align="center">
  <img src="assets/badges/python.svg" alt="Python 3.10–3.12" />
  <img src="assets/badges/license.svg" alt="License: Apache 2.0" />
  <img src="assets/badges/status.svg" alt="Status: Pre-Alpha" />
</p>

<p align="center">
  <a href="https://www.sensenova.ai/has-gpt-6-astra-achieved-embodied-intelligence"><img src="assets/badges/report.svg" alt="Report: Has GPT-6 Achieved Embodied Intelligence?" /></a>
  <a href="https://huggingface.co/sensenova/sensenova-roborsi-27b-preview"><img src="assets/badges/model.svg" alt="Model: SenseNova-RoboRSI-27B-Preview" /></a>
</p>

<p align="center">
  <strong>English</strong> · <a href="README.zh-CN.md">简体中文</a> ·
  <a href="#quick-start">Quick Start</a> · <a href="docs/README.md">Docs</a> ·
  <a href="https://www.sensenova.ai/has-gpt-6-astra-achieved-embodied-intelligence">Report</a> ·
  <a href="#benchzoo">BenchZoo</a> ·
  <a href="#harness-results">Harness Results</a> ·
  <a href="#sensenova-roborsi-27b-preview">RoboRSI-27B-Preview</a> ·
  <a href="#release-plan--roadmap">Release plan</a>
</p>

## News

- **2026-10-22** — **SenseNova-RoboRSI-27B-Preview** now runs across different environments, including on real robots and in simulation! [🤗 Model weights](https://huggingface.co/sensenova/sensenova-roborsi-27b-preview) and [usage guides](docs/sensenova-roborsi-27b-preview.md) are now available. The technical report is coming soon.

- **2026-10-22** — Initial code and usage guides for [**RoboCasa365 Target50**](benchmarks/robocasa365/README.md) are available in this branch.

- **2026-10-21** — Added **benchmark images and ICL experience packages** to [BenchZoo](#benchzoo), including image references and package download links.

- **2026-10-20** — Initial code and usage guides for [**BEHAVIOR-1K**](benchmarks/behavior_1k/README.md), [**Bench2Dex**](benchmarks/bench2dex/README.md) and [**DexJoCo**](benchmarks/dexjoco/README.md) are available in this branch.

- **2026-10-16** — Initial code and usage guides for [**LIBERO**](benchmarks/libero/README.md), [**LIBERO-PRO**](benchmarks/libero_pro/README.md), [**LIBERO-Long++**](benchmarks/libero_lilo/README.md) and [**LIBERO-Ultra-Long**](benchmarks/libero_lilo/README.md) are available in this branch. See the [getting started guide](docs/getting-started.md).

- **2026-10-16** — SenseNova-RoboRSI has been used in the empirical report [**Has GPT-6 Achieved Embodied Intelligence?**](https://www.sensenova.ai/has-gpt-6-astra-achieved-embodied-intelligence).

- **2026-10-09** — SenseNova-RoboRSI achieves an **average score of 60.08** and a **success rate of 54.17%** on RoboDojo, a gain of **+31.11 points** over the GPT-6 Astra baseline (**28.97**). Results are being submitted to the official RoboDojo leaderboard. See [Harness Results](#harness-results) for benchmark comparisons.

## Contributors

- **Project leads:** Hanming Deng, Zhongang Cai, Haiwen Diao, Zhi Hou, Zhonghua Wu, Qingyi Tao
- **Model team:** Silei Wu\*, Weiye Xu, Bo Liu, Pengfei Liu, Tailai Chen, Chenjing Ding, Jiahao Wang, Changxu Wei, Zongpu Zhang
- **Harness team:** Jiapeng Luo\*, Yuyao Shao, Linjun Dai, Quanzhou Li, Ziyi Shang, Wanqi Yin, Xiaowan Li, Donghao Wu, Tianbo Zhong
- **Advisor:** Dahua Lin, Xiaogang Wang, Ziwei Liu, Lewei Lu, Lei Yang

\* denotes team leads.

## Overview

Connect your model, skills, and robot tools through a shared runtime, customize
prompts and context, and evaluate policies in environments that retain their
native execution and scoring logic.

On top of this runtime, **recursive self-improvement (RSI)** uses execution
traces and task feedback to improve the Harness while keeping model parameters
fixed. Candidate changes are retested, and validated improvements are carried
into the next agent version, connecting task execution with continuous improvement.

🎉 **[SenseNova-RoboRSI-27B-Preview](#sensenova-roborsi-27b-preview)** extends this work to model training. Validated
interaction experience becomes multimodal supervision for goal understanding,
action decisions, and feedback-based correction. Updated models return to the
same runtime for evaluation and further exploration, connecting model training
and Harness improvement through shared experience and joint retesting.

<p align="center">
  <img src="assets/sensenova-roborsi-overview-en.webp" alt="SenseNova-RoboRSI agent interfaces, execution loop, benchmark integration, and RSI workflow" width="75%" />
</p>

### Core capabilities

#### Harness: A shared runtime

- **Flexible agent interfaces.** Connect models, prompts, skills, and tools/MCP;
  configure context and caching to suit each task.
- **Unified execution and evaluation.** Run VLA skills, scripted actions, or
  end-effector tools across benchmarks with native feedback and scoring.
- **Experiment records and training data.** Record full traces, export training
  episodes, and run batch evaluations with concurrency and recovery.

#### RSI: Continuous Harness improvement

- **Learn from execution feedback.** Refine action prediction, planning,
  feedback, and memory organization while keeping model parameters fixed.
- **Retest and retain improvements.** Track agent versions, evaluate candidate
  changes under the same task conditions, and carry validated improvements forward.

## Quick Start

Use Python 3.10–3.12 for the Agent and evaluator. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[runtime,test]'
bench --help
```

On Windows, activate with `.venv\Scripts\Activate.ps1`.

Follow the [installation and first evaluation guide](docs/getting-started.md)
to prepare a benchmark and run a fixed-action or model Agent. Native benchmark
images target Linux; connecting to an existing HTTP benchmark requires no local
simulator installation.

- **Benchmark setup:** [LIBERO](benchmarks/libero/README.md),
  [LIBERO-PRO](benchmarks/libero_pro/README.md),
  [LIBERO-Long++ / LIBERO-Ultra-Long](benchmarks/libero_lilo/README.md),
  [Bench2Dex](benchmarks/bench2dex/README.md),
  [DexJoCo](benchmarks/dexjoco/README.md),
  [BEHAVIOR-1K](benchmarks/behavior_1k/README.md), and
  [RoboCasa365](benchmarks/robocasa365/README.md).
- **Agent configuration:** [Models, prompts, context and budgets](docs/agents.md).
- **Evaluation:** [Batch execution, results and resume](docs/evaluation.md).

To deploy SenseNova-RoboRSI-27B-Preview, follow the
[model deployment instructions](docs/sensenova-roborsi-27b-preview.md#1-deploy-the-model).

## BenchZoo

Available benchmark integrations and upcoming releases. Supported services share
configurable EmbodiedAgent presets; select your model and follow the corresponding
benchmark directory for setup and evaluation.

ICL experience packages are hosted in
[SenseNova-RoboRSI-Packages](https://huggingface.co/datasets/sensenova/SenseNova-RoboRSI-Packages).
The links below open each released benchmark's versioned package directory;
follow the package documentation for download, verification and usage.

Prebuilt images are hosted on GHCR. Use `docker pull <image>` with the
corresponding image reference from the table.

| Benchmark | Status | ICL experience package | Docker image |
| --- | --- | --- | --- |
| RoboDojo | Coming soon | — | — |
| [RoboCasa365](benchmarks/robocasa365/) | Released | [v1.0.0](https://huggingface.co/datasets/sensenova/SenseNova-RoboRSI-Packages/tree/main/robocasa365/v1.0.0) | [Build guide](benchmarks/robocasa365/BUILD.md) |
| [LIBERO-Pro](benchmarks/libero_pro/) | Released | [v1.0.0](https://huggingface.co/datasets/sensenova/SenseNova-RoboRSI-Packages/tree/main/libero-pro/v1.0.0) | `ghcr.io/opensensenova/sensenova-roborsi-libero-pro@sha256:a83417f1459a4c555c457caa82ef00a5de50191e3d8deb943894fa9285cff172` |
| [LIBERO-Long++](benchmarks/libero_lilo/) | Released | [v1.0.0](https://huggingface.co/datasets/sensenova/SenseNova-RoboRSI-Packages/tree/main/libero-long-plus-plus/v1.0.0) | `ghcr.io/opensensenova/sensenova-roborsi-libero-long-plus-plus@sha256:814139261e914b114cb89a359c6a39d4f409a556f79928e986b6ade3dfe7dc7a` |
| [LIBERO-Ultra-Long](benchmarks/libero_lilo/) | Released | [v1.0.0](https://huggingface.co/datasets/sensenova/SenseNova-RoboRSI-Packages/tree/main/libero-ultra-long/v1.0.0) | `ghcr.io/opensensenova/sensenova-roborsi-libero-ultra-long@sha256:814139261e914b114cb89a359c6a39d4f409a556f79928e986b6ade3dfe7dc7a` |
| RoboLab | Coming soon | — | — |
| [Bench2Dex](benchmarks/bench2dex/) | Released | [v1.0.0](https://huggingface.co/datasets/sensenova/SenseNova-RoboRSI-Packages/tree/main/bench2dex/v1.0.0) | `ghcr.io/opensensenova/sensenova-roborsi-bench2dex@sha256:ba5253aa91e8e6888f0f0543626b9ace9c76dbe8559d7092caaa160e9b634a76` |
| [DexJoCo](benchmarks/dexjoco/) | Released | [v1.0.0](https://huggingface.co/datasets/sensenova/SenseNova-RoboRSI-Packages/tree/main/dexjoco/v1.0.0) | `ghcr.io/opensensenova/sensenova-roborsi-dexjoco@sha256:19fa30540e3956ae460efa0a4b473586637cb330d863ea65ba0fd6318c376d4d` |
| LHVLN | Coming soon | — | — |
| VLN-CE | Coming soon | — | — |
| SIMPLE | Coming soon | — | — |
| [BEHAVIOR-1K](benchmarks/behavior_1k/) | Released | [v1.0.0](https://huggingface.co/datasets/sensenova/SenseNova-RoboRSI-Packages/tree/main/behavior/v1.0.0) | `ghcr.io/opensensenova/sensenova-roborsi-behavior-1k:v1.0.0-cu128` |

## Harness Results

Each table shows SenseNova-RoboRSI alongside selected reference results. Higher is better.

### [RoboDojo](https://robodojo-benchmark.com/)

On RoboDojo-Sim, SenseNova-RoboRSI achieves an **average score of 60.08** and a
**success rate of 54.17%**. Compared with the **GPT-6-Astra** baseline, this is a
gain of **31.11 score points** and **31.69 percentage points** in success rate.

**Leaderboard submission:** Our results are being submitted to the official RoboDojo leaderboard. The official results may vary slightly; we will update the results here once they are available.

| Method | Average score ↑ | Success rate ↑ |
| --- | ---: | ---: |
| SenseNova-RoboRSI / GPT-6 Astra (ours) | **60.08** | **54.17%** |
| EmbodiedEvo | 45.08 | 38.95% |
| VPP2 | 39.26 | 32.26% |
| PhysicalRSI | 36.27 | 31.38% |
| Awomo-0.5 | 35.34 | 29.64% |
| Simate-beta | 33.95 | 27.96% |
| GPT-6-Astra (baseline) | 28.97 | 22.48% |

Reference results: [RoboDojo simulation leaderboard](https://robodojo-benchmark.com/leaderboard)
(updated **2026-10-14**, accessed **2026-10-15**).

### [LIBERO-PRO](https://zxy-mllab.github.io/LIBERO-PRO-Webpage/)

| Method | Success rate ↑ |
| --- | ---: |
| SenseNova-RoboRSI / GPT-6 Astra (ours) | **94.50%** |
| RPent / GPT-6 Astra | 92.63% |
| RPent / Opus-4.7 | 82.40% |
| RPent / GPT-5.5 | 75.13% |
| RPent Flash Mode / Molmo2-8B | 72.63% |

The reference scores above, including GPT-6 Astra's 92.63%, are reported by [RPent](https://rpent.readthedocs.io/zh-cn/latest/rst_source/leaderboard/performance.html) on Task/Swap suites. The [official LIBERO-PRO leaderboard](https://github.com/Zxy-MLlab/LIBERO-PRO#-libero-pro-model-leaderboard) reports a different aggregate across five perturbation types.

### [RoboCasa365](https://robocasa.ai/)

| Method | Evaluation setting | Success rate ↑ |
| --- | --- | ---: |
| SenseNova-RoboRSI / GPT-6 Astra (ours) | Target kitchens + task memory | **64.60%** |
| Paimon-0 | Pretraining kitchens | 58.10% |
| Xiaomi-Robotics-1 | Pretraining kitchens | 57.40% |
| Phasor-m7 | Pretraining kitchens | 56.80% |
| ProWAM | Pretraining kitchens | 48.10% |

Reference results: [official RoboCasa leaderboard](https://robocasa.ai/leaderboard.html). SenseNova-RoboRSI uses the Target50 setting; the official entries use pretraining kitchens. These results provide context across different evaluation settings and do not establish a ranking under the same protocol.

## SenseNova-RoboRSI-27B-Preview

**SenseNova-RoboRSI-27B-Preview** is an open-weight **27B embodied VLM** that
extends RoboRSI from Harness improvement to model training. Through **joint
model–Harness RSI**, it learns from validated interaction experience, bringing
goal understanding, action decisions, and feedback-based correction into the
model itself.

The model supports **VLA-assisted execution and direct VLM control**, with
evaluations on RoboDojo, LIBERO-Pro, and RoboCasa365, and demonstrations in
simulation and on real robots. The open weights provide a starting point for
community deployment, reproduction, and continued training.

![SenseNova-RoboRSI-27B-Preview results on RoboDojo, LIBERO-Pro, and RoboCasa365](assets/roborsi-27b-preview-benchmarks.webp)

Model weights are available on [🤗 Hugging Face](https://huggingface.co/sensenova/sensenova-roborsi-27b-preview).
See the [model guide](docs/sensenova-roborsi-27b-preview.md) for detailed results, demos, and usage instructions.

## Release Plan & Roadmap

### Release plan

- [x] Project introduction
- [x] [Empirical Study](https://www.sensenova.ai/has-gpt-6-astra-achieved-embodied-intelligence)
- [x] Initial code release: Agent runtime and LIBERO-family evaluation
- [x] [Usage documentation](docs/README.md)
- [x] [SenseNova-RoboRSI-27B-Preview release](docs/sensenova-roborsi-27b-preview.md)
- [ ] Harness demo
- [ ] RoboRSI Technical Report

### Research roadmap

- **Experience retrieval.** We plan to organize successful strategies, failure
  cases, and execution traces into a searchable experience library.
- **Transfer across tasks and environments.** We will investigate whether tool
  combinations, planning strategies, and feedback mechanisms validated in one
  setting can provide useful starting points in another.
- **More robot platforms and real-world scenarios.** We aim to extend integration
  and evaluation to more robot embodiments and physical settings, and examine the
  reliability of feedback-driven improvement in those environments.
- **Joint model–Harness RSI.** Continue converting validated interaction
  experience into training supervision, retest model and system changes together,
  and measure capability and efficiency across improvement rounds.

## Community & Acknowledgements

We welcome collaboration with teams working on foundation models,
VLA policies, simulation platforms, and real robots.

- **Contributions:** We welcome new integrations, evaluation tasks,
  reproducible failure cases, and suggestions for Harness design. See the
  [contribution guide](CONTRIBUTING.md) for the current release scope and checks.
- **Questions & collaboration:** Open an
  [issue](https://github.com/OpenSenseNova/SenseNova-RoboRSI/issues)
  describing your platform, use case, and proposed contribution.
- **Release updates:** Watch the
  [SenseNova-RoboRSI repository](https://github.com/OpenSenseNova/SenseNova-RoboRSI).
- **Acknowledgements:** We thank
  [RoboDojo](https://robodojo-benchmark.com/),
  [LIBERO-PRO](https://zxy-mllab.github.io/LIBERO-PRO-Webpage/),
  [RoboCasa](https://robocasa.ai/),
  [RPent](https://rpent.readthedocs.io/zh-cn/latest/), and the broader open-source
  community supporting embodied-agent research.
- **Licenses:** External models, simulators, and datasets retain their respective
  licenses.

**This achievement reflects the joint team's continued exploration and close collaboration in embodied AI and agent systems. SenseNova-RoboRSI has been used in the published report [Has GPT-6 Achieved Embodied Intelligence?](https://www.sensenova.ai/has-gpt-6-astra-achieved-embodied-intelligence). We will continue to advance and validate these technologies.**
