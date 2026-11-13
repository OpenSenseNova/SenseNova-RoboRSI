# 为全部 26 个 Bench2Dex 任务准备 ICL 并评测

> 来源：旧仓库 `8d9a0af66422e1faad7696281a76819cd165eb44` 的同名文档。以下保留原始工具、版本和流程；其中历史发布与验证记录不代表本次移植已运行模型或仿真器。来源哈希和兼容修改见 [移植清单](icl-coverage.zh-CN.md).

[English](bench2dex-icl.md) · [使用已发布的 ICL 评测](evaluation.zh-CN.md)

为**全部 26 个任务**准备 ICL，再分别测试索引 **5、6、7**，
共 **78 个回合**，使用标准 **1× 动作预算**。
ICL 准备会让模型学习演示数据，保存测试时使用的任务笔记和参考资料。

需要约 **406 GB 的演示数据**，另加场景、模拟器和记录空间，不需要已发布的 ICL ZIP。
准备阶段会额外调用模型，没有固定时间或 token 上限。
如需跳过准备阶段，请[使用已发布的 ICL](evaluation.zh-CN.md)。

两个阶段均使用 RoboRSI 附带的原始评测器。新生成的包请使用本页命令加载；
共享的 `bench eval` 预设只接受已发布包的哈希。

## 1. 准备工作目录

使用 x86-64 的 Ubuntu 22.04、兼容的 NVIDIA RTX GPU 和驱动、Git 及 Conda。
模拟器使用 Isaac Sim 5.1 和 Python 3.11，参见其
[硬件要求](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)。
Linux 用户命名空间需允许 `bubblewrap` 正常工作；账号需能通过 Codex 使用
`gpt-6-astra`。这条路径不需要 Docker 或容器仓库权限。

打开新的 Bash 终端。如果还没有 RoboRSI：

```bash
git clone https://github.com/SenseTime-FVG/RoboRSI.git
cd RoboRSI
git checkout 67fe9bc6daf985e438be0396ef4206a7f98b4d6b
```

在该 RoboRSI 目录中，解压评测工具并下载指定版本的依赖源码：

```bash
export BENCH2DEX_STUDY="$PWD/work/icl-from-scratch"
python3 -m zipfile -e agents/bench2dex_compat/reference-code.zip "$BENCH2DEX_STUDY"
git clone https://github.com/Bench2Dex/Bench2Dex.git "$BENCH2DEX_STUDY/Bench2Dex"
git -C "$BENCH2DEX_STUDY/Bench2Dex" checkout f96a8b2b4eb475483af66e9e03916b35bc43f1be
git clone https://github.com/isaac-sim/IsaacLab.git "$BENCH2DEX_STUDY/IsaacLab"
git -C "$BENCH2DEX_STUDY/IsaacLab" checkout 37ddf626871758333d6ed89cf64ad702aef127d0
```

不要修改 Bench2Dex 和 Isaac Lab 源码。每次安装使用新的 `BENCH2DEX_STUDY`
目录；保持当前终端打开，以保留环境变量。

## 2. 安装模拟器和模型工具

安装 Agent 隔离环境内系统 Python 所需的库：

```bash
sudo apt-get update
sudo apt-get install bubblewrap python3-numpy python3-scipy python3-pil python3-yaml python3-opencv
```

创建独立环境，并安装指定版本的模拟器组件：

```bash
unset PYTHONPATH
export PYTHONNOUSERSITE=1
conda create -n bench2dex-icl python=3.11 -y
conda activate bench2dex-icl
python -m pip install --upgrade 'pip>=23' 'setuptools>=68'
python -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e "$BENCH2DEX_STUDY/IsaacLab/source/isaaclab"
cd "$BENCH2DEX_STUDY/bench2dex-eval"
python -m pip install -e '.[data,test]'
python -m pip install Flask==3.1.2 openai-codex-cli-bin==0.154.0
python -m pip check
```

**后续命令都在这个环境下的 `bench2dex-eval/` 目录运行。**
选择刚安装的 Codex 二进制文件并登录：

```bash
export PATH="$(dirname "$(python -c 'from codex_cli_bin import bundled_codex_path; print(bundled_codex_path())')"):$PATH"
codex --version
codex login
```

版本必须为 `codex-cli 0.154.0`。学习和测试阶段都使用
`gpt-6-astra`，推理强度为 `max`。

## 3. 下载全部场景和演示数据

设置运行路径。部分场景使用相对路径，所以 `dex2bench_dataset` 必须与
Bench2Dex 源码目录处于同一级：

```bash
export BENCH2DEX_ROOT="$BENCH2DEX_STUDY/Bench2Dex"
export ISAACLAB_ROOT="$BENCH2DEX_STUDY/IsaacLab"
export ISAACLAB_PYTHON="$(command -v python)"
export BENCH2DEX_ASSETS="$BENCH2DEX_STUDY/dex2bench_dataset"
export BENCH2DEX_DEMOS="$BENCH2DEX_STUDY/teleopdata"
export BENCH2DEX_DOWNLOAD_LOGS="$BENCH2DEX_ASSETS/.download_logs/all_tasks"
bash scripts/download_all_assets.sh
python scripts/download_demonstrations.py --output "$BENCH2DEX_DEMOS" --workers 2
```

脚本会下载并校验以下版本：

| 数据 | 数据集 | 版本 |
| --- | --- | --- |
| 场景和机器人资源 | `Bench2Dex/Assets` | `bf65215f844d1fca30750e4cc70d7266650abf0a` |
| 训练演示 | `Bench2Dex/teleopdata` | `b195787c65046083e6a43776b67bdf1389dfa3eb` |

下载器获取全部演示，支持断点续传。需要 **406 GB + 至少 20 GiB 余量**，
另加约 13.6 GB 的场景、模拟器、准备输出和 78 个测试回合的记录空间。

这些脚本完成的下载可复用。请保留 `.download_logs/`：校验需要其中的清单，
只有 HDF5 文件不够。

## 4. 固定 26 个任务、78 个回合的计划

创建新的运行目录，并一次性设定任务列表：

```bash
export EVAL_RUNS="$PWD/runs/bench2dex-78"
mkdir -p runs
mkdir "$EVAL_RUNS"
TASKS=(3 6 7 8 9 12 21 22 24 26 27 32 34 42 43 44 51 60 61 62 64 67 73 76 79 80)
```

从记录的日程生成计划，保留 Full/unseen、四路相机、种子、模型和原始任务预算，
将每个任务的测试回合数设为三个：

```bash
python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["EVAL_RUNS"])
schedule = json.loads(Path("configs/historical_78_schedule.json").read_text())
task_ids = sorted({slot["task_id"] for slot in schedule["slots"]})
assert len(task_ids) == 26 and schedule["episode_indices"] == [5, 6, 7]
assert {(slot["task_id"], slot["episode_index"]) for slot in schedule["slots"]} == {
    (task, episode) for task in task_ids for episode in [5, 6, 7]
}
config = json.loads(Path("configs/icl_base.json").read_text())
config.update(start_episode=5, num_episodes=3)
plan = {
    "task_ids": task_ids,
    "task_names": [f"task_{task:02d}" for task in task_ids],
    "workers": 1,
    "episodes_per_task": 3,
    "episode_indices": [5, 6, 7],
    "smoke_episode_index": 1,
    "smoke_steps": 4,
}
for filename, document in [("base.json", config), ("plan.json", plan)]:
    with (root / filename).open("x") as stream:
        json.dump(document, stream, indent=2)
        stream.write("\n")
print("Plan saved: 26 tasks × 3 episodes = 78 scored episodes")
PY
```

运行器先在索引 1 做不调用模型的短检查，不计入正式的 78 个回合，
也不改变测试的动作预算。

通过 Isaac Sim 的安装流程接受其许可后，检查每个任务的运行环境：

```bash
export OMNI_KIT_ACCEPT_EULA=YES
(
  set -euo pipefail
  for task in "${TASKS[@]}"; do
    python -m gpt6_eval.preflight --task "$task" \
      --config "$EVAL_RUNS/base.json" --require-model
  done
  python -m gpt6_eval.doctor --config "$EVAL_RUNS/base.json" \
    --output "$EVAL_RUNS/environment-check"
)
```

每次 preflight 都应输出 `passed: true`；doctor 应输出
`tool_probe: passed` 和 `action_transport_probe: passed`。
这些检查不调用模型，也不验证账号的模型使用权限。请先解决错误再开始准备。

## 5. 为每个任务准备 ICL

下面的循环为每个任务选取固定版本的 100 个训练演示文件，
然后让模型学习并保存一个包。**此处开始 26 次独立准备的模型推理**，
没有固定 token 或时间上限。

```bash
(
  set -euo pipefail
  for task in "${TASKS[@]}"; do
    extra_args=()
    if [ "$task" = 51 ]; then
      extra_args+=(--task51-legacy-provenance)
    fi
    python -m gpt6_eval.icl prepare \
      --config "$EVAL_RUNS/base.json" --task "$task" \
      --upstream "$BENCH2DEX_ROOT" --assets "$BENCH2DEX_ASSETS" \
      --dataset "$BENCH2DEX_DEMOS" --access-mode trusted_reader_v1 \
      "${extra_args[@]}" --output "$EVAL_RUNS/training/task-$task"
    python -m gpt6_eval.icl explore \
      --config "$EVAL_RUNS/base.json" \
      --training "$EVAL_RUNS/training/task-$task" \
      --output "$EVAL_RUNS/preparation/task-$task"
  done
)
```

任务 51 的原始演示缺少已知的种子和采集来源信息，因此需要显式传入
legacy-provenance 参数。工具仍会核验具体允许使用的文件，并记录这一例外，
不会关闭数据校验。任务 62 的历史场景名称别名会自动核验。
报告新实验时需保留这些记录。

每次成功准备后，`$EVAL_RUNS/preparation/task-ID/preparation.json`
中会有 `state: ready`，包保存在 `frozen/`，任务笔记在
`frozen/payload/TASK.md` 中。
循环遇到第一个错误就会停止。不要在有任务尚未准备完成时开始测试，
也不要覆盖已经完成的包。

将**全部 26 个包**合并到测试配置：

```bash
(
  set -euo pipefail
  preparation_args=()
  for task in "${TASKS[@]}"; do
    preparation_args+=(--preparation "$EVAL_RUNS/preparation/task-$task")
  done
  python -m gpt6_eval.icl config --config "$EVAL_RUNS/base.json" \
    "${preparation_args[@]}" --output "$EVAL_RUNS/icl-config.json"
)
python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["EVAL_RUNS"])
config = json.loads((root / "icl-config.json").read_text())
plan = json.loads((root / "plan.json").read_text())
assert set(config["task_packages"]) == {str(task) for task in plan["task_ids"]}
assert config["start_episode"] == 5 and config["num_episodes"] == 3
print("All 26 ICL packages are included; ready for 78 test episodes")
PY
```

配置命令会校验各任务的准备状态和包哈希。至此，每个任务的包已经固定，
测试期间不应再修改。

## 6. 运行完整的 78 回合评测

运行器检查资源、场景依赖和模拟器后，逐个运行测试回合；
每个回合使用新的模拟器和模型会话：

```bash
python -u -m gpt6_eval.suite \
  --plan "$EVAL_RUNS/plan.json" --config "$EVAL_RUNS/icl-config.json" \
  --upstream "$BENCH2DEX_ROOT" --isaaclab "$ISAACLAB_ROOT" \
  --assets "$BENCH2DEX_ASSETS" --python "$ISAACLAB_PYTHON" \
  --inventory "$BENCH2DEX_DOWNLOAD_LOGS/selection.inventory.json" \
  --output "$EVAL_RUNS/evaluation"
```

此命令为全部 78 个正式回合调用模型，可能需要数天。请保持终端运行，
关注磁盘空间和账号用量。各阶段日志位于 `$EVAL_RUNS/evaluation/logs/`。

如需从第二个终端查看进度，激活相同的 Conda 环境，进入 `bench2dex-eval/`，
再运行：

```bash
python -m gpt6_eval.monitor status runs/bench2dex-78/evaluation
```

## 7. 读取完整结果

整套评测会写出：

- `evaluation/suite_result.json`：总体得分和完成状态；
- `evaluation/task_results.csv`：每个任务一行；
- `evaluation/evaluation/task_XX/result.json`：每个任务的三个回合结果；
- 逐回合 `result.json` 和 `telemetry.json`：原生评分、模型用量和耗时。

在第一个终端中，先检查全部 78 个回合是否有效，再打印最终成功率：

```bash
python - <<'PY'
import json
import os
from pathlib import Path

report = json.loads((Path(os.environ["EVAL_RUNS"]) / "evaluation/suite_result.json").read_text())
assert report["state"] == "completed" and report["eligible_for_gpt6_score"], report
assert report["planned_episodes"] == report["scored_episodes"] == 78, report
assert report["infrastructure_errors"] == 0 and len(report["tasks"]) == 26, report
print(f"Overall: {report['successes']}/78 = {100 * report['success_rate']:.2f}%")
for task in report["tasks"]:
    assert task["complete_valid"] and task["planned"] == task["scored"] == 3, task
    print(f"Task {task['task_id']:02d}: {task['successes']}/3")
PY
```

任务失败是有效的零分结果，不是缺失数据。基础设施错误或未完成回合会阻止
完整成功率的输出。不要去掉缺失任务，也不要重试已经得到有效失败结果的回合来提高得分。

使用发布 ICL 的[历史 1× 结果](evaluation.zh-CN.md#应该对比哪个结果)为
20/78（25.64%）。重新准备的 ICL 不会与发布包逐字节相同，
因此应报告为一次**新的准备与评测实验**，并给出它自己的得分。
这里运行的是完整的标准预算流程，不是重放原始推理，也不是另一项 10× 续跑实验。

保留 `base.json`、`plan.json`、`icl-config.json`、全部 26 个冻结包、
准备记录和整个评测目录。准备阶段用量与测试用量应分开统计：
一个包的准备供三个测试回合共用，不要将准备成本重复计三次。

## 中断和复测

准备循环和整套评测都不会自动恢复已有输出。如果某阶段失败，先阅读日志并保留
部分输出。已经完成的训练数据准备和 ICL 目录可以保留；需要替换的输出应使用
新路径，再生成明确选择这些包的新配置。

若只是用同一批 26 个包复测，复用 `icl-config.json` 和 `plan.json`，
但为整套评测换一个输出目录，例如 `$EVAL_RUNS/evaluation-repeat-1`。
除非要测量一次新的准备过程，否则无需重新运行 ICL。
不同次运行要分开报告，不要从多次运行中挑选各回合的最好成绩。

原始输出包含私有模型会话资料。发布前请检查并脱敏，不要发布认证文件或整个本地
Codex 目录。
