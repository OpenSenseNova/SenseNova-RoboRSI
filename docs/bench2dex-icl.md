# Prepare ICL and evaluate all 26 Bench2Dex tasks

> Source: the same path at upstream `8d9a0af66422e1faad7696281a76819cd165eb44`. Original tools, versions and workflow are preserved below. Historical release/validation statements are not results of this port. See the [source audit](icl-coverage.md).

[中文](bench2dex-icl.zh-CN.md) · [Evaluate with released ICL](../benchmarks/bench2dex/ICL.md)

Prepare ICL for **all 26 tasks**, then test each at indices **5, 6 and 7**:
**78 episodes** with standard **1× action budgets**. ICL preparation asks the model
to study demonstrations and save task notes and reference material for testing.

You need about **406 GB of demonstrations**, plus scenes, the simulator and
recordings—not the released ICL ZIPs. Preparation adds model calls with no fixed
time or token limit. To skip preparation, [use released ICL](../benchmarks/bench2dex/ICL.md).

Both stages use the original evaluator bundled with RoboRSI. Use the commands
here for newly generated packages: the shared `bench eval` preset only accepts
the published package hashes.

## 1. Set up the working folder

Use Ubuntu 22.04 on x86-64, a compatible NVIDIA RTX GPU/driver, Git and Conda.
The simulator is Isaac Sim 5.1 with Python 3.11; see its
[hardware requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html).
You need working Linux user namespaces for `bubblewrap`, and access to
`gpt-6-astra` through Codex. Docker and registry access are not needed here.

Open a fresh Bash terminal. If you do not already have RoboRSI:

```bash
git clone https://github.com/SenseTime-FVG/RoboRSI.git
cd RoboRSI
git checkout 67fe9bc6daf985e438be0396ef4206a7f98b4d6b
```

From that RoboRSI checkout, extract the evaluator and get its pinned dependencies:

```bash
export BENCH2DEX_STUDY="$PWD/work/icl-from-scratch"
python3 -m zipfile -e agents/bench2dex_compat/reference-code.zip "$BENCH2DEX_STUDY"
git clone https://github.com/Bench2Dex/Bench2Dex.git "$BENCH2DEX_STUDY/Bench2Dex"
git -C "$BENCH2DEX_STUDY/Bench2Dex" checkout f96a8b2b4eb475483af66e9e03916b35bc43f1be
git clone https://github.com/isaac-sim/IsaacLab.git "$BENCH2DEX_STUDY/IsaacLab"
git -C "$BENCH2DEX_STUDY/IsaacLab" checkout 37ddf626871758333d6ed89cf64ad702aef127d0
```

Do not modify the Bench2Dex or Isaac Lab checkouts. Use a new `BENCH2DEX_STUDY`
folder for each installation. Keep this terminal open to retain its environment variables.

## 2. Install the simulator and model tools

Install the libraries used inside the agent's isolated system Python:

```bash
sudo apt-get update
sudo apt-get install bubblewrap python3-numpy python3-scipy python3-pil python3-yaml python3-opencv
```

Create a separate environment and install the pinned simulator stack:

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

**All remaining commands run from `bench2dex-eval/` in this environment.**
Select the installed Codex binary and sign in:

```bash
export PATH="$(dirname "$(python -c 'from codex_cli_bin import bundled_codex_path; print(bundled_codex_path())')"):$PATH"
codex --version
codex login
```

The version must read `codex-cli 0.154.0`. Both learning and testing use
`gpt-6-astra` with `max` reasoning effort.

## 3. Download all scenes and demonstrations

Set the runtime paths. The `dex2bench_dataset` folder must sit beside the
Bench2Dex checkout because some scene files use relative paths:

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

The scripts download and verify these versions:

| Data | Dataset | Revision |
| --- | --- | --- |
| Scenes and robot assets | `Bench2Dex/Assets` | `bf65215f844d1fca30750e4cc70d7266650abf0a` |
| Training demonstrations | `Bench2Dex/teleopdata` | `b195787c65046083e6a43776b67bdf1389dfa3eb` |

The downloader fetches all demonstrations and resumes interrupted downloads.
Allow **406 GB + at least 20 GiB spare**, plus about 13.6 GB of scenes, the
simulator, preparation outputs and 78 test recordings.

Complete downloads made by these scripts can be reused. Keep `.download_logs/`:
its inventories are required for verification. HDF5 files alone are not enough.

## 4. Fix the 26-task, 78-episode plan

Create a new run folder and set the task list once:

```bash
export EVAL_RUNS="$PWD/runs/bench2dex-78"
mkdir -p runs
mkdir "$EVAL_RUNS"
TASKS=(3 6 7 8 9 12 21 22 24 26 27 32 34 42 43 44 51 60 61 62 64 67 73 76 79 80)
```

Generate the plan from the recorded schedule. Keep the base config's Full/unseen
split, four cameras, seeds, model and task budgets; select three episodes per task:

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

The suite first runs short simulator checks at index 1, without model calls.
These do not count toward the 78 scored episodes or change their action budgets.

After accepting the Isaac Sim license through its installation flow, check
the environment for every task:

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

Every preflight must report `passed: true`; the doctor must report
`tool_probe: passed` and `action_transport_probe: passed`. These checks do not
make model calls or verify account entitlement. Resolve failures before
starting preparation.

## 5. Prepare ICL for every task

The loop below selects the 100 pinned training demonstrations for each task,
then asks the model to study them and save a package. **This starts model
inference for 26 separate preparations**; there is no fixed token or time cap.

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

Task 51 needs the explicit legacy-provenance flag because its original
demonstrations lack known seed/collection provenance. The tool checks the
specific permitted files and records that exception; it does not turn off
verification. Task 62's historical scene-name alias is checked automatically.
Preserve these records when reporting a new run.

A finished preparation has `state: ready` in
`$EVAL_RUNS/preparation/task-ID/preparation.json` and a package under
`frozen/`. Its `frozen/payload/TASK.md` contains the task notes.
The loop stops on the first error. Do not proceed with missing preparations
or overwrite a completed package.

Combine **all 26** packages into the test configuration:

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

The config command verifies each preparation and its package hashes. Each
task's package is now fixed and must remain unchanged across testing.

## 6. Run the full 78-episode evaluation

The suite checks assets, scene dependencies and simulator setup, then runs
one episode at a time, each with a fresh simulator and model session:

```bash
python -u -m gpt6_eval.suite \
  --plan "$EVAL_RUNS/plan.json" --config "$EVAL_RUNS/icl-config.json" \
  --upstream "$BENCH2DEX_ROOT" --isaaclab "$ISAACLAB_ROOT" \
  --assets "$BENCH2DEX_ASSETS" --python "$ISAACLAB_PYTHON" \
  --inventory "$BENCH2DEX_DOWNLOAD_LOGS/selection.inventory.json" \
  --output "$EVAL_RUNS/evaluation"
```

This makes model calls for all 78 scored episodes and may take days.
Keep the terminal alive and monitor disk space and account usage. Stage logs
are in `$EVAL_RUNS/evaluation/logs/`.

To check progress from a second terminal, activate the same Conda environment,
enter `bench2dex-eval/`, and run:

```bash
python -m gpt6_eval.monitor status runs/bench2dex-78/evaluation
```

## 7. Read the complete result

The suite writes:

- `evaluation/suite_result.json`: overall score and completion status;
- `evaluation/task_results.csv`: one row per task;
- `evaluation/evaluation/task_XX/result.json`: each task's three-episode results;
- per-episode `result.json` and `telemetry.json`: native outcomes, model usage and timing.

From the first terminal, check that all 78 episodes are valid before printing
the final score:

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

Task failures count as zero, not as missing data. An infrastructure error or
unfinished episode prevents a complete-suite score. Never drop a missing task
or repeat a valid scored failure to improve the result.

The [recorded 1× result](https://github.com/SenseTime-FVG/RoboRSI/blob/8d9a0af66422e1faad7696281a76819cd165eb44/docs/evaluations.md#which-result-to-compare) is 20/78
(25.64%) with the released ICL. Newly generated ICL will not be byte-identical
to those packages, so report this as a **new preparation + evaluation run**,
including its own score. This is the full standard-budget workflow, not a
replay of the original inference or the separate 10× continuation study.

Keep `base.json`, `plan.json`, `icl-config.json`, all 26 frozen packages,
preparation records, and the whole evaluation folder. Report preparation usage
separately from test usage: preparing one package serves three test episodes,
so do not charge its preparation cost three times.

## Interrupted runs and repeats

Neither the preparation loop nor the suite silently resumes an existing output.
If a stage fails, read its log and preserve the partial output. Completed
training/preparation directories can be retained; any replacement needs a new
output path and a newly generated config that explicitly selects it.

To test the same 26 packages again, reuse `icl-config.json` and `plan.json`
but give the suite a new output such as `$EVAL_RUNS/evaluation-repeat-1`.
Do not run ICL again unless you intend to measure a new preparation.
Keep independent runs separate; do not select the best episode from each.

Raw outputs include private model-session material. Review and sanitize them
before publication; do not publish authentication files or entire local Codex
directories.
