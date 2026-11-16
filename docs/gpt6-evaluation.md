# Evaluate GPT-6 Astra across the released benchmarks

[简体中文](gpt6-evaluation.zh-CN.md) · [All guides](README.md)

[`evaluate_gpt6.py`](../scripts/evaluation/evaluate_gpt6.py) runs the existing
benchmark Agents in sequence and saves one suite summary. Its default selection
covers all **11 released benchmarks**, with VLN-CE R2R and RxR as separate runs.
RoboDojo and SIMPLE are not released and are not included.

The suite coordinates existing evaluators; it does not change native actions,
stopping rules, scoring, prompts or frozen ICL contracts. It is an evaluation
entry point, not a claim that every published paper result has been reproduced.

## Install and configure once

Use Linux/WSL, Python 3.10–3.12, Docker and the GPU/runtime requirements in each
benchmark guide. A Python 3.11 Runner is needed for the pinned Bench2Dex Agent.
Keep simulator and Agent dependencies separate as their guides require.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[runtime]'
```

Edit [`configs/evaluations/gpt6.yaml`](../configs/evaluations/gpt6.yaml):

- **LIBERO and RoboCasa365:** the existing automatic evaluator prepares images,
  assets and available ICL. Its GHCR/HF access and GPU requirements still apply.
- **Other benchmarks:** first prepare their assets, registry/service and Agent
  using the links below, then set `registry_root` or `endpoint` and `agent_config`.
  The script starts registry-backed environments through the existing Runner;
  HTTP services must already be running. In particular, start RoboLab with the
  documented 8 GiB `/tmp`, not the generic Docker executor's 1 GiB default.
- **CLI runtimes:** set each cohort's `codex` to its own pinned executable, and
  `python` to its Runner/Agent interpreter when dependencies differ. `null` uses
  the existing environment resolution. One global Codex installation cannot
  satisfy all four pinned versions at once.

Paths in this YAML are relative to the YAML file, not the shell's working
directory. `bundle:NAME` selects an installed Agent wheel. Credentials remain in
the environment/auth files; do not add keys to YAML.

| Cohort | Current default coverage and preparation |
| --- | --- |
| `libero` | [Standard LIBERO](../benchmarks/libero/README.md); the configured suite, default `libero_spatial`. `ROBORSI_LIBERO_SUITE` selects another suite. |
| `libero_pro` | [PRO](../benchmarks/libero_pro/README.md); released `migration_subset`, 80 seed-0 cases. |
| `libero_long_plus_plus` | [Long++](../benchmarks/libero_lilo/README.md); released subset, 12 seed-0 cases. |
| `libero_ultra_long` | [Ultra-Long](../benchmarks/libero_lilo/README.md); released subset, 9 seed-0 cases. |
| `robocasa365` | [Target50](model-evaluation.md); default 340 cells, including tasks without released memory. |
| `robolab` | [Standard RoboLab](../benchmarks/robolab/README.md#standard-http-execution); prepared 120-task seed-0 catalog and running HTTP service. No historical 120 × 10 app-server run. |
| `bench2dex` | [Frozen release](../benchmarks/bench2dex/ICL.md), [runtime setup](../agents/bench2dex_compat/README.md) and running HTTP service; the existing 26-task × 3-sample [78-case plan](../benchmarks/bench2dex/plan-78.json). Each task receives its matching Agent/ICL ZIP. |
| `dexjoco` | [Registry](../benchmarks/dexjoco/README.md) and [Agent assets](../benchmarks/dexjoco/ICL.md); 11 tasks × 5 seeds, original fresh-session 1x→10x continuation. |
| `behavior_1k` | [Licensed assets and registry](../benchmarks/behavior_1k/README.md), then [prepare the task Agent](../benchmarks/behavior_1k/README.md#prepare-export-and-evaluate-the-task-agent). Default: released `sorting_vegetables` ICL task, seed 0. Other tasks require their own learned package/Agent. |
| `lhpr_vln` | [LHPR-VLN](../benchmarks/lhpr_vln/README.md); separate prepared 114-task seed-42 registry. |
| `vlnce_r2r` | [VLN-CE](../benchmarks/vlnce/README.md); prepared R2R 100-task seed-0 registry. |
| `vlnce_rxr` | [VLN-CE](../benchmarks/vlnce/README.md); a **different** RxR 100-task seed-0 registry. |

“All benchmarks” means these released cohorts, not every upstream task or an
expanded official test set. The three navigation entries use their standard
source presets **without ICL**. For frozen-memory evaluation, install/select the
matching `bundle:vln_shared_lhpr_vln`, `bundle:vln_shared_r2r` and
`bundle:vln_shared_rxr` in `agent_config`, following the
[artifact instructions](vln_shared_evaluation.md#published-artifacts).

## Authentication and pinned CLIs

API Agents and the RoboLab/BEHAVIOR/navigation CLI Agents require an authorized
`OPENAI_API_KEY`. The API jobs use the dedicated
[`configs/models/gpt6.yaml`](../configs/models/gpt6.yaml), selecting
`gpt-6-astra` independently of a deployed model's `MODEL_NAME` and
`OPENAI_BASE_URL`. `GPT6_BASE_URL` can override the API jobs' endpoint. CLI jobs
retain the provider/authentication settings of their own presets; that variable
does not change their providers.

| Agent | Pinned Codex CLI |
| --- | --- |
| Bench2Dex, DexJoCo | 0.154.0, including their isolated policy runtimes and companion binaries |
| RoboLab | 0.160.0 |
| LHPR-VLN, VLN-CE | 0.161.0 |
| BEHAVIOR-1K | 0.162.0 |

Follow the linked guides for verified installations. Configure their individual
executable paths in the suite YAML. Bench2Dex still uses the
`openai-codex-cli-bin` package and its version/hash-checked numerical sandbox.

Bench2Dex and DexJoCo additionally need their existing local authentication and
policy environment. Set actual absolute paths, for example:

```bash
export OPENAI_API_KEY='YOUR_API_KEY'
export BENCH2DEX_CODEX_AUTH='/absolute/path/to/bench2dex/auth.json'
export ROBORSI_DEXJOCO_AUTH='/absolute/path/to/dexjoco/auth.json'
export ROBORSI_DEXJOCO_POLICY_PYTHON='/absolute/path/to/separate-policy-venv'
export ROBORSI_DEXJOCO_AGENT_ASSETS='/absolute/path/to/dexjoco-agent-assets'
```

When `codex` is null, existing `BENCH2DEX_CODEX_BIN` and
`ROBORSI_DEXJOCO_CODEX` settings still apply. Native image licenses and gated HF
downloads must already be authorized; the launcher does not obtain asset access
or generate replacement training knowledge.

## Run

Inspect the complete plan without credentials, downloads, subprocesses or files:

```bash
python scripts/evaluation/evaluate_gpt6.py --dry-run
```

After the one-time preparation and configuration, start all released cohorts:

```bash
python scripts/evaluation/evaluate_gpt6.py
```

Before any evaluation, all selected cohorts must pass local prerequisite checks
and their existing non-model preflights. Missing environments/Agents stop the
suite before evaluation starts. Actual execution can download large images and
make paid model calls. Runs are sequential, with one concurrent case per cohort.

Select fewer cohorts, for example when checking setup or running unfinished ones:

```bash
python scripts/evaluation/evaluate_gpt6.py --benchmark libero_pro --dry-run
python scripts/evaluation/evaluate_gpt6.py \
  --benchmark vlnce_r2r --benchmark vlnce_rxr --run-id gpt6-navigation-001
python scripts/evaluation/evaluate_gpt6.py \
  --config /absolute/path/to/gpt6.yaml --work-dir /data/gpt6 --run-id gpt6-all-001
```

Use a new run ID each time. The launcher never overwrites an existing invocation
or automatically retries scored failures. Task/seed filters and timeouts belong
in the corresponding suite entry; Bench2Dex uses its fixed `plan` instead.
Changing `--work-dir` moves managed caches/results, not the prepared registry,
Agent or ICL paths declared in YAML.

## Results and validation

The default overview is `work/gpt6/results/<run-id>/summary.json`, with the
resolved command plan and per-cohort preflight/evaluation logs beside it.
Detailed case results remain in `work/gpt6/native-results/`; Bench2Dex keeps its
original plan, per-task results and strict completeness report under the
invocation's `bench2dex/` directory.

Native benchmark summaries remain separate. There is no cross-benchmark average
combining incompatible metrics. A scored `task_failure` is a completed evaluation;
Agent/infrastructure errors or incomplete reports cause a nonzero suite exit.
The suite continues to the next cohort after an execution failure. Exit codes
are 0 for complete execution, 1 for failed/incomplete execution, 2 for setup or
preflight errors, and 130 for interruption. Logs redact inherited API/service
tokens. Ctrl-C stops the active child through the existing cleanup path.

Offline tests cover command construction, dry-run isolation, preflight gating,
failure reporting and result separation. They do not establish simulator
success, paid-model acceptance or reproduction of published scores.
