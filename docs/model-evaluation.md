# Evaluate a deployed model

[简体中文](model-evaluation.zh-CN.md) · [All guides](README.md)

[`scripts/evaluation/evaluate_model.py`](../scripts/evaluation/evaluate_model.py)
downloads the selected benchmark environment and available ICL, starts its
Benchmark service, evaluates your deployed model, and stops its containers on
completion or failure. Downloads are cached for reuse.

```text
Your deployment script → Running model API
    → Set base_url, api_key, model_name
    → evaluate_model.py --benchmark NAME
    → Download/build environment and ICL → Evaluate → Save results
```

Deployment commands belong in [`scripts/deploy/`](../scripts/deploy/README.md).
Evaluation and download orchestration live in `scripts/evaluation/`; native
environments and protocols remain under `benchmarks/`.

## Install and deploy

Use **Linux/WSL, Python 3.10–3.12 and a running Docker daemon**. Enhanced LIBERO
and RoboCasa365 require an NVIDIA GPU and Docker GPU support; standard LIBERO
uses CPU rendering. From this checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[runtime]'
```

Fill [`scripts/deploy/deploy_model.sh`](../scripts/deploy/deploy_model.sh) with your
verified serving command and start it in another terminal:

```bash
bash scripts/deploy/deploy_model.sh
```

This reserved file exits with an error until implemented. An already running API
can be used directly. The model must support **images and tool calls** through an
OpenAI-compatible API. Automatic presets use Chat Completions, with reasoning
and provider-specific prompt-cache extensions disabled. Pass
`--openai-format responses` if your server implements Responses.

## One-command evaluation

```bash
export OPENAI_BASE_URL='http://127.0.0.1:8000/v1'
export OPENAI_API_KEY='YOUR_API_KEY'
export MODEL_NAME='YOUR_SERVED_MODEL'

python scripts/evaluation/evaluate_model.py --benchmark libero_pro
```

**`--benchmark` is required.** No benchmark runs implicitly. Repeat it to select
several. `--base-url`, `--api-key`, and `--model-name` override these environment
variables. Prefer the environment for credentials. For an unauthenticated API
use `OPENAI_API_KEY=EMPTY`. Use the exact served model ID; the base URL normally
includes `/v1`.

| `--benchmark` | Automatic preparation and task coverage |
| --- | --- |
| `libero` | Downloads pinned assets; builds this checkout's service. One suite selected below. No released standard-LIBERO ICL package. |
| `libero_pro` | Pulls a pinned environment and verifies embedded sealed ICL. `migration_subset`: 80 seed-0 cases, 40 packages. |
| `libero_long_plus_plus` | Pulls a pinned environment and verifies embedded sealed ICL. `migration_subset`: 12 seed-0 cases/packages. |
| `libero_ultra_long` | Pulls a pinned environment and verifies embedded sealed ICL. `migration_subset`: 9 seed-0 cases/packages. |
| `robocasa365` | Builds this checkout's service from its pinned base, including kitchen assets and RLDX-1 weights; downloads pinned Target50 ICL. Default: 340 cells. |

Enhanced LIBERO uses this checkout's [combined delivery](../benchmarks/libero_three_combined/README.md).
Automatic setup overlays the current checkout's Benchmark service code on the
pinned published native image, so it does not rely on that image's older entry layout.
Its automatic catalogs expose **migration subsets, not the full official task
sets**: the sealed context maps only cover those subsets. For another/full
protocol, prepare its catalog and a compatible Agent preset using the benchmark
guides, then use the existing-source mode below. Changing the model API alone
does not reproduce published Harness scores.

RoboCasa ICL uses **only the same task's source-seed-0 experience**. The release
covers 43 of 50 tasks (305 of 340 evaluation cells). The seven tasks declared
without memory run with empty ICL. Each case records `icl-evidence.json`; global
and cross-task memory are not used. RLDX-1 is the native execution model, separate
from the evaluated model API, and starts automatically inside the environment.

The first run can download large images and build dependencies. Docker caches
images; assets, ICL and generated presets live under `work/model_eval/`.
`--work-dir /path/to/cache` changes this location. Reused assets/ICL are verified;
modified or incomplete caches fail. Choose a new work directory or move the
affected cache before retrying. LIBERO ICL is extracted from digest-pinned images
and checked against `assets.lock.json`; RoboCasa uses a fixed Hugging Face commit
and SHA-256 manifest. Enhanced LIBERO installs `tokenizers==0.22.2` into the active
virtual environment when needed.

Docker and GPU drivers are prerequisites. Image pulls use Docker credentials;
the pinned RoboCasa base currently requires GHCR access. If Docker reports
`denied`, log in with an account that can read the image. Hugging Face uses
`HF_TOKEN` or your existing Hub login when needed. Model API keys are not download
credentials. See [RoboCasa build details](../benchmarks/robocasa365/BUILD.md).

## Select catalogs with environment variables

These settings apply to automatically started Benchmark services:

| Variable | Default | Allowed values |
| --- | --- | --- |
| `ROBORSI_LIBERO_SUITE` | `libero_spatial` | `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90` |
| `ROBORSI_ROBOCASA_CELLS` | `all` | `all` (340 cells), `seed1` (50 seed-1 cells) |

```bash
ROBORSI_LIBERO_SUITE=libero_90 python scripts/evaluation/evaluate_model.py --benchmark libero
ROBORSI_ROBOCASA_CELLS=seed1 python scripts/evaluation/evaluate_model.py --benchmark robocasa365
```

They do not change external registries or already running services. There are no
`--libero-suite` or `--robocasa-cells` CLI options.

## Preparation, dry run and smoke checks

Download/build one benchmark's requirements without a model call or an episode;
model connection settings are unnecessary:

```bash
python scripts/evaluation/evaluate_model.py --benchmark robocasa365 --prepare-only
```

With no external source, `--dry-run` only prints the preparation plan: no
downloads, containers, catalog requests or model calls. It does not verify API support.

```bash
python scripts/evaluation/evaluate_model.py --benchmark libero_pro --dry-run
```

Run a single task and seed exposed by its catalog:

```bash
python scripts/evaluation/evaluate_model.py --benchmark robocasa365 \
  --task robocasa365:OpenDrawer --seed 1 --run-id smoke-001
```

Omit `--task` and `--seed` to run the selected catalog's tasks and seeds.
After the smoke check, run all 340 RoboCasa cells with a fresh run ID:

```bash
ROBORSI_ROBOCASA_CELLS=all python scripts/evaluation/evaluate_model.py \
  --benchmark robocasa365 --run-id robocasa-001
```

The inline `all` setting also overrides an earlier `ROBORSI_ROBOCASA_CELLS=seed1`.
Native Windows supports `--help` and automatic/HTTP `--dry-run` only;
preparation and evaluation require Linux/WSL.

## Existing environments and custom Agents

Pass exactly one external source to skip automatic environment/ICL preparation:

```bash
python scripts/evaluation/evaluate_model.py --benchmark libero \
  --registry-root /absolute/path/to/registry

python scripts/evaluation/evaluate_model.py --benchmark robocasa365 \
  --benchmark-url robocasa365=http://127.0.0.1:18800
```

For multiple HTTP benchmarks provide `--benchmark-url NAME=URL` for every selected
name. These are **simulator service origins**, distinct from the model's
`OPENAI_BASE_URL`. Registry mode starts its prepared services; HTTP mode connects
to services you already manage. With either external source, `--dry-run` reads
and validates catalogs/plans without episodes or model calls. Use
`--benchmark-token-env ENV_NAME` for HTTP service authentication.

`--agent-config NAME=PATH` overrides the automatic ICL preset. With an external
source the default is `configs/agents/embodied/default.yaml` (no automatic ICL).
Custom presets need their code, dependencies and context resources already
available. Their format, context and budgets are retained unless explicitly
overridden; the script always sets your model connection.

See `python scripts/evaluation/evaluate_model.py --help` for all options.
`--max-turns`, `--max-tokens`, and `--keep-turns` override Agent limits.
Enhanced LIBERO retains its preset's context and large output budget; set limits
compatible with your model. Generic LIBERO defaults to 100 turns, 2048 output
tokens and eight retained turns. RoboCasa preserves its native planner budgets:
1800 seconds for atomic tasks, 3600 for composite tasks. `--timeout-s` defaults
to 1800 seconds per transport call. Whole-case `--case-timeout-s` defaults to
43200 seconds for enhanced LIBERO and 4500 otherwise.
Automatic/HTTP services require `--max-concurrent 1`.

## Results

Results default to `<work-dir>/results/`; `--results-root` changes this location.
A run named `my-model-001` creates only selected non-empty groups, for example:

```text
work/model_eval/results/
  my-model-001-libero_pro/
  my-model-001-robocasa365-atomic/
  my-model-001-robocasa365-composite/
```

Groups contain the normal batch plan, per-case results, native artifacts and
traces. Use a fresh `--run-id` each time. Query a group with `bench status`.
Score RoboCasa by passing both result directories to
`python -m benchmarks.robocasa365.score`; append `--cells seed1` for a complete
50-cell seed-1 run. A smoke check is not a full benchmark score. See
[evaluation](evaluation.md) and [RoboCasa365](../benchmarks/robocasa365/README.md).
