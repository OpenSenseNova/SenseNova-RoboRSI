# Evaluate a deployed model

[简体中文](model-evaluation.zh-CN.md) · [All guides](README.md)

[`scripts/evaluation/evaluate_model.py`](../scripts/evaluation/evaluate_model.py)
connects a deployed model API to the existing benchmark evaluator. The workflow is:

```text
Prepare the framework and benchmarks once
    → Start the model API with your deployment script
    → Configure base_url, api_key and model_name
    → Run evaluate_model.py and inspect the saved results
```

The deployment entry point is reserved in
[`scripts/deploy/`](../scripts/deploy/README.md), and the evaluation entry point
lives in `scripts/evaluation/`. Benchmark-specific images, assets and native
protocols remain under `benchmarks/`.

## 1. Prepare the environment and benchmarks

Use Python 3.10–3.12. Actual evaluation requires a **Linux/WSL Runner** in both
Registry and HTTP modes. Native Windows Python can show `--help` and perform
HTTP `--dry-run` checks, but cannot execute evaluation cases. From the repository
root in Linux/WSL:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[runtime,test]'
```

Then follow the relevant benchmark guides to prepare images, assets and a
registry, or start existing Benchmark HTTP services. The script supports all
five entries below and selects all five when `--benchmark` is omitted.

| Benchmark | `--benchmark` value | Preparation guide |
| --- | --- | --- |
| LIBERO | `libero` | [LIBERO](../benchmarks/libero/README.md) |
| LIBERO-PRO | `libero_pro` | [LIBERO-PRO](../benchmarks/libero_pro/README.md) |
| LIBERO-Long++ | `libero_long_plus_plus` | [Long++ / Ultra-Long](../benchmarks/libero_lilo/README.md) |
| LIBERO-Ultra-Long | `libero_ultra_long` | [Long++ / Ultra-Long](../benchmarks/libero_lilo/README.md) |
| RoboCasa365 Target50 | `robocasa365` | [RoboCasa365](../benchmarks/robocasa365/README.md) |

For branch-specific setup, use the current local `LIBERO` branch for LIBERO
integration and the [remote `main` RoboCasa365 guide](https://github.com/OpenSenseNova/SenseNova-RoboRSI/blob/main/benchmarks/robocasa365/README.md)
for RoboCasa365. The local LIBERO entry points may evolve with ongoing changes.

Use exactly one execution mode:

- **Registry:** `--registry-root /absolute/path/to/registry`. The prepared registry
  must contain every selected benchmark. The evaluator starts its Docker services.
  Use Linux/WSL for this mode, including `--dry-run`. Native benchmark images target
  Linux; follow each guide's hardware and Docker requirements.
- **Existing HTTP services:** repeat `--benchmark-url NAME=URL` for every selected
  benchmark. The evaluator connects to those services without installing a local
  simulator. Actual execution still requires Linux/WSL. On native Windows,
  activate with `.venv\Scripts\Activate.ps1` and use this mode only for `--dry-run`.

The available tasks, seeds, profiles and physical action budgets come from the
manifest or HTTP catalog. A catalog exposing a subset only evaluates that subset.
For a full leaderboard protocol, prepare the corresponding full catalog and
official profile according to its benchmark guide. This general API workflow
does not by itself reproduce the published model or Harness scores.

RoboCasa365 also requires its native assets and a separate **RLDX-1 execution
service**. Its image can start that process automatically, or connect to an
existing one. Your evaluated model API does not replace RLDX-1; see the
[RoboCasa365 guide](../benchmarks/robocasa365/README.md).

## 2. Start the model API

Skip this step if your model service is already running. Otherwise, put your
verified serving command in
[`scripts/deploy/deploy_model.sh`](../scripts/deploy/deploy_model.sh), then start it:

```bash
bash scripts/deploy/deploy_model.sh
```

This file is an explicit placeholder and exits with an error until you implement
it. The repository does not assume your model weights, serving framework or GPU
layout. See the [deployment notes](../scripts/deploy/README.md) for the interface
your command needs to expose.

The model must support **image inputs and tool calling** through an
OpenAI-compatible API. The generic preset uses Chat Completions; pass
`--openai-format responses` for a service implementing the Responses API.
Use the exact model ID advertised by your serving process.

## 3. Configure the model and run

Set the three connection values in your evaluation shell. Prefer the environment
variable for the API key so it does not appear in command-line arguments:

```bash
export OPENAI_BASE_URL='http://127.0.0.1:8000/v1'
export OPENAI_API_KEY='YOUR_API_KEY'
export MODEL_NAME='YOUR_SERVED_MODEL'
```

For a service with authentication disabled, use a non-empty placeholder such as
`OPENAI_API_KEY=EMPTY`. The API base URL is usually the serving root including
`/v1`, not the `/chat/completions` endpoint. Confirm the path with your deployment.

| Address | Purpose | Example |
| --- | --- | --- |
| `--base-url` / `OPENAI_BASE_URL` | The model being evaluated | `http://127.0.0.1:8000/v1` |
| `--benchmark-url NAME=URL` | The simulator's Benchmark service | `robocasa365=http://127.0.0.1:18800` |

First, check one LIBERO task from a prepared catalog:

```bash
python scripts/evaluation/evaluate_model.py \
  --registry-root /absolute/path/to/registry \
  --benchmark libero --task libero:0 --seed 0 --dry-run
```

`--dry-run` validates and prints the evaluation plans without model requests,
Docker startup or case results. HTTP mode still reads the remote catalog.
It does not establish that the model supports the requested API, images or tools.
Remove `--dry-run` to execute the single-task smoke check.

To run every catalog task and seed for all five benchmarks:

```bash
python scripts/evaluation/evaluate_model.py \
  --registry-root /absolute/path/to/registry --run-id my-model-001
```

To run just LIBERO-PRO, add `--benchmark libero_pro`. Repeat `--benchmark` to
select several entries. Omit `--task` and `--seed` to use each catalog's defaults;
LIBERO and RoboCasa365 can expose different seed ranges.

For already running HTTP services, explicitly select the services you provide:

```bash
python scripts/evaluation/evaluate_model.py \
  --benchmark libero --benchmark robocasa365 \
  --benchmark-url libero=http://127.0.0.1:18000 \
  --benchmark-url robocasa365=http://127.0.0.1:18800
```

The ports above are examples; use your actual published service addresses. For a
single RoboCasa365 smoke check, select `--benchmark robocasa365` and add
`--task robocasa365:OpenDrawer --seed 1` with its registry or HTTP address.

In native Windows PowerShell, you can check an existing HTTP catalog as follows.
Run the actual evaluation from Linux/WSL:

```powershell
$env:OPENAI_BASE_URL = 'http://127.0.0.1:8000/v1'
$env:OPENAI_API_KEY = 'YOUR_API_KEY'
$env:MODEL_NAME = 'YOUR_SERVED_MODEL'
python scripts/evaluation/evaluate_model.py --benchmark robocasa365 --benchmark-url robocasa365=http://127.0.0.1:18800 --task robocasa365:OpenDrawer --seed 1 --dry-run
```

## Options

Run `python scripts/evaluation/evaluate_model.py --help` for the complete CLI.

| Option | Default / behavior |
| --- | --- |
| `--base-url`, `--api-key`, `--model-name` | Override `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `MODEL_NAME` respectively |
| `--openai-format` | Generic preset: `chat`; also accepts `responses`; an explicit value overrides a custom preset |
| `--registry-root` / `--benchmark-url NAME=URL` | Required choice of execution mode; mutually exclusive |
| `--benchmark NAME` | Repeatable; defaults to all five supported integrations |
| `--agent-config NAME=PATH` | Repeatable; use a specific Agent preset for a benchmark |
| `--task NAME:ID`, `--seed N` | Repeatable filters; omitted values use the catalog |
| `--benchmark-token-env ENV` | Environment variable holding authentication for an existing Benchmark service |
| `--max-concurrent` | `1`; HTTP evaluation requires `1` |
| `--max-turns`, `--max-tokens`, `--keep-turns` | Generic preset: `100`, `2048`, `8`; explicit values override custom preset settings |
| `--timeout-s` | `1800` seconds per transport call |
| `--case-timeout-s` | `43200` seconds for LIBERO-PRO / Long++ / Ultra-Long, `4500` for LIBERO / RoboCasa365; an explicit value overrides these defaults |
| `--results-root` | `work/model_eval/results` under the repository |
| `--run-id` | Automatically generated if omitted; use a fresh ID for each run |
| `--dry-run` | Validate and print plans without executing cases |

RoboCasa365 tasks are grouped automatically by their catalog's planner limit:
**1800 seconds for atomic tasks, 3600 seconds for composite tasks**. Transport
and complete-case timeouts are separate limits. The script preserves these
native planner budgets when running both groups.

## Use a benchmark-specific Agent preset

For a benchmark-specific Harness, repeat `--agent-config NAME=PATH` with the
appropriate preset. The script sets the model connection to your OpenAI-compatible
service while retaining the preset's API format, context and budgets, unless you
explicitly override the corresponding CLI options. RoboCasa365 planner budgets
still follow its native catalog.

The local `LIBERO` branch contains specialized presets and sealed ICL settings
for PRO, Long++ and Ultra-Long; consult `docs/libero-three-benchmarks.md` in that
branch alongside its current evaluation entry points. Those presets require a
checkout or installed Agent environment
containing their **complete runtime source, dependencies and context assets**.
Pointing this checkout at an external YAML file alone does not install those
modules. Follow that branch's setup and protocol before selecting its preset;
the generic defaults here are not its sealed ICL reproduction configuration.

## Inspect and score results

For `--run-id my-model-001`, results are saved in separate run directories:

```text
work/model_eval/results/
  my-model-001-libero/
  my-model-001-libero_pro/
  my-model-001-libero_long_plus_plus/
  my-model-001-libero_ultra_long/
  my-model-001-robocasa365-atomic/
  my-model-001-robocasa365-composite/
```

Only selected, non-empty groups are created. Each uses the normal `bench eval`
layout, including `batch_plan.json` and per-case `result.json`, native score
artifacts and traces. Query a specific group with:

```bash
bench status --results-root work/model_eval/results --run-id my-model-001-libero
```

Score RoboCasa365 by passing both existing group directories to its scorer:

```bash
python -m benchmarks.robocasa365.score \
  work/model_eval/results/my-model-001-robocasa365-atomic \
  work/model_eval/results/my-model-001-robocasa365-composite
```

The default scoring protocol expects **340 cells** and reports the mean of the
50 per-task success rates. If you prepared and ran the seed-1 protocol, append
`--cells seed1`; this still expects **all 50 seed-1 cells**. Missing or invalid
cells count as failures, so a smoke check is not a full benchmark score. See
[Evaluation](evaluation.md) for case status, traces and result comparison.
