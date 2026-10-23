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

## Configure environments, models and Agents

The entry point automatically reads `configs/benchmarks/<benchmark>.yaml`.
That file selects the environment and references a model profile and Agent
preset; no extra configuration argument is required. See
[configuration layout](../configs/README.md).

```yaml
# configs/benchmarks/libero_pro.yaml (selection excerpt)
schema_version: 1
benchmark: libero_pro
model: ../models/default.yaml
agent: ../agents/libero/pro.yaml
# environment and agent_overrides are defined in this same file
```

Configure the shared model in [`configs/models/default.yaml`](../configs/models/default.yaml):

```yaml
schema_version: 1
provider: openai
base_url: http://127.0.0.1:8000/v1
model_name: YOUR_SERVED_MODEL
api_key_env: OPENAI_API_KEY
openai_format: chat
```

Model values use **CLI > the profile's named environment variables > YAML**.
The shipped profile leaves the address/name empty for environment configuration.
`openai_format: null` preserves the Agent format. Separate benchmark YAML files
can reference different model profiles; each evaluation child receives its own
resolved key as `OPENAI_API_KEY`. No key is written to generated presets.

The same model file holds optional `vla` deployment settings; `--model-config`
selects both main and VLA models. The Agent only sets `parameters.enable_vla`.
See the [complete model example](libero-vla.md). VLA weights and tokenizer are
prepared by [`downloads.py`](../scripts/evaluation/downloads.py) before serving.

Environment images, build arguments, resource download pins and catalog defaults
are in the benchmark YAML. Agent policies are under `configs/agents/`, including
the migrated `libero/{base,v13,pro,long_plus_plus,ultra_long}.yaml` and context maps.
Benchmark `agent_overrides` applies the existing one-command API compatibility
settings over those presets. Local ICL/tokenizer paths are then written to
`<work-dir>/presets/<benchmark>.json`; edit source YAML rather than generated files.
CLI Agent options take precedence. `configs/evaluations/embodied.yaml` remains
the existing Agent comparison matrix.

You can also select each configuration by path:

```bash
python scripts/evaluation/evaluate_model.py --benchmark libero_pro \
  --benchmark-config configs/benchmarks/examples/libero_pro_smoke.yaml \
  --agent-config configs/agents/libero/pro_vla.yaml \
  --model-config configs/models/default.yaml --openai-format chat
```

One benchmark accepts bare paths; several use repeated `NAME=PATH` entries.
A minimal benchmark YAML inherits the native environment and selects tests with
`evaluation.tasks`, `evaluation.seeds` or `evaluation.selector: path.py:function`.
CLI task/seed lists override the corresponding YAML filters before the Python
selector runs. Selectors return a nonempty unique list of existing catalog cells;
different task-specific seed sets are preserved. See the
[smoke config](../configs/benchmarks/examples/libero_pro_smoke.yaml) and
[Python example](../configs/benchmarks/examples/select_first.py).

LIBERO Agents with `parameters.enable_vla: true` additionally prepare isolated
OpenPI dependencies, download pi05 weights, start and warm the VLA service.
`--prepare-only` prepares resources without GPU inference. See the
[LIBERO VLA guide](libero-vla.md) for configuration, GPU selection and existing services.

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

The RoboCasa Agent ([`configs/agents/robocasa365/icl.yaml`](../configs/agents/robocasa365/icl.yaml))
uses the Target50 harness prompt with up to 100 turns, 32768 output tokens and the
two most recent image groups in context. When the time limit or the turn budget
runs out, the Agent ends the episode and the scene at that moment is scored. An
action that does not match the action schema is returned to the model with the
reason, up to ten times per step.

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

These variables override `environment.catalog` in the benchmark YAML and apply
to automatically started Benchmark services:

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

`--dry-run` always prints an offline plan, for managed, HTTP and registry sources: no
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
Native Windows supports `--help` and offline `--dry-run` only;
preparation and evaluation require Linux/WSL.

## Existing environments and custom Agents

The benchmark YAML can select an already running image's HTTP service:

```yaml
schema_version: 1
benchmark: libero
endpoint: http://127.0.0.1:18800
evaluation:
  seeds: [0]
```

`endpoint: null` (default) prepares and starts the environment automatically.
An HTTP origin uses the existing service's catalog, reset and step interfaces.
Edit the canonical benchmark YAML to keep `--benchmark libero`, or use:

```bash
python scripts/evaluation/evaluate_model.py --benchmark libero \
  --benchmark-config configs/benchmarks/examples/libero_http.yaml
```

CLI `--benchmark-url NAME=URL` overrides the corresponding YAML endpoint;
`--registry-root` overrides all YAML endpoints. Multi-benchmark HTTP mode requires
an endpoint for every selected name. This is a simulator service origin, without
the model API's `/v1` prefix, and is not a container registry URL. A configured
endpoint selects the benchmark's referenced Agent. HTTP skips native simulator
preparation; ICL, tokenizer and optional VLA resources are prepared independently.

Pass one external source to skip local simulator preparation:

```bash
python scripts/evaluation/evaluate_model.py --benchmark libero \
  --registry-root /absolute/path/to/registry

python scripts/evaluation/evaluate_model.py --benchmark robocasa365 \
  --benchmark-url robocasa365=http://127.0.0.1:18800
```

For multiple HTTP benchmarks provide `--benchmark-url NAME=URL` for every selected
name. These are **simulator service origins**, distinct from the model's
`OPENAI_BASE_URL`. Registry mode starts its prepared services; HTTP mode connects
to services you already manage. `--dry-run` is offline and does not read catalogs.
Use benchmark `token_env: ENV_NAME`, or override it with
`--benchmark-token-env ENV_NAME`, for HTTP authentication. Tokens stay in the environment.

`--agent-config NAME=PATH` overrides the referenced Agent. Every transport uses
the benchmark YAML's selected Agent and prepares required ICL/tokenizer/VLA resources.
An explicit Agent preserves its policy and bypasses benchmark Agent overrides.
Custom factories/resources remain supported; automatically supplied resources
apply to the built-in LIBERO and RoboCasa ICL/VLA factories. Their format, context and budgets are retained unless explicitly
overridden; the script always sets your model connection.

See `python scripts/evaluation/evaluate_model.py --help` for all options.
`--max-turns`, `--max-tokens`, and `--keep-turns` override Agent limits.
Enhanced LIBERO retains its preset's context and large output budget; set limits
compatible with your model. Generic LIBERO defaults to 100 turns, 2048 output
tokens and eight retained turns. RoboCasa keeps the settings of its Agent YAML
and the native planner budgets: 1800 seconds for atomic tasks, 3600 for composite
tasks. `--timeout-s` defaults
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


## Startup checks, live logs and resume

Before downloads, normal evaluation sends one small main-model request using
the selected API format, a 1×1 image and a tool definition, capped at 16 output
tokens. Identical connections share one probe. This checks connectivity/request
acceptance, not task competence. Existing Benchmark catalogs and VLA health/
protocol are checked too. `--prepare-only` skips model requests; with existing
HTTP/registry sources it prepares Agent resources only.

`--dry-run` is always offline: no files, subprocesses, catalog or model requests,
and no model URL/key required. It prints resolved settings and their sources,
configuration paths, filters, service/resource plans and result directories.
Exact case counts are resolved from the catalog during execution.

Task, seed, completed count and elapsed time stream live, with heartbeat messages
during long cases and service warmup. Invocation artifacts sit alongside the
existing per-group directories:

```text
<results-root>/
  robocasa-001/
    resolved_config.json   # Sources, effective Agent configs/fingerprints; no keys
    summary.json           # Case success ratio, runtime errors, native group summaries
    logs/                  # Preflight/eval output and services/ container logs
  robocasa-001-robocasa365-atomic/
  robocasa-001-robocasa365-composite/
```

The overview ratio counts selected cases directly; official benchmark scoring
still uses the native summaries/scoring entry points. Inspect `complete` and
`exit_code` for interrupted/failed runs. Preparation failures produce
`failure.json`; service logs are saved before owned containers are removed.
Caches are preserved.

Resume the previous **invocation ID**, with the same inputs/results root and a new ID:

```bash
python scripts/evaluation/evaluate_model.py --benchmark robocasa365 \
  --run-id robocasa-retry --resume-from robocasa-001
```

The entry point maps all atomic/composite/seed groups, reuses terminal results,
and runs only missing or explicitly retryable cases. Normal task failures remain
valid scores. Changed selection, model/Agent identity or locked benchmark inputs
reject resume. Changing managed VLA ports does not change Agent identity.
First-time enhanced LIBERO ICL extraction may pull its pinned package-bearing
image and use Docker even with HTTP; it does not start a local simulator.
