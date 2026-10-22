# Automatic LIBERO VLA preparation and deployment

[中文](libero-vla.zh-CN.md) · [Model evaluation](model-evaluation.md)

Deploy your main vision/tool model API first. When a LIBERO Agent enables VLA,
the evaluation entry point additionally **builds OpenPI, downloads weights,
starts and warms the VLA service, evaluates, and removes its owned containers**.
The main model plans; pi05_libero predicts actions from current observations.

## Usage

Install the framework using the [evaluation guide](model-evaluation.md), fill
the main model connection in [models/default.yaml](../configs/models/default.yaml)
or its environment variables, and set `OPENAI_API_KEY`. Then run:

```bash
python scripts/evaluation/evaluate_model.py --benchmark libero_pro \
  --agent-config configs/agents/libero/pro_vla.yaml --openai-format chat
```

Use Linux/WSL, Docker, NVIDIA Container Toolkit and an NVIDIA GPU for VLA.
Standard LIBERO's simulator can still render on CPU. Released presets remain
VLA-disabled; they do not build or download OpenPI. The optional
[PRO VLA preset](../configs/agents/libero/pro_vla.yaml) simply enables
`parameters.enable_vla: true`. The standard, PRO, Long++ and Ultra-Long presets
support the same flag and declare `resources.vla_profile` pointing to
[models/pi05_libero.yaml](../configs/models/pi05_libero.yaml).

That profile controls the OpenPI commit, policy config, checkpoint, GPUs,
startup/request timeouts and `action_steps`. OpenPI source is pinned and uses its
`uv.lock`. The official GCS checkpoint URI is not an immutable weights revision;
upstream caching and recorded source do not constitute a weights SHA-256 lock.
Use your own controlled local checkpoint when that guarantee is required.

```bash
# Build/download without GPU inference or episodes.
python scripts/evaluation/evaluate_model.py --benchmark libero_pro \
  --agent-config configs/agents/libero/pro_vla.yaml --prepare-only

# Plan without downloads, files or containers.
python scripts/evaluation/evaluate_model.py --benchmark libero_pro \
  --agent-config configs/agents/libero/pro_vla.yaml --dry-run \
  --base-url http://127.0.0.1:8000/v1 --model-name YOUR_MODEL

# Choose a separate GPU for VLA.
ROBORSI_VLA_GPUS=device=1 python scripts/evaluation/evaluate_model.py \
  --benchmark libero_pro --agent-config configs/agents/libero/pro_vla.yaml --openai-format chat
```

`ROBORSI_LIBERO_VLA_CHECKPOINT` overrides the profile with a GCS URI or absolute
local checkpoint directory; local checkpoints are mounted read-only. Downloads
use `<work-dir>/vla/openpi-cache/` (`work/model_eval/` by default). JAX, Torch and
OpenPI stay in the isolated container. Main model API keys are not passed to it.

## Independent configuration paths and Python task selection

```bash
python scripts/evaluation/evaluate_model.py --benchmark libero_pro \
  --benchmark-config configs/benchmarks/examples/libero_pro_smoke.yaml \
  --agent-config configs/agents/libero/pro_vla.yaml \
  --model-config configs/models/default.yaml --openai-format chat
```

For one benchmark, each option accepts a bare path. With several benchmarks,
repeat options as `NAME=PATH`. CLI paths resolve from the current directory;
YAML references resolve from their declaring file. An explicit Agent preserves
its policy, budgets and format, so select `chat` for a Chat Completions service.

A minimal benchmark config inherits the canonical native environment and
changes only test selection:

```yaml
schema_version: 1
benchmark: libero_pro
evaluation:
  seeds: [0]
  selector: select_first.py:select_cases
  parameters:
    task_count: 1
```

`evaluation.tasks` can name task IDs directly. Or define a Python selector:

```python
def select_cases(cases, *, benchmark, parameters):
    tasks = list(dict.fromkeys(c['task_id'] for c in cases))[:parameters['task_count']]
    return [dict(c) for c in cases if c['task_id'] in tasks]
```

The function runs as user-supplied Python in the Runner and returns a nonempty,
unique list of existing `{task_id, seed}` cells. It can choose different seeds
per task without re-expanding a cross product. CLI `--task`/`--seed` override the
corresponding YAML filters; the selector then runs on the remaining cells.
New native tasks still need a compatible benchmark-owned catalog.

## Execution and existing services

The planner gains a `roborsi_action` variant with `primitive: vla` and a short
skill instruction. The Agent reads the current case's verified images and
eight-dimensional state, requests a prediction and submits native actions.
Enhanced LIBERO executes at most `action_steps` predicted actions per call;
standard LIBERO retains its one-action contract. Native stepping, stopping and
scoring remain owned by the Benchmark. Each case records `vla.jsonl` evidence.

Identical profiles share one VLA server within an invocation. Normal exit,
errors and interruptions remove only owned containers and preserve caches.
Set `ROBORSI_LIBERO_VLA_URL=http://host:port` to attach an existing compatible
HTTP server instead of downloading/deploying. This is this repository's
`/v1/health` and `/v1/infer` protocol, not upstream's WebSocket endpoint.
An existing `ROBORSI_VLA_MCP_URL` (or the variable named by `vla_url_env`)
retains the original MCP mode and skips automatic OpenPI setup. Both addresses
are separate from the main model's `OPENAI_BASE_URL`.

Upstream references: [OpenPI](https://github.com/Physical-Intelligence/openpi),
[LIBERO observation and execution example](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/main.py).
