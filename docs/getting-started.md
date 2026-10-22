# Getting started

[简体中文](getting-started.zh-CN.md) · [All guides](README.md)

Use Python 3.10–3.12 for the Agent and evaluator. The native benchmark images
use Linux and their own pinned dependencies. Running an Agent against an
existing HTTP benchmark needs no local simulator installation.

## Install

Run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[runtime,test]'
bench --help
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. The published CI and
native benchmark recipes target Linux. Unit tests use local fakes and do not
call model services or start a simulator.

```bash
pytest tests/unit_tests -q
```

## Prepare a benchmark

Choose [LIBERO](../benchmarks/libero/README.md),
[LIBERO-PRO](../benchmarks/libero_pro/README.md),
[Long++ / Ultra-Long](../benchmarks/libero_lilo/README.md),
[Bench2Dex](../benchmarks/bench2dex/README.md),
[DexJoCo](../benchmarks/dexjoco/README.md),
[BEHAVIOR-1K](../benchmarks/behavior_1k/README.md), or
[RoboCasa365](../benchmarks/robocasa365/README.md). Follow its recipe to
build an image, obtain the required assets, and write a manifest under a local
registry directory. A manifest identifies the image digest, tasks, seeds,
action schema, evaluation limits and native score rules.

LIBERO-PRO requires its evaluation asset tree at the exact digest specified in
its recipe. The repository downloader obtains the pinned public Task/Swap BDDL
and init files from the official Hugging Face dataset and verifies the complete
tree before image construction; no private cluster path or account token is
required. Follow the LIBERO-PRO or three-LIBERO single-image recipe before
running the benchmark.

## Check the evaluation path

With a LIBERO registry prepared by the recipe, use the fixed-action example:

```bash
bench eval --registry-root /absolute/path/to/registry \
  --results-root /absolute/path/to/results --run-id hold-1 \
  --benchmark libero --task libero:0 --seed 0 \
  --agent-factory examples.roborsi.hold_agent:make_agent \
  --agent-version hold-example-v1 --timeout-s 180 --case-timeout-s 900
```

The Agent reads every camera file and keeps the gripper open. A native failure
score is expected; the check establishes transport and result recording, not
task-solving ability. Use `bench status --results-root /absolute/path/to/results
--run-id hold-1` to inspect the case state.

## Run a model Agent

Set `OPENAI_API_KEY` in your environment. Set `OPENAI_BASE_URL` only when using
a compatible custom endpoint. Replace the model placeholder in
[`default.yaml`](../configs/agents/embodied/default.yaml) with your vision and
tool-calling model, or supply an override:

```bash
bench eval --registry-root /absolute/path/to/registry \
  --results-root /absolute/path/to/results --run-id model-1 \
  --benchmark libero --task libero:0 --seed 0 \
  --agent-config configs/agents/embodied/default.yaml \
  --agent-set 'runtime.llm.model="YOUR_VISION_TOOL_MODEL"' \
  --timeout-s 180 --case-timeout-s 3600
```

For an existing HTTP service, replace `--registry-root ...` with
`--benchmark-url libero=https://benchmark.example`. If the service requires a
token, set it in an environment variable and add
`--benchmark-token-env ROBORSI_BENCHMARK_TOKEN`.

Read [Agents](agents.md) for context and execution budgets, and
[Evaluation](evaluation.md) for results, concurrency and resume.
