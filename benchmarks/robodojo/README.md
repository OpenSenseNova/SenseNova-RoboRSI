# RoboDojo

[简体中文](README.zh-CN.md) · [Build and installation](BUILD.md)

[RoboDojo](https://robodojo-benchmark.com/) evaluates dual-arm robot manipulation
across object interaction, spatial reasoning, memory and long-horizon tasks.
This integration supports the shared model evaluation entry and retains native
tasks, action budgets and scoring.

## Setup

Use Linux x86-64, a supported NVIDIA GPU and Python 3.10.12.
Follow the [build and installation guide](BUILD.md) to prepare the environment
and task assets, then start the Benchmark service.

## Evaluation

Configure the model connection and run from the repository root:

```bash
export OPENAI_BASE_URL='https://YOUR_ENDPOINT/v1'
export OPENAI_API_KEY='YOUR_API_KEY'
export MODEL_NAME='YOUR_MODEL'
python scripts/evaluation/evaluate_model.py --benchmark robodojo \
  --benchmark-url robodojo=http://127.0.0.1:18800 \
  --run-id robodojo-001
```

`--benchmark-url` selects the Benchmark service; `OPENAI_BASE_URL` selects the
model endpoint. Add `--task robodojo:cover_blocks --seed 0` for one task and
layout, or `--dry-run` to inspect the evaluation plan.

Model and evaluation settings are in
[`configs/models/robodojo.yaml`](../../configs/models/robodojo.yaml) and
[`configs/benchmarks/robodojo.yaml`](../../configs/benchmarks/robodojo.yaml).
Results are saved under `work/model_eval/results/<run-id>/`.
See the [model evaluation guide](../../docs/model-evaluation.md) for more options.
