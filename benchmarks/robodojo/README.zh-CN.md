# RoboDojo

[English](README.md) · [构建与安装](BUILD.zh-CN.md)

[RoboDojo](https://robodojo-benchmark.com/) 是面向双臂机器人操作的评测基准，
覆盖物体操作、空间推理、记忆与长程任务。本接入支持统一的模型评测入口，
保留原生任务、动作预算和评分。

## 环境准备

使用 Linux x86-64、受支持的 NVIDIA GPU 和 Python 3.10.12。
按[构建与安装指南](BUILD.zh-CN.md)准备环境、任务资源并启动 Benchmark 服务。

## 评测

在仓库根目录配置模型连接，然后运行：

```bash
export OPENAI_BASE_URL='https://YOUR_ENDPOINT/v1'
export OPENAI_API_KEY='YOUR_API_KEY'
export MODEL_NAME='YOUR_MODEL'
python scripts/evaluation/evaluate_model.py --benchmark robodojo \
  --benchmark-url robodojo=http://127.0.0.1:18800 \
  --run-id robodojo-001
```

`--benchmark-url` 指向 Benchmark 服务，模型地址由 `OPENAI_BASE_URL` 指定。
添加 `--task robodojo:cover_blocks --seed 0` 可选择单个任务与布局；
`--dry-run` 可查看评测计划。

模型与评测设置分别位于
[`configs/models/robodojo.yaml`](../../configs/models/robodojo.yaml) 和
[`configs/benchmarks/robodojo.yaml`](../../configs/benchmarks/robodojo.yaml)。
结果保存在 `work/model_eval/results/<run-id>/`，用法见[模型评测指南](../../docs/model-evaluation.zh-CN.md)。
