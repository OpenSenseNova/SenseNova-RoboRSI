# Benchmark release guides

This release contains the Agent runtime, benchmark evaluation, and adapters for
LIBERO, LIBERO-PRO, LIBERO Long++, LIBERO Ultra-Long, Bench2Dex, DexJoCo,
BEHAVIOR-1K and RoboCasa365.
Benchmark task definitions and scoring belong to each native runtime.

| Guide | English | 简体中文 |
| --- | --- | --- |
| Installation and first evaluation | [Getting started](getting-started.md) | [快速开始](getting-started.zh-CN.md) |
| One-command evaluation of a deployed model | [Model evaluation](model-evaluation.md) | [模型一键评测](model-evaluation.zh-CN.md) |
| LIBERO VLA download and deployment | [OpenPI VLA](libero-vla.md) | [VLA 自动部署](libero-vla.zh-CN.md) |
| Model, prompt, context and budgets | [Agents](agents.md) | [Agent 配置](agents.zh-CN.md) |
| Batch execution, results and comparison | [Evaluation](evaluation.md) | [评测](evaluation.zh-CN.md) |

The benchmark build recipes are in [LIBERO](../benchmarks/libero/README.md),
[LIBERO-PRO](../benchmarks/libero_pro/README.md),
[Long++ / Ultra-Long](../benchmarks/libero_lilo/README.md),
Bench2Dex ([English](../benchmarks/bench2dex/README.md) /
[简体中文](../benchmarks/bench2dex/README.zh-CN.md)),
DexJoCo ([English](../benchmarks/dexjoco/README.md) /
[简体中文](../benchmarks/dexjoco/README.zh-CN.md)),
BEHAVIOR-1K ([English](../benchmarks/behavior_1k/README.md) /
[简体中文](../benchmarks/behavior_1k/README.zh-CN.md)), and
RoboCasa365 ([English](../benchmarks/robocasa365/README.md) /
[简体中文](../benchmarks/robocasa365/README.zh-CN.md)).
See [the test guide](../tests/README.md) for offline and native checks.

Generic EmbodiedAgent presets can connect to the supported services.

DexJoCo's derived horizon code retains its
[MIT notice](../benchmarks/dexjoco/LICENSE.reference); upstream simulators and
assets retain their respective terms.
