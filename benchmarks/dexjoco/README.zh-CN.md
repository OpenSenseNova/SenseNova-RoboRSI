# DexJoCo Benchmark

[English](README.md)

DexJoCo Adapter 通过 RoboRSI 的 Benchmark HTTP 服务提供固定版本的上游仿真器。
任务发现包含 11 个 `rand_full` 任务，每个任务使用种子 `100000`–`100004`，
共 55 个任务/种子案例。仿真器版本固定为
`8d23b0fab23b17a58c4b55f3942e17013aaf8267`。

本次发布包含 benchmark、原生进程封装、动作与观测协议以及容器构建配方。
通过统一评测接口选择自己的 Agent。

## 构建与注册

按本说明准备原生运行环境与资产；安装框架不会自动提供这些依赖。
分发运行环境或资产前，请阅读[许可与归属声明](#验证与许可)。

原生运行时使用 Linux x86-64。Dockerfile 固定 Python 3.10 基础镜像，
安装带哈希锁定的依赖并检出固定版本的仿真器。渲染使用 CPU OSMesa；
benchmark 服务不需要 GPU 或模型。每个仿真器预留约 4 GiB 内存，
并为逐步保存的图像留出磁盘空间。

安装框架后，从仓库根目录执行：

```bash
python -m benchmarks.package --benchmark dexjoco --output ../dexjoco-context
docker build -t dexjoco-benchmark:local ../dexjoco-context

git clone https://github.com/brave-eai/dexjoco.git ../dexjoco-upstream
git -C ../dexjoco-upstream checkout 8d23b0fab23b17a58c4b55f3942e17013aaf8267
```

仅使用导出的构建上下文。其 `source.json` 和 `build.json` 记录运行时与
构建输入的哈希。用于发现任务的上游 checkout 必须处于固定版本且没有修改。

通过自己的镜像仓库发布镜像，再将 `DEXJOCO_IMAGE_WITH_DIGEST` 设置为实际的
`repository@sha256:digest`。仅有本地 tag 无法生成 registry manifest；
本仓库不提供已发布镜像的 digest。

```bash
python -m benchmarks.dexjoco.manifest --upstream ../dexjoco-upstream \
  --image "$DEXJOCO_IMAGE_WITH_DIGEST" \
  --output ../benchmark-registry/dexjoco/manifest.json
```

命令会自动创建 `--output` 缺少的父目录。
生成的 manifest 描述默认 10x 服务、全部 55 个案例、原生评分和镜像内资产，
不需要额外挂载宿主机资产。

## 评测 Agent

在环境变量中设置模型服务凭据，选择支持视觉和工具调用的模型。
使用通用 Agent 配置，并按下方命令指定 DexJoCo benchmark 名称与任务 ID。
服务提供原生动作协议和评分规则。[快速开始](../../docs/getting-started.zh-CN.md)
中的固定动作示例仅适用 LIBERO。原始 ICL 构建源码、冻结包准备和双会话任务 Agent
见 [ICL 指南](ICL.zh-CN.md)。历史 ICL 包本身需要单独下载，
通用配置不代表能够复现历史成绩或已经成功完成任务。

```bash
bench eval --registry-root ../benchmark-registry \
  --benchmark dexjoco --task dexjoco:water_plant --seed 100000 \
  --agent-config configs/agents/embodied/default.yaml \
  --agent-set 'runtime.llm.model="YOUR_VISION_TOOL_MODEL"' \
  --results-root ../dexjoco-results --run-id water-pilot \
  --timeout-s 3600 --case-timeout-s 21600 --dry-run
```

移除 `--dry-run` 后才会执行案例并调用模型。省略 `--task` 和 `--seed` 可运行
全部 55 个案例；使用 `--max-concurrent 1` 顺序执行。Runner 的超时限制约束
实际运行时间，不替代物理动作预算。自定义 Agent factory、上下文、工具预算、
结果和续跑请见 [Agent](../../docs/agents.zh-CN.md) 与
[评测](../../docs/evaluation.zh-CN.md)。

如需直接启动本地服务，使用独立的 Python 3.10 环境，先执行
`pip install --require-hashes -r benchmarks/dexjoco/requirements.txt`，
再从仓库根目录运行：

```bash
export ROBORSI_DEXJOCO_UPSTREAM="$(realpath ../dexjoco-upstream)"
export ROBORSI_DEXJOCO_MULTIPLIER=10
export MUJOCO_GL=osmesa
python -m roborsi.evaluation.service_host --kind benchmark \
  --factory robots.dexjoco.benchmark_service:make_dexjoco_benchmark \
  --host 127.0.0.1 --port 8000
```

宿主环境还需安装 `git`、`libosmesa6`、`libgl1` 和 `libglib2.0-0`。
将评测命令中的 registry 选项替换为
`--benchmark-url dexjoco=http://127.0.0.1:8000`。同一任务/种子的独立运行应启动
新的服务进程，因为服务会保留 episode ID。若使用原始时长，在启动服务前将
multiplier 改为 `1`；仅支持 `1` 和 `10`。

## 动作、停止和评分

每个动作是 `{"vector": [...]}`，单臂为 22 个有限数值，双臂为 44 个：
依次为绝对 xyz（米）、旋转向量（弧度）和 16 个手部目标；双臂右手在前。
Adapter 将旋转转换为标量在前的四元数，并保持原生手部顺序。
每个动作对应原生 30 Hz 控制的一步，不裁剪或插值。Click Mouse 的前 30 步
必须严格使用 [`contracts.py`](../../robots/dexjoco/contracts.py) 中的
`ALIGNMENT` 向量。

公开观测包含任务指令、机器人本体状态和配置指定的 RGB 相机。
原生奖励、物体状态、评分和私有 checkpoint 保留为评测产物。

原始任务上限为 1,000–1,500 步。10x 模式仅在内存中延长原生终止步数比较，
不修改上游文件或其他成功、失败判据。尚未终止的 episode 经过 1x 边界时
继续使用同一个仿真器，不重置场景。Adapter 保存私有 checkpoint，并报告
1x 和累计 10x 结果。到达该边界前，`remaining_steps` 仅报告原始预算。
是否在 1x 时切换会话或重新规划由 Agent 决定。

成功与否由原生终止结果决定。策略显式停止计为 `policy_failure`，不会获得
额外步数；基础设施错误仍报告为错误。主评分是原生成功率，以任务加权平均，
并以预期案例数为分母。超时、重试和模型/工具预算不能用于突破原生物理步数限制。

## 验证与许可

```bash
pytest tests/unit_tests/robots/dexjoco -q
```

这些离线测试使用 fake 仿真器，检查任务发现、动作、信息边界、停止、评分产物
和资源清理。它们验证接口行为，不代表已运行原生仿真器或取得模型任务成功。

框架许可证不替代上游代码、机器人、网格及其他资产的许可证。分发
[上游仿真器](https://github.com/brave-eai/dexjoco) checkout 或容器资产时，
应保留相应 notices。时长扩展代码的 MIT 声明保留在
[LICENSE.reference](LICENSE.reference) 中。
