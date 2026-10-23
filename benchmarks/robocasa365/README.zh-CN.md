# RoboCasa365 Target50 benchmark

[English](README.md)

本集成通过 RoboRSI 通用 Benchmark HTTP 接口提供 RoboCasa365 Target50。
每个动作执行一个原生 RoboCasa 工具，包括导航、查看相机和 RLDX-1 操作。
最终原生结果由模拟器的 `state.success` 判定。RLDX-1 属于已发布工具契约的
执行依赖，运行在独立进程中。

| 项目 | 内容 |
| --- | --- |
| Benchmark ID | `robocasa365` |
| 模拟器 | `RLinf/robocasa@2692d8fc5fd86708a1b2028dcbce892ec418a9e7`、`RLinf/robosuite@97cfbde4b68d8ec43dad20cf4747297866a6ca2e`、MuJoCo 3.3.1 |
| RLDX 运行时 | `RLinf/RLDX-1@ebcfd13df5177e4b3e574bdf5a3b427c7c4a1e8a` |
| RLDX 权重 | `RLWRLD/RLDX-1-FT-RC365@587e9ecdcc5e7184fcc17f58713908edff5af041` |
| 评测单元 | 共 340 个：18 个原子任务 × seeds 1–10，16 个已见组合任务和 16 个未见组合任务 × seeds 1–5 |
| 评分 | 50 个任务各自成功率的均值；缺失或无效单元计为失败 |

任务清单、各 split 预算和依赖版本见
[`target50.json`](../../robots/robocasa/eval/target50.json)。每个 episode 提供当前
任务语言、公开状态和相机图像。`finish` 和显式 episode stop 对当前原生状态评分。
工具参数错误会返回工具反馈；动作仍须选择受支持的 primitive。

## 一键评测

在仓库根目录设置模型 API 后运行：

```bash
export OPENAI_BASE_URL='http://127.0.0.1:8000/v1'
export OPENAI_API_KEY='YOUR_API_KEY'   # 服务不需要密钥时填 EMPTY
export MODEL_NAME='YOUR_SERVED_MODEL'

# 先跑一个 cell 检查环境
python scripts/evaluation/evaluate_model.py --benchmark robocasa365 \
  --task robocasa365:OpenDrawer --seed 1 --run-id robocasa-smoke

# seed 1 的 50 个 cell
ROBORSI_ROBOCASA_CELLS=seed1 python scripts/evaluation/evaluate_model.py \
  --benchmark robocasa365 --run-id robocasa-seed1

# 全部 340 个 cell
ROBORSI_ROBOCASA_CELLS=all python scripts/evaluation/evaluate_model.py \
  --benchmark robocasa365 --run-id robocasa-full
```

脚本会按当前 checkout 构建服务镜像，下载 Target50 ICL 包，并使用 Target50 Agent
（[`configs/agents/robocasa365/icl.yaml`](../../configs/agents/robocasa365/icl.yaml)）。
Responses API 模型加 `--openai-format responses`。使用已经启动的 Benchmark 服务时，加
`--benchmark-url robocasa365=http://127.0.0.1:18800`。

结果写入 `work/model_eval/results/`，每个 split 组一个目录。seed 1 的评分命令：

```bash
python -m benchmarks.robocasa365.score \
  work/model_eval/results/robocasa-seed1-robocasa365-* --cells seed1
```

评全部 340 个 cell 时去掉 `--cells seed1`。完整参数见
[模型一键评测指南](../../docs/model-evaluation.zh-CN.md)。

## 构建与启动

按 [BUILD.zh-CN.md](BUILD.zh-CN.md) 导出上下文并构建镜像。原始固定 digest 的 GHCR
基础镜像需要拉取权限，本仓库暂未提供从公开输入完整重建该基础镜像的配方。
其原生环境已包含厨房资产和 RLDX-1 权重；在宿主机安装 RoboRSI 不会安装这些内容。
模拟器、资产和权重分别适用其上游条款。

原生运行需要 Linux x86-64、Docker、NVIDIA Container Toolkit 和支持 CUDA 12.6
的驱动。继承环境使用 Python 3.10.21 和 Torch 2.7.0+cu126。RLDX-1 需要 GPU
（源交付预算约 16 GB 显存），厨房使用 CPU OSMesa 渲染。基础镜像压缩体积约
25.94 GB，还需为解压、构建层和结果预留空间。

将 `ROBOCASA365_IMAGE` 设为本次构建的镜像。默认服务在第一个 episode 开始时
根据 `RLDX_CHECKPOINT` 启动自己的 RLDX-1 进程：

```bash
docker run --rm --gpus all -p 127.0.0.1:18800:8000 \
  -e ROBORSI_ROBOCASA_CELLS=seed1 "$ROBOCASA365_IMAGE"
```

`seed1` 提供 50 个单元；省略该变量或设为 `all` 则提供 340 个单元。
每个 episode 拥有自己的环境子进程，结束时关闭对应 RLDX session。
服务固定使用 40 个 RLDX chunk、999 的 settle patience 和每个 chunk 八步动作。
服务退出时，其拥有的 RLDX 进程也会退出。

## 评测与评分

在 Runner 环境安装框架，按 [Agent 指南](../../docs/agents.zh-CN.md) 配置模型。
从仓库根目录运行：

```bash
bench eval --benchmark-url robocasa365=http://127.0.0.1:18800 \
  --benchmark robocasa365 --task robocasa365:OpenDrawer --seed 1 \
  --agent-config configs/agents/embodied/default.yaml \
  --agent-set 'runtime.llm.model="YOUR_VISION_TOOL_MODEL"' \
  --agent-set 'agent.max_turns=100' \
  --agent-set 'agent.planner_timeout_s=1800' \
  --results-root work/robocasa365/results --run-id open-drawer-1 \
  --max-concurrent 1 --timeout-s 1800 --case-timeout-s 4500
```

Target50 对原子任务设置 1800 秒、
组合任务设置 3600 秒预算；每次运行需选择匹配的 Agent planner 预算。
Runner 超时是独立的传输与 case 限制。服务最多接受 400 个原生工具动作，
该上限不能替代 Target50 的 planner 限制。

对保存的运行结果评分：

```bash
python -m benchmarks.robocasa365.score \
  work/robocasa365/results/open-drawer-1 --cells seed1
```

`--cells seed1` 以全部 50 个 seed-1 单元为分母，包括示例未执行的单元。
省略该参数则使用完整 340 单元协议。评分器列出缺失或无效单元，后续有效尝试
可替换先前的无效尝试。Planner 超时视为有效的任务失败。

## Registry 或共享 RLDX 服务

使用 Registry 管理容器时，先将本次构建的镜像推送至自己的 registry，
将 `ROBOCASA365_IMAGE_DIGEST` 设为其实际 `repository@sha256:digest`，
然后在同一镜像内部生成 manifest：

```bash
mkdir -p work/robocasa365/registry
docker run --rm \
  --mount type=bind,src="$PWD/work/robocasa365/registry",dst=/registry \
  "$ROBOCASA365_IMAGE_DIGEST" \
  python -m benchmarks.robocasa365.manifest --image "$ROBOCASA365_IMAGE_DIGEST" \
  --output /registry/robocasa365/manifest.json --cells seed1
```

Manifest 请求一个 GPU、八个 CPU、32 GB 内存和 900 秒启动时限。
资产和权重位于镜像中，因此资产列表为空。评测时将 `--benchmark-url` 替换为
`--registry-root work/robocasa365/registry`。
在交付镜像内部生成 manifest 可使 Adapter 身份与运行服务一致。
不要将当前 checkout 生成的 manifest 与源码 digest 不同的旧发布镜像搭配使用。

如需让多个 Benchmark 服务共享已有 RLDX 进程，镜像保留原生入口：

```bash
docker run --rm --gpus all --network host "$ROBOCASA365_IMAGE" \
  python robots/robocasa/vla_server.py --model-path /opt/rldx/RLDX-1-FT-RC365 \
  --transport http --host 127.0.0.1 --port 18710
```

各 Benchmark 服务使用 `--network host` 与
`-e ROBORSI_ROBOCASA_VLA_ENDPOINT=http://127.0.0.1:18710` 启动，并分别选择不同
HTTP 端口。此入口替代服务自行启动的 RLDX 进程，任务动作和评分保持一致。

离线测试覆盖 catalog 与 manifest 一致性、工具调用、原生结果发布、清理和源码导出。
这些测试不证明本次迁移已成功构建原生镜像、执行模拟器 episode 或完成原生任务。
