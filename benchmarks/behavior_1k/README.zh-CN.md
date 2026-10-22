# BEHAVIOR-1K

[English](README.md) · [全部指南](../../docs/README.md)

本接入通过通用 Benchmark 服务与 Runner 提供 BEHAVIOR-1K 评测，包含原生任务检查、
独立模拟器进程、官方 21 维动作接口、RGB-D 观测与原生评分。

## 任务、动作与评分

原生源码固定为 BEHAVIOR-1K v3.9.2，commit
`b1979916ec1549b10a4e65e630bc6504a9af1b00`。
[任务列表](official_tasks.json)包含 100 个任务。生成 catalog 时逐个原生 reset，
读取实际任务指令与步数上限。服务提供该 catalog 中的任务，使用 `public_test`
instance 301（public-test index 0）、seed 0。

每次动作格式为 `{"action": [...]}`，必须包含恰好 21 个有限数值，依次为：

| 维数 | 控制内容 |
| --- | --- |
| 3 | 底盘速度 |
| 4 | 躯干关节目标 |
| 6 | 左末端在底盘坐标系下的位置与旋转向量 |
| 1 | 左夹爪 |
| 6 | 右末端在底盘坐标系下的位置与旋转向量 |
| 1 | 右夹爪 |

动作必须符合原生边界；适配器拒绝越界值，不裁剪或归一化。观测包含全分辨率头部、
双腕 RGB/depth PNG、公开的本体状态和动作边界。Depth 图像使用 0.01–10 m 范围、
3.5 m 偏移的反向对数强度编码，像素值为显示强度，不是米制距离。

成功由原生 `task.success` 决定，分数来自官方聚合的 `q_score.final`。
显式停止在当前状态评分，不推进物理。原生 step 或 metric 失败报告为基础设施错误。
每个 episode 独占模拟器进程，并有清理时限。

## 环境与镜像

使用 Linux x86_64、Python 3.11、Docker、支持的 NVIDIA 驱动和 NVIDIA Container
Toolkit。仿真需要 RTX 级 GPU，建议 24 GB 显存。准备 32 GB 内存，并为镜像、
授权资产及其快照预留磁盘空间；从源码重建时另需至少 80 GB 空闲构建空间。

在仓库根目录安装 Agent 与评测器：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[runtime]'
```

将 `BEHAVIOR_DELIVERY` 设为源码目录外的绝对路径，路径中不包含符号链接。
直接拉取已发布的 Benchmark 镜像，并读取其 digest。若 GHCR Package 需要认证，
先用有拉取权限的账号执行 `docker login ghcr.io -u YOUR_GITHUB_USERNAME`。

```bash
mkdir -p "$BEHAVIOR_DELIVERY"
BEHAVIOR_IMAGE_TAG=ghcr.io/opensensenova/sensenova-roborsi-behavior-1k:v1.0.0-cu128
docker pull "$BEHAVIOR_IMAGE_TAG"
BEHAVIOR_IMAGE=$(docker image inspect "$BEHAVIOR_IMAGE_TAG" --format '{{index .RepoDigests 0}}')
```

后续命令使用 `BEHAVIOR_IMAGE` 中解析出的 `repository@sha256:digest`，
并将其保存在部署记录中。使用已发布镜像评测无需重新构建，也不需要镜像仓库写权限。
如需从源码重建，可参考 [Dockerfile](Dockerfile)、[构建锁](build-lock.json)
和[构建脚本](build_image.py)。

## 资产、catalog 与 registry

### 下载数据集并获取 key

评测需要模拟器资产、任务实例以及 `omnigibson.key`。
官方下载仓库是
[behavior-1k/zipped-datasets](https://huggingface.co/datasets/behavior-1k/zipped-datasets/tree/main)。
上述固定版本的原生源码使用以下压缩包：

| 资产目录 | 官方仓库中的压缩包 |
| --- | --- |
| `behavior-1k-assets` | `behavior-1k-assets-3.9.0.zip` |
| `omnigibson-robot-assets` | `omnigibson-robot-assets-3.8.2.zip` |
| `2026-challenge-task-instances` | `2026-challenge-task-instances.zip` |

**1. 使用已提供的镜像。** 按上文拉取
`ghcr.io/opensensenova/sensenova-roborsi-behavior-1k:v1.0.0-cu128`
并设置 `BEHAVIOR_IMAGE`。镜像已包含 OmniGibson，本机无需安装或激活原生环境。

**2. 设置数据目录并下载。** 选择源码目录外、可写的绝对路径，为压缩包、解压数据及
快照副本预留空间。以下命令将本机目录挂载到容器，下载结果保存在本机：

```bash
export BEHAVIOR_DATA=/absolute/path/outside/checkout/behavior-data
mkdir -p "$BEHAVIOR_DATA"
docker run --rm -it --gpus all \
  --mount type=bind,src="$BEHAVIOR_DATA",dst=/roborsi/assets/behavior_1k \
  -e OMNIGIBSON_DATA_PATH=/roborsi/assets/behavior_1k \
  "${BEHAVIOR_IMAGE:?Pull the provided image and resolve its digest first}" \
  python -m omnigibson.utils.asset_utils \
  --download_behavior_1k_assets \
  --download_omnigibson_robot_assets \
  --download_2026_challenge_task_instances
```

**3. 获取 key。** 首次运行时，阅读许可并在同意后输入 `y`。命令会自动将 key
保存到 `$BEHAVIOR_DATA/omnigibson.key`，并继续下载数据。

下载完成后的本机目录为：

```text
$BEHAVIOR_DATA/
  behavior-1k-assets/
  omnigibson-robot-assets/
  2026-challenge-task-instances/
  omnigibson.key
```

**4. 检查下载结果。** 命令完成后检查必要路径，不输出 key 内容：

```bash
test -s "$BEHAVIOR_DATA/omnigibson.key" &&
test -s "$BEHAVIOR_DATA/behavior-1k-assets/VERSION" &&
test -s "$BEHAVIOR_DATA/omnigibson-robot-assets/VERSION" &&
test -d "$BEHAVIOR_DATA/2026-challenge-task-instances/metadata" &&
printf 'Key and required asset paths are present.\n'
```

以上只检查文件／目录是否存在，不代表完整性校验通过。上游下载器会跳过任何已存在的
数据集目录，包括此前解压中断留下的不完整目录；遇到这种情况，先将对应目录移到
其他位置，或选择新的数据目录，再重新下载。下载器未固定 HF 仓库 revision，
复现时应保留下文生成的完整资产快照及 SHA-256。

### 生成资产快照并注册任务

`prepare assets` 只复制并计算已下载文件的哈希，不会下载数据集或获取 key。
Key 保存在私有资产挂载中，并需为快照副本预留空间。在仓库根目录执行：

```bash
python -m benchmarks.behavior_1k.prepare assets \
  --source "$BEHAVIOR_DATA" \
  --output-dir "$BEHAVIOR_DELIVERY/registry/behavior_1k/assets"
BEHAVIOR_SNAPSHOT=$(python -c \
  'import json, sys; print(json.load(open(sys.argv[1]))["path"])' \
  "$BEHAVIOR_DELIVERY/registry/behavior_1k/assets/asset.json")
```

在仓库根目录使用 Bash 执行以下命令，从 `official_tasks.json` 读取全部 100 个任务，
在已拉取的镜像中逐个执行原生 reset 并生成完整 catalog。每个任务使用 seed 0、
public_test instance 301。Catalog 和 work 目录必须尚不存在；此步骤会加载 100 次
原生任务，耗时明显长于单任务检查。

```bash
python - "$BEHAVIOR_DELIVERY/catalog-task-args.txt" <<'PY'
import json
import sys
from pathlib import Path

tasks = json.loads(Path("benchmarks/behavior_1k/official_tasks.json").read_text())["tasks"]
assert len(tasks) == len(set(tasks)) == 100, "Expected 100 unique official tasks"
Path(sys.argv[1]).write_text("".join(f"--task\n{task}\n" for task in tasks))
PY
mapfile -t BEHAVIOR_TASK_ARGS < "$BEHAVIOR_DELIVERY/catalog-task-args.txt"

docker run --rm --gpus all \
  --mount type=bind,src="$BEHAVIOR_SNAPSHOT",dst=/roborsi/assets/behavior_1k,readonly \
  --mount type=bind,src="$BEHAVIOR_DELIVERY",dst=/delivery \
  "$BEHAVIOR_IMAGE" python -m benchmarks.behavior_1k.prepare catalog \
  "${BEHAVIOR_TASK_ARGS[@]}" \
  --output-dir /delivery/registry/behavior_1k/catalog \
  --work-dir /delivery/catalog-work
```

catalog 生成后，确认它恰好包含全部 100 个任务，再继续注册。若生成或检查失败，
先解决失败原因，不要将部分任务登记为完整任务集：

```bash
python - "$BEHAVIOR_DELIVERY/registry/behavior_1k/catalog/catalog.json" <<'PY'
import json
import sys
from pathlib import Path

expected = json.loads(Path("benchmarks/behavior_1k/official_tasks.json").read_text())["tasks"]
rows = json.loads(Path(sys.argv[1]).read_text())["tasks"]
assert len(rows) == len(expected) == 100
assert {row["task_id"] for row in rows} == set(expected), "Catalog task mismatch"
assert all(row["seed"] == 0 and row["native"]["instance_id"] == 301 for row in rows)
print("Verified catalog: 100 tasks, seed 0, public_test instance 301")
PY
```

从同一镜像读取运行时标识，再生成 manifest：

```bash
docker run --rm \
  --mount type=bind,src="$BEHAVIOR_DELIVERY",dst=/delivery \
  "$BEHAVIOR_IMAGE" python -m benchmarks.behavior_1k.prepare runtime-info \
  --output /delivery/runtime-info.json

python -m benchmarks.behavior_1k.prepare manifest \
  --image "$BEHAVIOR_IMAGE" \
  --assets "$BEHAVIOR_SNAPSHOT" \
  --catalog "$BEHAVIOR_DELIVERY/registry/behavior_1k/catalog" \
  --runtime-info "$BEHAVIOR_DELIVERY/runtime-info.json" \
  --output "$BEHAVIOR_DELIVERY/registry/behavior_1k/manifest.json"
```

最终目录结构为：

```text
registry/behavior_1k/
  assets/
    asset.json
    snapshots/<sha256>/
  catalog/catalog.json
  manifest.json
```

资产与 catalog 必须位于 manifest 目录内部，路径不包含上级跳转或符号链接。
Manifest 记录相对路径与哈希、镜像摘要和实际检查过的任务。Registry 启动时，
快照挂载至 `/roborsi/assets/behavior_1k`，catalog 挂载至
`/roborsi/assets/behavior_1k-catalog`。

## 使用 Agent 评测

在 Agent 环境中设置 `OPENAI_API_KEY`，选择支持视觉与工具调用的模型。
通用预设接收任务、21 维动作 schema 与 RGB-D 观测。完成上面的完整 catalog 注册后，
以下命令评测全部 100 个任务，每个任务使用 seed 0，共 100 个 case。省略 `--task`
表示选择该 Benchmark catalog 中的全部任务；`--max-concurrent 1` 串行执行，
避免多个模拟器竞争同一 GPU。每次新评测请使用新的 `--run-id`：

```bash
bench eval --registry-root "$BEHAVIOR_DELIVERY/registry" \
  --results-root "$BEHAVIOR_DELIVERY/results" --run-id behavior-100-seed0 \
  --benchmark behavior_1k --seed 0 --max-concurrent 1 \
  --agent-config configs/agents/embodied/default.yaml \
  --agent-set 'runtime.llm.model="YOUR_VISION_TOOL_MODEL"' \
  --agent-set 'agent.action_timeout_s=900' \
  --timeout-s 900 --case-timeout-s 7200
```

可在同一评测命令末尾追加 `--dry-run`，先检查计划是否包含 100 个 case，而不启动
case 服务。以上是全量运行说明，不代表已经完成 100 个任务的真实评测。

原生任务步数来自 catalog；Agent 轮数限制与 Runner 超时分别控制。
预算、产物与恢复见 [Agent 配置](../../docs/agents.zh-CN.md)和
[评测指南](../../docs/evaluation.zh-CN.md)，离线检查见[测试指南](../../tests/README.md)。

原生模拟器、机器人资产与数据集遵循各自的上游许可证及访问条款。
