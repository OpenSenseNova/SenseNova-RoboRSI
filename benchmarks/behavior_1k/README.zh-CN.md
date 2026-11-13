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

## ICL 与评测

可以下载冻结的任务包，也可以从该任务的 200 条公开演示中学习。
两种方式生成相同的封存 ICL 格式。任务专用预设使用现有 Codex CLI planner，
评测前必须附加学习包。上面的通用 Agent 命令仍可独立使用。

### 下载冻结任务包

来源集成固定使用聚合仓库
[SenseNova-RoboRSI-Packages](https://huggingface.co/datasets/sensenova/SenseNova-RoboRSI-Packages/tree/e977b5ad667b4ce753ee947cbb0b95d4fc0fe98c/behavior/v1.0.0)
的指定版本。在 Agent 环境运行，`BEHAVIOR_DELIVERY` 按前文设置在 checkout 外。
缓存可复用，学习、预设和构建输出必须使用新目录：

```bash
ICL_PACKAGE=$(python -m robots.behavior_1k.agents.baseline.icl download \
  --task sorting_vegetables --version v1.0.0 \
  --revision e977b5ad667b4ce753ee947cbb0b95d4fc0fe98c \
  --cache-dir "$BEHAVIOR_DELIVERY/icl-cache")
```

命令输出校验后的任务目录，不重新封包。缓存旁的 receipt 记录固定 HF revision、
subpath 和 manifest 哈希。会校验 payload 哈希、任务标识、学习阶段、21 维动作
profile，以及 `TASK.md`、`SOURCES.md`。仅在仓库需要时进行 HF 身份认证。

### 或者从 200 条演示学习

使用独立环境：材料转换需要原生 BEHAVIOR 源码、Isaac Sim 的 Lula 扩展和
R1Pro 资产；学习只需 Agent、Codex CLI 和已准备的材料。
在实际使用相应依赖的环境中安装附加锁文件：

```bash
python -m pip install --require-hashes -r benchmarks/behavior_1k/requirements-icl.txt
```

从 `behavior-1k/2026-challenge-demos` 下载固定 revision
`4f50b44796641a4d526a19d9aeadc8aa51e2f2c2` 到 `BEHAVIOR_DEMOS`：

```python
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="behavior-1k/2026-challenge-demos", repo_type="dataset",
    revision="4f50b44796641a4d526a19d9aeadc8aa51e2f2c2",
    local_dir=os.environ["BEHAVIOR_DEMOS"],
)
```

必须先下载指定 revision。材料构建会记录配置声明的 revision，并在存在
`selection.json` receipt 时检查其是否冲突；普通本地目录本身不证明 HF 来源。
`BEHAVIOR_NATIVE` 应包含固定原生仓库的 `OmniGibson/` 目录，
`BEHAVIOR_DATA` 指向前文下载的资产。
每个任务必须有 200 个不同的训练 instance ID，范围为 1–300；ID 允许不连续，
转换保留真实值，不重新编号。只暴露该任务的 RGB 视频、标注和数值记录。
离线 FK 将原始 23 维关节目标转换为评测的 21 维格式，并记录资产和转换器哈希，
不运行评测 episode。

在原生材料准备环境运行：

```bash
python -m benchmarks.behavior_1k.materials --task sorting_vegetables \
  --source "$BEHAVIOR_DEMOS" --native-root "$BEHAVIOR_NATIVE" \
  --data-root "$BEHAVIOR_DATA" --output "$BEHAVIOR_DELIVERY/materials"
```

在 Agent 环境通过独立 npm prefix 安装 `@openai/codex@0.162.0`，将其 `bin`
加入 `PATH`；`codex --version` 必须输出 `codex-cli 0.162.0`。
设置 `OPENAI_API_KEY` 并选择获授权使用的模型：

```bash
python -m robots.behavior_1k.agents.baseline.icl learn \
  --agent-set 'agent.model="YOUR_MODEL_ID"' \
  --materials "$BEHAVIOR_DELIVERY/materials" \
  --case-dir "$BEHAVIOR_DELIVERY/icl-run" --output "$BEHAVIOR_DELIVERY/icl-package"
ICL_PACKAGE="$BEHAVIOR_DELIVERY/icl-package"
```

学习提示词允许网络参考，要求在 `learned/` 保存有演示依据的 `TASK.md`、
`SOURCES.md` 和可复用程序。提供记录与抽帧工具，逐条查看演示，解码前校验视频哈希。
学习结束后先关闭 Agent，再只封存 `learned/`；演示数据、对话记录和生成的字节码
不进入交接包。学习 planner 默认预算 960 秒，可用 `--agent-set` 调整。

### 准备、导出并评测任务 Agent

两种来源都通过以下命令生成独立源码目录：

```bash
python -m robots.behavior_1k.agents.baseline.icl preset \
  --package "$ICL_PACKAGE" --baseline robots/behavior_1k/agent.yaml \
  --output "$BEHAVIOR_DELIVERY/task-agent"
bench eval --registry-root "$BEHAVIOR_DELIVERY/registry" \
  --results-root "$BEHAVIOR_DELIVERY/results" --run-id behavior-icl-001 \
  --benchmark behavior_1k --task behavior_1k:sorting_vegetables --seed 0 \
  --timeout-s 900 --case-timeout-s 7200 \
  --agent-config "$BEHAVIOR_DELIVERY/task-agent/robots/behavior_1k/agent.yaml" \
  --agent-set 'agent.model="YOUR_MODEL_ID"'
```

预设使用与学习相同的固定版本 Codex CLI。评测禁用网页搜索，使用当前公开观测和
已学习的任务文件。每个 case 获得校验后的独立可写副本；任务或 benchmark 不匹配时，
在模型会话启动前拒绝运行。工作副本不是操作系统访问边界；部署评测 Agent 时，
应使其无法访问演示、模拟器内部状态或历史运行目录。

需要独立安装 Agent 时，可导出声明的输入并安装 wheel：

```bash
python -m roborsi.agents.package \
  --config "$BEHAVIOR_DELIVERY/task-agent/robots/behavior_1k/agent.yaml" \
  --name behavior_official_icl --output "$BEHAVIOR_DELIVERY/agent-build"
python -m build --wheel "$BEHAVIOR_DELIVERY/agent-build"
python -m pip install --no-compile "$BEHAVIOR_DELIVERY"/agent-build/dist/*.whl
# 安装后使用：--agent-config bundle:behavior_official_icl
```

使用 `--no-compile`，防止安装在封存 payload 中生成额外 `.pyc` 文件。
学习程序应在可写 case 副本中执行。预设、学习文件、视频和日志放在代码 checkout 外。
通用封包与快照见 [ICL 指南](../../docs/icl.zh-CN.md)。本次移植运行离线检查；
以上命令不代表已完成真实模型学习、原生 FK 转换或原生任务评测。

原生模拟器、机器人资产与数据集遵循各自的上游许可证及访问条款。
