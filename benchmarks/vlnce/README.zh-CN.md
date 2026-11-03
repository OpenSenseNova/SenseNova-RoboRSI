# VLN-CE (R2R / RxR)

[English](README.md) · [文档索引](../../docs/README.md) · [构建参考](BUILD.md)

本接入通过公共 Benchmark 服务、Registry 和 Runner 运行 VLN-CE (R2R / RxR)，
采用隔离的原生进程及原生评分。
本分支已提供首批代码与使用指南。
固定版本的交付物及安装要求见下文和[共享评测指南](../../docs/vln_shared_evaluation.zh-CN.md)。

## 任务、动作与评分

R2R 和 RxR 各使用一个 **100 任务的独立目录**，seed 均为 **0**。
R2R 保留选定的 `rand100` 标注、起点和参考路径；RxR 使用选定的
`val_unseen` guide episodes，保留原语言和行号。两者均非上游验证集全量。
原生 VLN-CE commit 为 `729d141b2ee10628061ada74dd3a5b9f70faeba5`。

动作格式为 `{"actions": ["move_forward", "turn_left", "stop"]}`，每批 1–32 个。
R2R 支持 `stop`、`move_forward`、`turn_left`、`turn_right`；RxR 还支持
`look_up`、`look_down`。移动幅度由原生配置决定，每个物理动作计入预算，episode
结束即停止该批。观测含指令和一张 512×512 RGB 图像，不向模型提供 oracle 进度或参考路径。

成功来自原生 `success`；RoboRSI score 为 `float(success)`。SPL、nDTW、距离等
原生指标保留在结果中。STOP 结束 episode，达到原生步数上限属于截断；未终止的
episode 保持未评分，不能计为任务失败。

下列标准 Registry 使用 500 个物理动作预算。独立的共享 campaign 仅对已评分失败任务
以全新 5000 步预算重跑，两阶段保持任务、原始行号、seed 和冻结经验一致。
详见[共享评测协议与输出](../../docs/vln_shared_evaluation.zh-CN.md)。

## 环境与镜像

使用 Linux x86_64、NVIDIA GPU／驱动、Docker 和 NVIDIA Container Toolkit，
主控／Agent 使用 Python 3.11；已验证机器为 RTX 4090。
镜像的原生环境为 Python 3.8.20 / Habitat-Lab and Habitat-Sim 0.1.7 / NumPy 1.23.5，HTTP 服务另用 Python 3.11。
不要将旧模拟器依赖混入 Agent 环境。为镜像、许可场景及资产副本预留磁盘空间。

在包含导航接入和已接受框架修复 #36 的源码根目录安装固定依赖：

```bash
python3.11 -m venv .venv-vln
source .venv-vln/bin/activate
python -m pip install -r agents/vln_shared/requirements.txt
python -m pip install --no-deps -e .
python -m pip check
```

#36 合并前，使用[隔离源码安装方式](../../docs/vln_shared_evaluation.zh-CN.md#已提交的交付物)
或[贡献者附件](https://github.com/panda7777777/SenseNova-RoboRSI/releases/tag/vln-delivery-20261102)中的已验证框架 wheel。
按[固定 CLI 安装说明](../../docs/vln_shared_evaluation.zh-CN.md#安装固定-cli)
安装并核验完整 Codex CLI 0.161.0，在实际 Agent 账号下为该程序认证；凭据不进入交付包。

选择无符号链接路径分量的绝对交付目录。如镜像要求权限，先用有拉取权限的账号执行
`docker login ghcr.io -u YOUR_GITHUB_USERNAME`。

```bash
export VLN_DELIVERY=/absolute/path/outside/checkout/vln-delivery
export VLN_IMAGE=ghcr.io/opensensenova/sensenova-roborsi-vln-ce@sha256:f54607562ca465101d6e1091f97d776755c2b00517f1852643bd8e0161c0e787
mkdir -p "$VLN_DELIVERY"
docker pull "$VLN_IMAGE"
```

正式 tag 为 `v1.0.0-cu120`，评测固定使用上面的 digest。
镜像源码 commit 为 `1435a63f853649c9e04fe135b664b9835d10fde7`。
[Dockerfile](Dockerfile)、[原生构建配方](Dockerfile.native)、
[固定输入](release-inputs.json)、[原生版本清单](native-environment.json)、
[服务依赖锁](service-requirements.txt) 和[构建命令](BUILD.md)记录重建材料。
使用现有镜像不需要重新构建。

## 资产、任务目录与 Registry

复用已获许可的 Matterport3D 场景。源码及数据获取说明见
[固定 commit 的 VLN-CE README](https://github.com/jacobkrantz/VLN-CE/blob/729d141b2ee10628061ada74dd3a5b9f70faeba5/README.md)。
场景布局为 `mp3d/<scene>/<scene>.glb`，配置的场景根目录必须与标注解析后的路径一致。
保留上游许可说明。

所需输入包括该 commit 的 `habitat_extensions/`、R2R 的 `rand100.json.gz` 和
`rand100_gt.json.gz`、RxR 的 `val_unseen_guide.json.gz` 和
`val_unseen_guide_gt.json.gz`，以及精确的 episode 选择及顺序。
R2R 标注 SHA-256 为
`b563a76dbbdec5da61ee656ccf5e096b62070423e633987fb437ca1003b4fc67`；
RxR 标注 SHA-256 为
`93abddce58e8232867a8738daa8107eb82953693a9d5e70779d1eb00386b222a`。

上述精确输入由独立的[评测输入包 v1.0.0](https://github.com/panda7777777/SenseNova-RoboRSI/releases/tag/vln-evaluation-inputs-v1.0.0)提供：105 个原始文件、
约 25.2 MB，另附完整校验索引、来源和许可说明。它与 HF 训练经验包分别管理。
原生文件、标注、任务顺序和原始行号保持不变；不需要作者的 `gpt6-test` 或历史运行目录。
已有校验通过的输入包时直接复用，跳过下载。场景不在包内，仍使用已有的获许可副本。

```bash
export VLN_INPUT_DIR="$VLN_DELIVERY/input-download"
mkdir -p "$VLN_INPUT_DIR"
curl -fL 'https://github.com/panda7777777/SenseNova-RoboRSI/releases/download/vln-evaluation-inputs-v1.0.0/vln-evaluation-inputs-v1.0.0.tar.gz' \
  -o "$VLN_INPUT_DIR/vln-evaluation-inputs-v1.0.0.tar.gz"
printf '%s  %s\n' 'c20368128789105a55b0e0d507c7feaad6a8fbe28819f5d691bc6892450e1f66' \
  "$VLN_INPUT_DIR/vln-evaluation-inputs-v1.0.0.tar.gz" | sha256sum -c -
tar -xzf "$VLN_INPUT_DIR/vln-evaluation-inputs-v1.0.0.tar.gz" -C "$VLN_INPUT_DIR"
export VLN_INPUT_PACKAGE="$VLN_INPUT_DIR/vln-evaluation-inputs-v1.0.0"
export VLN_SCENES=/absolute/path/to/existing/scene_datasets
export VLN_REGISTRY="$VLN_DELIVERY/r2r-registry"
python -m robots.vln_shared.public_assets \
  --package "$VLN_INPUT_PACKAGE" --dataset r2r --scenes-root "$VLN_SCENES" \
  --output "$VLN_REGISTRY" --image "$VLN_IMAGE" --budget 500
```

准备命令逐文件校验小输入和所需场景；任一缺失或哈希不符即报错，不自动下载或替换。
默认硬链接场景，输出目录必须与场景在同一文件系统；跨文件系统时只有显式增加
`--copy-scenes` 才复制，并需预留空间。原场景与输出快照都应保持不变。
输出必须是新目录且各级无符号链接；manifest 位于 `$VLN_REGISTRY/vlnce/manifest.json`，
资产相对路径为 `assets`，在容器内只读挂载至 `/roborsi/assets/vlnce`。

5000 步阶段使用同一命令，改为 `--budget 5000` 和新的输出目录；只对第一阶段
有原生评分的失败任务执行第二阶段，保持 Agent wheel、task 和 seed 相同。
不要将错误／超时／未评分任务当成阶段一失败；它们需要单独处理。
本输入包不包含新探索所需的完整训练目录，冻结 ICL 评测不依赖它们。
旧 campaign 的迁移／场景映射方式仍见[构建参考](BUILD.md)。

RxR 使用同一个输入包和场景根目录，将 `--dataset` 改为 `rxr`，
输出改为独立的 `$VLN_DELIVERY/rxr-registry`。两者 benchmark ID 均为 `vlnce`，
不能放在同一 Registry 搜索根目录。两个目录各有 100 个任务。

发现 Registry、核对镜像身份并检查完整任务目录：

```python
import os
from roborsi.benchmarks.registry import BenchmarkRegistry

registry = BenchmarkRegistry([os.environ["VLN_REGISTRY"]])
print(registry.discover())
locked = registry.lock("vlnce")
assert locked.image_reference == os.environ["VLN_IMAGE"]
tasks = locked.list_tasks()
assert len(tasks) == 100
print(tasks[:3])
```

## 使用 Agent 评测

标准入口 `robots/vlnce/agent.yaml` 继承本 benchmark 基线及
[共享导航工厂](../../robots/vln_shared/agents/baseline/README.md)，通过 `EmbodiedAgent`
调用 **Codex CLI 0.161.0**、**gpt-6-astra**、**max** 推理、MCP 工具；模型轮数不限。
当前仅交付这一 backend，未声明其他 backend／模型组合已经验证。

标准源码预设**不含 ICL**。复现冻结经验评测时，从[贡献者附件](https://github.com/panda7777777/SenseNova-RoboRSI/releases/tag/vln-delivery-20261102)下载匹配的
Agent wheel，执行 `python -m pip install --no-compile /absolute/path/to/AGENT.whl`，
并将下面命令改为 `--agent-config bundle:vln_shared_r2r`。
经验来源为 [vln-ce/v1.0.0](https://huggingface.co/datasets/sensenova/SenseNova-RoboRSI-Packages/tree/495b4e2a9a3652726f52dd6363f7e2c5a77c2b34/vln-ce/v1.0.0)，
HF commit 固定为 `495b4e2a9a3652726f52dd6363f7e2c5a77c2b34`。
单独下载 HF payload 时遵循[下载和校验步骤](../../docs/vln_shared_evaluation.zh-CN.md#已提交的交付物)；
下载文件本身不会自动把经验挂到 Agent 预设。

在仓库根目录先检查一个实际任务的计划：

```bash
export VLN_CODEX=/absolute/path/to/pinned/codex
bench eval --registry-root "$VLN_REGISTRY" --benchmark vlnce \
  --task vlnce:r2r-7 --seed 0 \
  --agent-config robots/vlnce/agent.yaml \
  --agent-set "agent.codex_cli.executable=\"$VLN_CODEX\"" \
  --results-root "$VLN_DELIVERY/results" --run-id navigation-smoke-001 \
  --max-concurrent 1 --timeout-s 900 --case-timeout-s 7200 --dry-run
```

移除 `--dry-run` 开始模型评测，每次使用新 `--run-id`。
移除 `--task` 可运行该目录全部 100 个任务，seed 为 0。
任务预算是 500 个物理动作；`--timeout-s`、`--case-timeout-s` 分别控制操作及 case
的墙钟时限，不替代动作预算。

RxR 使用其独立 Registry、`--task vlnce:rxr-198`、seed 0，冻结经验入口为
`--agent-config bundle:vln_shared_rxr`；不能把 R2R wheel 用于 RxR。
两套目录分别执行并保留各自汇总。

标准命令只执行一个固定预算。训练集探索／冻结经验导入及条件式 500/5000 两阶段流程见
[共享 campaign 指南](../../docs/vln_shared_evaluation.zh-CN.md)。该迁移入口需要原 campaign
输入，不会自动下载缺失的公开资产。

## 结果与验收

检查 `<results-root>/<run-id>/summary.json`、各 case 的 outcome、Runner 动作／观测、
原生指标和图片工件；保留 CLI `codex-cli/call-*/` 的 runtime／stdin／events／stderr／result
以及清理状态。技术错误、超时、缺失评分与原生任务失败分开统计；未获得的 token 用量为
`null`，不能写成零。

[交付记录](../../docs/vln_release_20261102.json)和[可下载容器验收摘要](https://github.com/panda7777777/SenseNova-RoboRSI/releases/tag/vln-delivery-20261102)
覆盖最终镜像的 GPU reset、固定动作、HTTP 图像／原生评分、直接原生对照、最终资产校验和
资源清理，不代表模型精度。三个 Agent wheel 已通过仓库外安装及资源校验。
目前报告的真实模型证据仅有旧构建上的 R2R 原生／HTTP 单例；各数据集的最终镜像模型验收和
完整 benchmark 成绩尚未全部具备。合并后的全量复测由负责人执行。
离线检查见[测试指南](../../tests/README.md)。

模拟器、原生源码和数据集保留各自许可及访问条件。
公开输入包用于冻结测试目录；重新训练集探索仍需另行准备对应训练输入。
