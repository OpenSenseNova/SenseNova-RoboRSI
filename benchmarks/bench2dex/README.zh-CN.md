# Bench2Dex benchmark

[English](README.md)

本接入通过 RoboRSI 通用 benchmark HTTP 接口提供固定的 26 个 Bench2Dex 任务，
保留原生动作、相机标定、控制预算、样本索引映射、终止、评分和清理语义。
Agent 策略通过标准评测接口单独提供。

原生运行需要 Linux x86-64、受支持的 NVIDIA RTX GPU 及驱动、支持 GPU 的
Docker，以及固定 Isaac Sim 镜像和上游仓库的访问权限。镜像使用 Isaac Sim、
Isaac Lab 和 Bench2Dex，分别适用其许可与访问要求。资产单独下载；
安装 RoboRSI 不会自动提供原生运行环境或资产。
自定义参考 harness 采用
[Apache-2.0](../../robots/bench2dex/reference/LICENSE_NOTICE.md) 许可证。
本次迁移不构成原生任务成功或模型性能的验证。

## 准备外部资产

安装框架后，从仓库根目录执行以下命令。资产下载命令需要 `huggingface_hub`。

```bash
python -m pip install huggingface_hub
python -m benchmarks.bench2dex.assets --download \
  --source work/bench2dex/download --output work/bench2dex/registry/bench2dex/assets
python -m benchmarks.bench2dex.prepare catalog \
  --assets work/bench2dex/registry/bench2dex/assets --sample 5 \
  --output work/bench2dex/registry/bench2dex/catalog.json
```

资产工具从官方 `Bench2Dex/Assets` 数据集的固定版本
`bf65215f844d1fca30750e4cc70d7266650abf0a` 选择文件。复制前验证 Git/LFS 对象，
复制后逐文件校验，并在旁边写出文件清单和目录校验值。验证已有下载时省略
`--download`。评测期间保持封存资产和 catalog 只读。

重复 `--sample` 可选择更多原生 episode 索引，重复 `--task` 可选择
`robots/bench2dex/tasks.json` 中的任务。评测的 `--seed` 映射到该 episode 索引；
原生 RNG seed 由 `protocol.json` 固定。使用 `--smoke-steps 4` 生成的 catalog
用于诊断，会明确标记为不具备正式评分资格。

## 构建与注册

当前源码可使用已发布的 `v1.0.4-cu128` 镜像，其 benchmark 运行时代码与提交
`b513a1c7289ab11eb844856588b64c378705c405` 一致。按上文准备 assets 和 catalog 后，
使用不可变 digest 注册：

```bash
export ROBORSI_BENCH2DEX_IMAGE=ghcr.io/opensensenova/sensenova-roborsi-bench2dex@sha256:edb0701c6ea3904c53a64820775336b998e6a3d0a9ac8b3b5be2caf824db3a1b
python -m benchmarks.bench2dex.prepare manifest \
  --image "$ROBORSI_BENCH2DEX_IMAGE" --assets work/bench2dex/registry/bench2dex/assets \
  --catalog work/bench2dex/registry/bench2dex/catalog.json \
  --output work/bench2dex/registry/bench2dex/manifest.json
```

如需从修改后的源码构建并发布镜像，请使用自己的 Registry：

```bash
python -m benchmarks.package --benchmark bench2dex \
  --output work/bench2dex/build
python -m benchmarks.bench2dex.build \
  --context work/bench2dex/build --tag YOUR_REGISTRY/bench2dex:BUILD \
  --evidence work/bench2dex/build-evidence --publish
export ROBORSI_BENCH2DEX_IMAGE=$(docker image inspect YOUR_REGISTRY/bench2dex:BUILD \
  --format '{{index .RepoDigests 0}}')
python -m benchmarks.bench2dex.prepare manifest \
  --image "$ROBORSI_BENCH2DEX_IMAGE" --assets work/bench2dex/registry/bench2dex/assets \
  --catalog work/bench2dex/registry/bench2dex/catalog.json \
  --output work/bench2dex/registry/bench2dex/manifest.json
```

将 `YOUR_REGISTRY` 和 `BUILD` 替换为自己的镜像仓库与标签。Dockerfile 固定
模拟器基础镜像和原生源码版本。导出的构建上下文包含声明的 benchmark 依赖闭包
和构建输入。构建工具在发布前检查依赖与源码导入，并要求镜像的源码 digest
与导出的构建上下文一致；随后按 digest 拉取已发布镜像并重复这些检查。
校验后的源码 digest 和 adapter 版本记录在 `build-evidence/image.json` 中。
只做本地构建时省略 `--publish`；manifest 需要带仓库 digest 的镜像引用。
这些检查不执行 GPU episode。

使用预构建镜像时，应从与该镜像匹配的源码 checkout 生成 manifest。
修改 Bench2Dex adapter 后，需要重新构建并发布镜像、更新 digest 引用，
并重新生成 manifest。旧镜像会触发
`runtime adapter_version differs from Benchmark manifest`；
不要手动覆盖 manifest 中的 adapter 版本或关闭运行时校验。

资产与 catalog 必须位于 manifest 所在目录下；生成器记录 Registry 要求的相对路径，
并拒绝目录之外的路径或这些相对路径中的符号链接；registry 目录本身可以通过符号链接访问。
迁移已有布局时，先将资产与 catalog 移至该目录下，
再重新生成 manifest。manifest、下载数据、凭据和生成的证据
应保存在发布源码之外。

## 使用 Agent 评测

按 [Agent 指南](../../docs/agents.zh-CN.md) 配置模型和凭据，然后使用标准预设
或自己的 Agent：

```bash
bench eval --registry-root work/bench2dex/registry \
  --benchmark bench2dex --task bench2dex:26 --seed 5 \
  --agent-config configs/agents/embodied/default.yaml \
  --results-root work/bench2dex/results --run-id bench2dex-1 \
  --timeout-s 900 --case-timeout-s 86400 --max-concurrent 1
```

该预设用于演示通用接口，不代表复现历史分数。原始任务 ICL Agent、包准备和构建源码
见[冻结 ICL](ICL.zh-CN.md)和[从演示学习](../../docs/bench2dex-icl.zh-CN.md)，
保留原始模型、CLI 和包校验约定。
[快速开始](../../docs/getting-started.zh-CN.md)
中的 LIBERO 固定动作示例仅适用于 LIBERO。请为实际运行设置模型、上下文和决策预算。

服务提供 Bench2Dex 原生动作协议和评分规则。动作使用有限数值的绝对关节位置目标：
`{"actions": [[...], ...]}`，每个动作的维度与活跃关节数一致。上一动作块仍有剩余动作时使用 `{"continue": true}`。
Agent 可见 RGB 图像和经过标定的机器人信息；原生成功状态和评分详情仅在结束
case 时返回。

若自行管理原生环境，配置 `BENCH2DEX_ROOT`、`ISAACLAB_ROOT`、
`ISAACLAB_PYTHON`、`BENCH2DEX_ASSETS` 和 `BENCH2DEX_CATALOG`。
原生 Bench2Dex checkout 旁的 `dex2bench_dataset` 链接必须指向封存资产。
启动通用服务：

```bash
python -m roborsi.evaluation.service_host --kind benchmark \
  --factory robots.bench2dex.benchmark_service:make_bench2dex_benchmark \
  --host 127.0.0.1 --port 8000
```

随后将评测命令的 `--registry-root` 替换为
`--benchmark-url bench2dex=http://127.0.0.1:8000`。
结果、状态查询和恢复见 [评测指南](../../docs/evaluation.zh-CN.md)。
离线测试覆盖模拟原生提供方、HTTP 传输、构建边界和可重试清理，不代表该模拟器
镜像已完成 GPU 验证。
