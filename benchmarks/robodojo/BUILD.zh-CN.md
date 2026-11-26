# RoboDojo 构建与安装

[English](BUILD.md) · [使用指南](README.zh-CN.md) · [来源记录](PROVENANCE.md)

Benchmark 镜像和 Agent 环境独立部署。按[官方 RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo)的上游安装说明，
准备 `native_files.json` 固定版本的 RoboDojo、Isaac Sim、IsaacLab、CUDA PyTorch 与 cuRobo 原生环境。默认源码提交为 `726e9aabfaa642203722eb126f5eaf0f37f3e1ad`；清单中的其他 profile 必须在导出和运行时显式选择同一版本。原生源码位于 `/workspace/RoboDojo`，`Assets` 资源目录按官方配置链接到 `/roborsi/assets/robodojo`。

## Benchmark 镜像

已发布[公开 Native 基础镜像](https://github.com/orgs/OpenSenseNova/packages/container/package/sensenova-roborsi-robodojo)：
`ghcr.io/opensensenova/sensenova-roborsi-robodojo:v1.0.0-cu128-native`，固定摘要为：

```text
ghcr.io/opensensenova/sensenova-roborsi-robodojo@sha256:e82c95f893834ebe16a65441a071a6b0b2fb6dbfe3b59d846fbc8cee7cb2bfda
```

该镜像提供 Native 环境，当前 Benchmark 服务须另行覆盖构建。
原生 Python 为 `/root/miniconda3/envs/RoboDojo/bin/python`，构建及运行时须通过基础镜像的
`PATH` 选中它；服务覆盖层会清空原 ENTRYPOINT，不能仅依赖入口脚本激活 Conda。

将准备好的原生环境封装为镜像并记录实际摘要，然后导出标准构建目录：

```bash
python -m benchmarks.package --benchmark robodojo --output /data/robodojo-context
docker build /data/robodojo-context \
  --build-arg ROBODOJO_BASE_IMAGE=YOUR_NATIVE_IMAGE@sha256:ACTUAL_DIGEST \
  -t YOUR_REGISTRY/roborsi-robodojo:local
```

使用清单支持的其他 profile 时，构建增加
`--build-arg ROBODOJO_NATIVE_COMMIT=SUPPORTED_COMMIT`，导出 catalog 时设置同一
`ROBORSI_ROBODOJO_NATIVE_COMMIT`。服务会检查原生 `Assets` 路径与声明快照指向
同一目录；请保留大写目录名。

原生基础镜像是显式输入；此 Dockerfile 在其上构建 Benchmark 服务，不负责从通用 Python 镜像安装 Isaac/cuRobo。构建会检查原生源码，并拒绝 Agent SDK 或未声明代码。镜像不包含策略、提示词、skill、凭据或模型权重。服务镜像的摘要与上述 Native 基础镜像不同；成功构建后将服务镜像的实际摘要用于 registry manifest。

## 任务资源与服务

在准备好的原生环境中，从仓库根目录执行。先固化官方任务资源，再使用命令输出的快照路径：

```bash
python -m benchmarks.robodojo.prepare assets \
  --source /data/RoboDojo-assets --output-dir /data/registry/robodojo
export ROBORSI_ROBODOJO_NATIVE_ROOT=/workspace/RoboDojo
export ROBORSI_ROBODOJO_ASSETS=/data/registry/robodojo/snapshots/ACTUAL_SNAPSHOT_SHA256
```

将 `/workspace/RoboDojo/Assets` 配置为指向该快照，然后导出任务目录并启动服务：

```bash
python -m benchmarks.robodojo.prepare catalog --all-tasks --layout 0 --eval-seed 0 \
  --output /data/registry/robodojo/catalog.json --work-dir /data/robodojo-inspection
export ROBORSI_ROBODOJO_CATALOG=/data/registry/robodojo/catalog.json
python -m roborsi.evaluation.service_host --kind benchmark \
  --factory robots.robodojo.benchmark_service:make_robodojo_benchmark \
  --host 127.0.0.1 --port 18800
```

## Agent

使用 Linux x86-64 和原控制器的 Python 3.10.12：

```bash
python3.10 -m venv .venv-robodojo
source .venv-robodojo/bin/activate
python -m pip install uv==0.8.9
python -m pip install --require-hashes \
  -r robots/robodojo/agents/submission/requirements.lock
python -m pip install --no-deps --no-build-isolation -e .
python -m benchmarks.robodojo.install_agent --vla-backend simate_beta
```

安装程序核对固定资源，校验 Codex 0.160.1 及两个配套二进制，构建独立工具环境并安装选定 VLA；不请求模型 API。权重下载可能很大。原始源码、归一化、tokenizer、依赖版本及哈希保存在 submission 的 `pi05/`、`simate/`、`tools/` 和 `vendor/` 中。`--checkpoint-dir` 指定外部权重时，评测也要通过 `parameters.checkpoint_path` 传入同一绝对路径。

Simate 的安装脚本会准备模型运行环境与权重。使用外部权重目录时，安装与评测应指定同一路径。

## 验证

```bash
python -m benchmarks.robodojo.provenance
pytest tests/unit_tests/robots/robodojo -q
python -m benchmarks.robodojo.prepare --help
python -m benchmarks.robodojo.install_agent --help
```

离线测试验证协议和资源契约。原生验收仍需实际镜像、授权资源、受支持 GPU/驱动、模型服务以及真实任务运行。
