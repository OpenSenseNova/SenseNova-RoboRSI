# LHPR-VLN 交付入口

正式版本为 `v1.0.0-cu120`，平台 `linux/amd64`。镜像源码为 `1435a63f853649c9e04fe135b664b9835d10fde7`。

```bash
docker pull ghcr.io/opensensenova/sensenova-roborsi-lhvln@sha256:7d72dd910d5c9048ca6ec40ad158214497c3a9ac276641df1429314f2d7a3695
```

评测固定使用以上 digest。GHCR 包当前仍为私有，公开权限由负责人协调。
`release-inputs.json` 固定原始与规范化 native 镜像；`Dockerfile.native`、
`native-fixes.txt` 和 `native-environment.json` 记录依赖修复及完整版本清单。
完整重建、场景映射和资产 manifest 命令见 [配方说明](README.md)。

冻结经验包为 `lhvln/v1.0.0`，固定 HF commit `495b4e2a9a3652726f52dd6363f7e2c5a77c2b34`。
[HF PR #1](https://huggingface.co/datasets/sensenova/SenseNova-RoboRSI-Packages/discussions/1) 已提交并重新下载校验，等待管理员合并。
[共享评测指南](../../docs/vln_shared_evaluation.zh-CN.md) 给出安装、下载与校验、
Agent wheel 导出、标准评测入口、500/5000 阶段及结果核验方法。

容器运行依赖独立的[框架 PR #36](https://github.com/OpenSenseNova/SenseNova-RoboRSI/pull/36)。
验收使用导航源码加该修复：最终资产校验最多 30 秒，仍受 case 总时限约束。
合并前可按共享指南在独立工作树应用固定修复 commit。宿主原生流程不依赖此 Docker 修复。

[验收记录](../../docs/vln_release_20261102.json) 包含镜像、源码和构建上下文摘要，
以及真实容器中 HTTP/直接原生执行的图片、动作数、评分对照和资源释放检查。
这些检查不代表完整 benchmark 精度；合并后的全量复测由负责人完成。

Agent／框架 wheel 与验收摘要见[贡献者交付附件](https://github.com/panda7777777/SenseNova-RoboRSI/releases/tag/vln-delivery-20261102)，安装前校验 `SHA256SUMS`。
