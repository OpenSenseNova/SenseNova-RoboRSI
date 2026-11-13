# 下载 Bench2Dex 冻结 ICL 包

> 来源：旧仓库 `8d9a0af66422e1faad7696281a76819cd165eb44` 的同名文档。以下保留原始工具、版本和流程；其中历史发布与验证记录不代表本次移植已运行模型或仿真器。来源哈希和兼容修改见 [移植清单](../../docs/icl-coverage.zh-CN.md).

[English](ICL.md)。[固定版本 Hugging Face 链接](https://huggingface.co/datasets/sensenova/SenseNova-RoboRSI-Packages/tree/861d045d47466673826d0e9c6374fa0dba944096/bench2dex/v1.0.0)。

包目录：`bench2dex/v1.0.0`；设置：`task_icl_v3`；26 个任务，ZIP 总计约 1.194 GB。
HF 完整 commit：`861d045d47466673826d0e9c6374fa0dba944096`。
原始 33 个发布文件完整保留，包括 26 个 ZIP 及其 3,610 个 payload 文件，字节未改动。
无需重新生成 ICL。该材料与官方 `Bench2Dex/Assets` 运行资产、GHCR 原生镜像分别管理。

安装 `huggingface_hub`，使用有权限的账号执行 `hf auth login`；仓库目前为私有。
在 RoboRSI 代码目录运行：

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="sensenova/SenseNova-RoboRSI-Packages",
    repo_type="dataset",
    revision="861d045d47466673826d0e9c6374fa0dba944096",
    allow_patterns=["bench2dex/v1.0.0/*", "scripts/verify_package.py", "packages.json"],
    local_dir="work/roborsi-packages",
    token=True,
)
```

校验整个发布包并生成单任务 Agent：

```bash
python work/roborsi-packages/scripts/verify_package.py \
  work/roborsi-packages/bench2dex/v1.0.0 \
  --manifest-sha256 e537864c8744b1918ba1fd3571915bafa39fd2af5d45f66155973eac4c507150
python -m benchmarks.bench2dex.agent_bundle --task 26 \
  --source work/roborsi-packages/bench2dex/v1.0.0 --output work/task26-agent
```

`run_suite --icl-source` 同样应指向上述**版本目录**。聚合仓库根目录没有直接的
`packages/`，不能套用独立仓库的 `--repo-id` 下载方式。Agent 会按代码内固定目录
校验所选 ZIP 及全部 payload。输出目录必须尚未存在。只需一个任务时，可将
`allow_patterns` 改为 `bench2dex/v1.0.0/packages/task_26.zip`，仍由 Agent 校验；
完整发布校验器则需要全部发布文件。

评测时记录完整 HF SHA、版本目录、代码 commit 和镜像 digest，保留 ZIP 原始字节。
包内旧 README 介绍独立评测器，共享仓库的当前用法见包内 `RELEASE.md`。
原始 `LICENSE_NOTICE.md` 仍记录许可待定；本次私有上传未解决作者及第三方公开分发权限。

上传验证：按上述 SHA 从远端重新下载，35 个发布文件、26 个任务 manifest 和
3,610 个 payload 哈希全部一致；从下载包生成 task-26 Agent 也通过。
本次未运行新的模拟器或 Astra 测试；参见[验证范围](https://github.com/SenseTime-FVG/RoboRSI/blob/8d9a0af66422e1faad7696281a76819cd165eb44/benchmarks/bench2dex/VALIDATION.md)和
[框架前置条件](https://github.com/SenseTime-FVG/RoboRSI/blob/8d9a0af66422e1faad7696281a76819cd165eb44/benchmarks/bench2dex/FRAMEWORK_REGISTRATION.md)。
