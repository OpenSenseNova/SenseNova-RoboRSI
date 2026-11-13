# 构建和使用 ICL 材料

[English](icl.md) · [全部指南](README.md)

各 benchmark 的原始构建代码、源码包与现有等价流程见[来源覆盖核对](icl-coverage.zh-CN.md)。

ICL 包保存选定的笔记、技能、图片、程序和辅助文件。构建、加载不会执行包内程序，
也不会调用模型。材料范围和上下文导出项由调用方明确指定。

## 离线体验文件流程

安装仓库后，在根目录执行，输出目录必须是新目录：

```bash
python -m examples.components.study_and_pack --dry-run --output work/icl-demo
python -m roborsi.cli.icl verify work/icl-demo/package
python -m roborsi.cli.icl inspect work/icl-demo/snapshot
```

示例使用合成笔记和不执行的程序，依次构建封存包、修改独立工作副本、保存新快照，
无需仿真器或模型凭据。实际学习需要调用方提供 Agent 和工具；
`study_and_pack.py` 演示通过 deferred `ready` 工具确定学习结束边界。

## 构建、复制和快照

```python
from pathlib import Path
from roborsi.components.icl import (
    ContextExports, ContextPackage, build_package,
    materialize_package, snapshot_workspace,
)
from roborsi.components.icl.adapters import context_from, make_icl_tools

package = build_package(
    Path("selected-materials"), Path("reference-v1"), package_id="reference-v1",
    exports=ContextExports(memory=(("Task", "TASK.md"),)),
    metadata={"dataset": "caller-selected"},
)
verified = ContextPackage.load(package.root)
workspace = materialize_package(verified, Path("case-001/workspace"))
context = context_from(workspace, planner="api")
tools = make_icl_tools(workspace, name_prefix="reference")
(workspace.root / "TASK.md").write_text("New experience", encoding="utf-8")
revision = snapshot_workspace(workspace, Path("reference-v2"), package_id="reference-v2")
```

构建会复制整个输入目录，只有显式导出项进入上下文。技能文件、技能目录以及文本/
图片初始上下文分别通过 `skills`、`skill_paths`、`initial_context` 声明。
程序依赖由调用方准备。源目录和目标目录不得嵌套，目标必须不存在。
空目录树、缺失导出项、链接、特殊文件和不可移植的名称冲突会报错。
内容、导出项、标识和元数据相同时，manifest 字节一致，不生成时间戳。

封存格式仍是 `manifest.json` 加 `payload/`，类型为 `rpent.icl.context`，
schema version 为 1。加载会校验全部 SHA-256 并拒绝额外文件。
旧的 `roborsi.agents.context_package` 导入仍然有效。工作副本是相互独立的普通文件；
所有写入停止后才能做快照。快照保留导出项，并记录
`metadata.icl.parent_manifest_sha256`；删除导出文件时必须更新声明。
不要在封存 payload 内生成 Python 字节码。

## 接入 Agent

将 `context.memory`、`skills`、`skill_paths`、`initial_context` 传给
`EmbodiedAgent.run` 或 `start_episode`，并显式注册 `tools`。
Runner 评测使用构造参数 `evaluation_memory`、`evaluation_skills`、
`evaluation_skill_paths`、`evaluation_initial_context`，见
[icl_build_and_load.py](../examples/components/icl_build_and_load.py)。

`context_from` 支持 `api`、`codex`、`codex_cli`、`claude_code`，技能模式需明确选择：

| `skill_mode` | 行为 |
| --- | --- |
| `declared` | 保留声明；目录技能要求 API planner。 |
| `preload` | 发现声明的技能集合并预加载 `SKILL.md`。 |
| `on_demand` | API planner 按需读取声明的技能目录。 |
| `none` | 省略技能，保留记忆和初始上下文。 |

Codex CLI 的目录导出项应选 `preload` 或 `none`。按
[Agent 指南](agents.zh-CN.md) 配置 `expected_version`，并将模型 turn/token
上限设为 null。只读工具提供 `<prefix>_list`、`<prefix>_read_text`、
`<prefix>_read_image`，支持分页和截断，并在读取时重新检查路径。
封存包读取会校验哈希，工作副本读取当前内容；写入和执行工具由调用方提供。

## CLI 和固定来源

使用 `roborsi-icl` 或 `python -m roborsi.cli.icl`。
成功向 stdout 输出 JSON，失败向 stderr 输出诊断并返回非零状态。
构建 spec 中的路径相对 spec 文件解析：

```json
{
  "source": "selected-materials", "destination": "reference-v1",
  "package_id": "reference-v1",
  "exports": {"memory": [{"title": "Task", "path": "TASK.md"}]},
  "metadata": {"dataset": "caller-selected"}
}
```

```bash
roborsi-icl build --spec build.json
roborsi-icl verify reference-v1
roborsi-icl inspect reference-v1
roborsi-icl fetch --source-json source.json --cache-dir icl-cache
roborsi-icl fetch --source-json source.json --cache-dir icl-cache --offline
roborsi-icl materialize reference-v1 case-001/workspace --workspace-record case-001/workspace.json
roborsi-icl snapshot --workspace-record case-001/workspace.json reference-v2 --package-id reference-v2
roborsi-icl import-policylab shared-task imported-package --package-id imported-v1
```

`source.json` 选择一种来源：

```text
{"type": "local", "path": "reference-v1"}
{"type": "http", "url": "https://host/package.zip", "archive_sha256": "<64 lowercase hex>"}
{"type": "hf", "repo_id": "owner/repository", "revision": "<40 lowercase commit hex>", "subpath": "packages/reference", "repo_type": "dataset"}
```

HTTP/HF 还支持 `manifest_sha256`。ZIP 根目录必须是完整包结构。
下载上限默认 1 GiB，解包内容默认 4 GiB，通过 `--max-download-bytes`、
`--max-unpacked-bytes` 调整。缓存 receipt 在包外记录来源，每次命中都会重新校验；
`--offline` 要求已有精确匹配固定来源的完整缓存。

工作副本 receipt 必须是新文件，并位于工作副本和源包之外。移动工作副本后应更新
receipt 的 root。快照支持 `--exports` 和 `--metadata` JSON 文件。
PolicyLab 导入接受单个任务的 `workspace/` 和 `provenance.json`，保留未知来源字段，
并将原始附件复制到 `payload/__policylab_source__/`；不会重写历史路径，也不会
独立证明来源声明的 training-only 属性。

Agent preset 中通过 `resources` 声明包，以纳入指纹和 wheel。生成数据和轨迹应放在
checkout 外。完整任务学习流程见 [BEHAVIOR-1K](../benchmarks/behavior_1k/README.zh-CN.md#icl-与评测)。
离线示例和测试验证文件及接口行为，不代表模型或仿真任务成功。
