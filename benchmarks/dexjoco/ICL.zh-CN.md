# DexJoCo ICL 源码与任务 Agent

[English](ICL.md) · [Benchmark 准备](README.zh-CN.md) · [来源审查](../../docs/icl-coverage.zh-CN.md)

以下入口来自旧仓库提交 `8d9a0af66422e1faad7696281a76819cd165eb44`。
原始源码 ZIP 按字节保留在 `robots/dexjoco/agents/baseline/resources/reference-code.zip`，
SHA-256 为 `e883be2bf8f22674b24e56758721f754cce6353a45044f32025e26be7e870674`。
本次没有新编学习算法或提示词。

## 使用冻结任务包

原始 benchmark README 固定使用下面的 HF 聚合快照。仅在需要访问权限时使用现有
HF 认证：

```python
from pathlib import Path
from huggingface_hub import snapshot_download

download = snapshot_download(
    repo_id="sensenova/SenseNova-RoboRSI-Packages", repo_type="dataset",
    revision="8b81a9d779230705d4ebc427a3422073b157b02d",
    allow_patterns=["dexjoco/v1.0.0/*"], local_dir="../roborsi-packages",
)
print(Path(download) / "dexjoco/v1.0.0")
```

将 `KNOWLEDGE_ROOT` 设为输出的、直接包含 `rand_full/` 的目录。使用 benchmark
指南中的固定版本原生 checkout：

```bash
python -m benchmarks.dexjoco.prepare_agent_assets \
  --upstream "$ROBORSI_DEXJOCO_UPSTREAM" \
  --knowledge "$KNOWLEDGE_ROOT" --output ../dexjoco-agent-assets
export ROBORSI_DEXJOCO_AGENT_ASSETS="$(realpath ../dexjoco-agent-assets)"
```

原始准备工具按归档 catalog 校验全部任务包，并复制公开机器人参考资料。
通过 `--agent-config robots/dexjoco/agent.yaml` 选择原始兼容 Agent；
`configs/agents/dexjoco/compat.yaml` 是同一预设的别名。

Agent 保留原始 Astra/max、Codex CLI 0.154.0、外部文件工具，以及 1x→10x 时启动
新会话的逻辑。实际提示词是 `robots/dexjoco/agents/baseline/prompts/` 下显式声明的
文件，原始测试会与源码 ZIP 中的基线字节比较。Agent 与数值策略环境相互独立，
原始锁文件和环境说明保存在
[`agents/dexjoco_compat/`](../../agents/dexjoco_compat/README.md)。外部运行输入为：

```bash
export ROBORSI_DEXJOCO_POLICY_PYTHON=/absolute/path/to/separate-policy-venv
export ROBORSI_DEXJOCO_CODEX=/absolute/path/to/native/codex
export ROBORSI_DEXJOCO_AUTH=/absolute/path/to/your/auth.json
```

按当前 [README](README.zh-CN.md) 准备 Benchmark 服务或 registry，然后在同一
`bench eval` 命令中选择任务 Agent。只有去掉 `--dry-run` 启动评测时才会调用模型。

## 原始演示到 ICL 的构建实现

源码 ZIP 中包含构建代码，不只有包加载器：

- `dexjoco-eval/scripts/download_dexjoco_demos.py`：固定演示下载及任务清单校验。
- `dexjoco-eval/src/gpt6_eval/dexjoco_reader.py`：经过校验的按需演示读取。
- `dexjoco-eval/src/gpt6_eval/dexjoco_icl.py`：隔离学习及封包。
- `dexjoco-eval/prompts/dexjoco_preparation.md`：原始学习指令。

将归档解压到新目录。原始项目声明 Python `>=3.10,<3.12` 及独立固定依赖，
构建时使用该环境，并遵循其 README 和 `pyproject.toml`：

```bash
python -m zipfile -e robots/dexjoco/agents/baseline/resources/reference-code.zip ../dexjoco-study
cd ../dexjoco-study/dexjoco-eval
python -m pip install -e .
python scripts/download_dexjoco_demos.py --help
python -m gpt6_eval.dexjoco_icl --help
```

安装记录的 Codex CLI、配置认证并准备固定版本 upstream 后，原始参数解析器支持
以下单任务流程：

```bash
python scripts/download_dexjoco_demos.py \
  --task water_plant --regime rand_full --output "$DEXJOCO_DEMOS"
python -m gpt6_eval.dexjoco_icl \
  --config configs/dexjoco_icl_5ep_rand_full_robot_v2_two_session.json \
  --upstream "$ROBORSI_DEXJOCO_UPSTREAM" --task water_plant \
  --dataset "$DEXJOCO_DEMOS" --output "$DEXJOCO_PREPARATION" \
  --auth "$ROBORSI_DEXJOCO_AUTH"
```

输出目录必须是新目录。原始代码会校验训练清单，按需提供选定演示，要求 readiness、
`TASK.md` 和 `SOURCES.md`，再封存 `frozen/`，记录来源和模型证据。
这一步会启动真实模型学习，输出属于新实验。当前共享兼容 Agent 仍按原始发布
catalog 校验 manifest；新包应使用归档 evaluator 的 `--package`、
`--package-sha256` 参数评测，不可替换发布包固定哈希。

源码及锁文件保留原始许可说明。本次离线测试不代表真实学习成功或原生任务得分。
