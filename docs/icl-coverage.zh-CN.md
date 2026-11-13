# ICL 源码覆盖核对

[English](icl-coverage.md) · [通用包工具](icl.zh-CN.md)

本次以旧 `SenseTime-FVG/RoboRSI` 仓库提交
[`8d9a0af66422e1faad7696281a76819cd165eb44`](https://github.com/SenseTime-FVG/RoboRSI/tree/8d9a0af66422e1faad7696281a76819cd165eb44)
为来源。审查时，本地保存的 `NewRPent/origin/main` 与认证读取的旧仓库 `main`
一致。以下均为实际查看过的源码，包括两个原始 ZIP 内的文件。
[机器可读清单](icl-port-sources.json)记录来源哈希和适配修改。
演示和学习后的 ICL 数据需要另外下载。

## 当前支持的 benchmark

| Benchmark | 原始仓库中的实际来源 | 核对结果 |
| --- | --- | --- |
| BEHAVIOR-1K | `benchmarks/behavior_1k/materials.py`、`official_demonstrations.py`、`r1pro_fk.py`；`robots/behavior_1k/agents/baseline/icl.py`、`icl_prompt.md`、工具及预设 | 已恢复材料转换、学习、冻结下载和任务交接。[指南](../benchmarks/behavior_1k/README.zh-CN.md#icl-与评测)。 |
| Bench2Dex | `agents/bench2dex_compat/reference-code.zip` 中的 `src/gpt6_eval/{icl,icl_workflow,demo_reader}.py`、`prompts/preparation*.md`；`benchmarks/bench2dex/{agent_bundle,run_suite}.py`；原始 Agent 和 catalog | 已恢复原始源码包、任务包准备、Agent、提示词和 suite。[冻结包](../benchmarks/bench2dex/ICL.zh-CN.md)；[原始构建说明](bench2dex-icl.zh-CN.md)。 |
| DexJoCo | `robots/dexjoco/agents/baseline/resources/reference-code.zip` 中的 `src/gpt6_eval/dexjoco_icl.py`、`dexjoco_reader.py`、`scripts/download_dexjoco_demos.py`、`prompts/dexjoco_preparation.md`；`benchmarks/dexjoco/prepare_agent_assets.py` | 已恢复原始源码包、资产准备、提示词和双会话 Agent。[指南](../benchmarks/dexjoco/ICL.zh-CN.md)。 |
| LIBERO | 原始探索/评测提示词，以及 `robots/libero/flash/{generate,relations}.py` | 原仓库没有已发布的标准 LIBERO ICL 语料或可恢复的封存包学习流水线。另有独立 Flash 材料计划生成器，已恢复，见下文。 |
| LIBERO-PRO | `scripts/libero_three/prepare_context.py`；原始 PRO context map | 保留现有冻结加载和镜像提取，补回本地目录整理：40 个包映射到 80 个 case。 |
| LIBERO Long++ | 同一整理工具；原始 Long++ context map | 保留现有加载，补回 `longpp/task-XX/sealed` 本地整理，共 12 个包。 |
| LIBERO Ultra-Long | 同一整理工具；原始 Ultra-Long context map | 保留现有加载，补回 `ultra/task-XX/sealed` 本地整理，共 9 个包。 |
| RoboCasa365 | `robots/robocasa/agents/baseline/{factory.py,agent.yaml}` 和原始 benchmark 指南 | 当前 `model_eval_agent.py`、`configs/agents/robocasa365/icl.yaml` 及 evaluation bootstrap/download 已覆盖同任务 seed-0 冻结经验。原仓库未找到包生成脚本。 |
| RoboLab | `robots/robolab/agents/{baseline,historical}/` | 当前 Agent 代码、预设和提示词与原始源码一致。历史路径明确不含探索，未发现遗漏的 ICL 构建实现。 |
| LHPR-VLN | `robots/vln_shared/{exploration,campaign,storage,execution,prepare,cli}.py`、baseline 提示词/工厂及 `scripts/run_vln_shared.sh` | 学习、冻结、导入及条件评测代码已与原始源码一致。[指南](vln_shared_evaluation.zh-CN.md)。 |
| VLN-CE：R2R / RxR | 同一共享代码及 `robots/vlnce/agents/baseline/agent.yaml` | 两种数据集条件已使用原始共享学习及冻结快照流程。 |

## LIBERO 本地材料准备

原始整理工具接收**已经生成的封存包族**，不负责从演示学习。
目标必须是源树之外的新目录：

```bash
python scripts/libero_three/prepare_context.py \
  --benchmark libero_pro --source /path/to/frozen-families \
  --destination /path/to/run/icl
```

另外支持 `libero_long_plus_plus` 和 `libero_ultra_long`。
输出包含 `packages/` 和 `package-map.json`；PRO 的 task/swap 别名共用同一份经过
校验的 family 包。现有 LIBERO 评测准备流程仍支持从已发布镜像提取。

独立的 Flash 生成器接收一次成功运行的 audit 和对应 primitive recipe：

```bash
python -m robots.libero.flash.generate \
  --audit /path/to/goal_swap_t3_s7.json \
  --recipe /path/to/goal_swap_t3_s7_recipe.jsonl \
  --destination /path/to/new-plan
```

原始 `--segments` 参数可提供实测分段记录。输出是原始 plan/anchor 材料；
本次未恢复已退役的 Flash replay runner，也不将其输出称为封存 ICL 包。
这里只恢复原始自包含生成器及 relations 代码。

## 原仓库中的其他条目

旧仓库还有 RoboDojo、RoboTwin 和 SIMPLE，均不在当前 release 注册的 benchmark
范围内。RoboDojo 有 EEF prior/recipe 消费和训练导出代码；RoboTwin 有探索提示词和
工具，但未发现封存 ICL 发布器；SIMPLE 明确排除了历史冻结 ICL。
本次构建源码移植不新增它们的原生运行时。

旧 `icl_vla_factory` / reviewed VLA policy 示例消费已有封存 ICL，依赖旧 VLA
组件，属于合成组合示例，不是遗漏的 benchmark ICL 构建器；本次不宣称完整代码对齐。

## 兼容修改

保留原始包哈希、源码 ZIP 字节、模型提示词和训练入口。
Bench2Dex 仅将 ICL 专用契约函数及清单函数移到 Agent 专用模块，并补充预设声明，
使当前原生 Benchmark 契约和 RGB 编码器保持原样。
完整的原始模型协议另存为 `icl_protocol.json`；相机路径按当前 case 作用域的
`MediaRef` 校验，再还原原始文件布局，PNG 字节和像素校验保持原样。
同时恢复原始 `roborsi/embodied_agent.py` 的外部 CLI 会话与取消支持，
并保留当前无效动作重试配置。
前一轮 BEHAVIOR 还增加了封包前关闭 Agent、拒绝嵌套复制路径，以及启动脚本权限/
全局参数处理。这些都作为适配单列，不冒充原始实现。

指南区分上游历史验证与本次实跑检查，不根据文件存在推断新的学习算法、历史分数
或原生任务成功。
