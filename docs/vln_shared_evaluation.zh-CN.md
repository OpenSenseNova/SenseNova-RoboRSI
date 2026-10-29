# VLN 共享探索与两阶段测评

入口为 `scripts/run_vln_shared.sh`。默认每个数据集完成一次训练集探索并冻结文件，再对全部测试任务执行最多 500 步。仅原生评分失败的任务从初始状态重跑最多 5000 步；两次最多累计 5500 个动作。支持选择数据集、限制每个数据集的测试 episode 数量及仅运行 500 步阶段。超时、工具异常、提前退出和缺少评分均保持未评分，不触发第二阶段。

## 标准 Agent 入口

| Benchmark | `bench eval --agent-config` 入口 | 数据集 |
| --- | --- | --- |
| VLNCE | `robots/vlnce/agent.yaml` | R2R、RxR |
| LHPR-VLN | `robots/lhpr_vln/agent.yaml` | LHPR |

两个入口分别继承各自的 `agents/baseline/agent.yaml`，再复用 `robots/vln_shared/agents/baseline/` 中唯一一份工厂、提示词和工具策略。旧路径 `agents/vln_shared/agent.yaml` 保留为兼容别名；共享依赖锁仍在 `agents/vln_shared/requirements.txt`。

一键流程在探索和测评时均加载对应的标准入口，再生成 `agent-input/` 下的自包含预设。探索提示词通过 `exploration_prompt` 资源声明并随包导出。标准入口本身不含冻结训练记忆；共享探索实验应使用流程生成的 `agent-input/<dataset>/agent.yaml` 或其 wheel，无需复制各 Benchmark 的 Agent 实现。

标准入口和兼容别名的 `source_root` 为仓库根目录，导出目录须位于仓库外；wheel 只包含声明的文件。

实现和目录调整会改变代码／配置指纹。新实验使用新的输出目录；已有冻结预设、wheel 和历史结果保持不变，续跑仍要求原代码和输入指纹一致。

服务和 manifest 的 `adapter_version` 现统一使用 `benchmark_source_digest`，
只计算公共登记中的服务文件，因此完整仓库和独立导出包的版本一致。修改 Agent、
探索或调度代码不影响服务版本；campaign 仍单独校验完整导航模块及已声明 Agent
资源，防止变更代码后续跑旧实验。本次更新后请使用新输出目录，并针对匹配的
服务源码重新生成交付 manifest；历史结果保留作原有证据。

## 安装与启动

宿主环境使用 Python 3.11；原生模拟器保留独立环境。已验证 VLNCE 为 Python 3.8.20 / Habitat 0.1.7，LHPR 为 Python 3.9.23 / Habitat-Sim 0.3.0。具体包版本见两个 `benchmarks/` 目录的 `native-environment.json`。

```bash
uv venv --python 3.11 .venv-vln
uv pip install --python .venv-vln/bin/python -r agents/vln_shared/requirements.txt
uv pip install --python .venv-vln/bin/python --no-deps -e .
export ROBORSI_PYTHON="$PWD/.venv-vln/bin/python"

# 只读导入现有任务、标注与原生源码；自动完成探索、500、失败案例 5000 和汇总。
scripts/run_vln_shared.sh run \
  --source-project /path/to/gpt6-test \
  --history /path/to/navigation_shared_explore_20261006T161135Z \
  --codex /path/to/codex \
  --output work/vln-shared-new

# 查看任务和输入检查结果，不调用模型。
scripts/run_vln_shared.sh run --config work/vln-shared-new/inputs/campaign.json \
  --output work/vln-preflight --dry-run

# 中断后继续；已评分结果不会重复执行。
scripts/run_vln_shared.sh run --config work/vln-shared-new/inputs/campaign.json \
  --output work/vln-shared-new --resume
```

第一次导入输入时可加 `--reuse-exploration`，校验并复制所选数据集的历史探索快照。此选项只复用训练记忆，所选测试仍重新执行 500 步，完全不读取旧成功率来决定是否重跑；默认全量为 314 条。默认重新探索。使用已有 `--config` 时，是否复用探索由配置中的 `import_snapshot` 决定。迁移到其他机器后，先修改生成配置中的外部输入路径并重新准备哈希；不要修改已经启动的 campaign。

## 选择数据集和小样本检查

准备好本机的 `campaign.json` 后，以下命令对 LHPR、R2R、RxR 各测一个 episode，仅运行 500 步阶段：

```bash
scripts/run_vln_shared.sh run --config /path/to/campaign.json \
  --datasets lhvln r2r rxr --episodes-per-dataset 1 --only-500 \
  --output work/vln-500-check
```

| 参数 | 行为 |
| --- | --- |
| `--datasets lhvln r2r rxr` | 空格分隔，可只选其中一个或多个；`lhvln` 等价于 `lhpr_vln`，输出统一用后者。省略时运行配置中的全部数据集。 |
| `--episodes-per-dataset N` | 每个所选数据集取现有测试列表的前 N 条；N 须为正整数，不得超过任何所选数据集的可用数量。省略时取全部。训练任务不缩减，不随机抽样，不改变原始索引或 seed。 |
| `--only-500` | 仅运行最多 500 个物理动作；原生环境可提前终止。成功和有评分的任务失败均作为最终结果，绝不进入 5000 步阶段。 |
| `--dry-run` | 校验输入并显示所选数据集、任务数、预算及探索来源，不调用模型或启动模拟器。 |
| `--progress-interval 30` | 等待心跳的间隔秒数，须为正的有限数。默认 30 秒；启动、步数变化和结束立即输出。 |

删除 `--only-500` 即恢复“500 → 失败重跑 5000”，仍保留指定的小样本；同时删除 `--datasets` 和 `--episodes-per-dataset` 即使用全量默认流程。每次新实验使用新的 `--output` 目录。

断点续跑时重用同一命令并追加 `--resume`，保留相同的数据集、数量和阶段参数。仅 500 步和两阶段的运行身份不同，不能在同一已启动目录内切换。配置文件保持原样；选择仅作用于本次运行。只准备和校验所选数据集的输入，只探索或导入所选数据集的训练记忆。

默认模型 `gpt-6-astra`、推理强度 `max`、Codex CLI `0.161.0`、并发 4。`--concurrency` 控制测试并发，`--retries` 为每次启动中每个未评分任务的额外技术重试次数（默认 1），`--case-timeout` 默认 7200 秒，`--exploration-timeout` 默认 14400 秒。动作预算不受模型轮数影响；技术重试沿用当前阶段预算。退出码 2 表示仍有未评分任务；有效任务失败不等于运行异常。

CLI 在隔离工作区中安装声明的 `tool-policy.toml`，只预授权本 Benchmark 的模拟动作与记忆工具，保留 `workspace-write` shell 沙箱。工作区必须允许加载项目配置。配置依据：[官方 MCP 配置说明](https://learn.chatgpt.com/docs/extend/mcp?surface=cli)。工具策略被宿主管理策略限制时，结果保持异常，不能当作任务失败。

## 实时终端进度

`run` 默认立即向 **stderr** 输出进度，不需要额外开关或 `python -u`。最终汇总 JSON
仍只写到 **stdout**。输入准备、校验、历史记忆导入或训练探索会先显示当前阶段；
测评时按数据集、原始 episode ID 和尝试编号分别输出进度。以下为格式示例：

```text
[09:00:00] event=stage_start budget=500 cases=3
[09:00:00] dataset=rxr episode=198 attempt=attempt-0001 retry=0/1 event=start steps=?/500 elapsed=0.0s idle=0.0s artifacts=...
[09:00:04] dataset=rxr episode=198 attempt=attempt-0001 retry=0/1 event=step steps=0/500 elapsed=4.0s idle=0.0s
[09:00:12] dataset=rxr episode=198 attempt=attempt-0001 retry=0/1 event=step steps=32/500 elapsed=12.0s idle=0.0s
[09:00:30] dataset=rxr episode=198 attempt=attempt-0001 retry=0/1 event=heartbeat steps=32/500 elapsed=30.0s idle=18.0s
[09:00:40] dataset=rxr episode=198 attempt=attempt-0001 retry=0/1 event=attempt_end steps=32/500 elapsed=40.0s idle=28.0s status=agent_error
[09:00:40] dataset=rxr episode=198 attempt=attempt-0002 retry=1/1 event=retry steps=?/500 elapsed=0.0s idle=0.0s artifacts=...
```

- `steps` 是模拟器最近返回的物理动作数，包含原生自动脱困动作；每批动作返回后更新。
  `?` 表示尚未收到 reset 观测，随后从 0 开始；不是模型轮数或 RPC 批次数。
- `heartbeat` 每隔指定时间报告当前计数；`elapsed` 为该次尝试耗时，`idle` 为距上次
  步数更新的时间。心跳表示启动器仍在等待，不保证模型网络正常，也不会改变超时或评分。
- `retry=1/1` 表示本次启动允许的 1 次额外技术重试已开始，步数重新计算。5000 阶段
  单独显示预算和初始计数，不与技术重试混淆。`attempt-XXXX` 在续跑时继续编号。
- `attempt_end` 的 `status` 区分 `success`、`task_failure` 和技术错误；错误详情在该次
  `artifacts` 目录的 `outcome.json` 与子进程日志中。只有有效原生失败会进入第二阶段。
- `reuse` 表示复用已经评分的结果；`stage_progress processed=N/M` 表示该阶段已处理
  的 episode 数，可能包含技术重试耗尽的未评分任务。末尾 `dataset_end` 报告已评分、
  成功和未完成数量，`run_end status=complete/incomplete` 表示整个测评是否完成评分。

终端显示不依赖 stdout 的缓冲。需要保存日志时，在运行命令末尾追加
`> summary-console.json 2> progress.log`，并在另一个终端执行 `tail -f progress.log`。
这两个文件位于当前 shell 目录；完整结果仍保存在 `--output` 下的 `evaluation/summary.json`。
保持实时终端显示且只保存最终 JSON 时，仅追加 `> summary-console.json`。

本次进度改动也改变代码指纹。升级后使用新的输出目录（例如上面的 `work/vln-500-progress`）；
旧 `work/vln-500-cli-check` 结果保留，不能用新代码直接 `--resume`。

## 行为和工件

- LHPR 114、R2R 100、RxR 100；保留原始标注索引及 seed，校验训练／测试场景分离。R2R 使用历史 `rand100` 的起点与参考轨迹，RxR 使用选定的 guide 任务，不重新随机抽样。
- 两阶段使用相同冻结快照、独立工作区和新 CLI 会话；5000 步不继承 500 步的测试记忆或 simulator 状态。
- 动作批次最多 32 项；在终止或 LHPR 子任务边界提前停止。自动脱困由原生运行时负责，计入物理动作预算。RPC 步号与物理步数分别保存。
- 全部 500 步任务完成或技术重试耗尽后，才建立第二阶段队列。第二阶段任务数动态产生，绝不写死历史 128 条。
- `evaluation/summary.json` 的 `budgets` 标明 `[500]` 或 `[500, 5000]`，保存各阶段、整次选中的结果、原生指标、动作数、耗时和可取得的 token 统计。仅 500 步时 `stage_5000` 为 `null`，`stage-5000.json` 标明 `enabled: false` 且队列为空。未知 token 为 `null`；不把未知值填零。分母为本次所选 episode 数，存在未评分任务时完整成功率为 `null`，同时报告已评分成功率和未完成数。
- 探索统计单独列出；历史快照不冒充本次探索成本。技术重试记录独立保存。每次尝试含请求、结果、stdout/stderr、公共 Runner trace、CLI runtime/stdin/events/result、HTTP 图像和原生评分。
- canonical 快照和每次结果均核验哈希；锁防止同一输出目录并发启动。冻结时去除可再生缓存及空目录，方便独立 wheel 导出，原文件保持不变。SIGINT/SIGTERM 关闭本次子进程；续跑可恢复已经由 `bench` 提交的结果。

本地桥接模式校验原生源码和标注输入。交付时还须通过资产 manifest 对完整场景目录校验。当前保存的是观测工件，没有新增逐物理动作视频；历史 LHPR 视频仍按采样观测视频解释。

## 测试、打包及提交

在 RoboRSI 服务环境中运行离线检查。原生检查还需要有使用权限的场景数据、固定版本的原生源码和独立模拟器环境。这些检查比较 HTTP 服务与原生适配器执行相同固定动作时的结果，不调用模型，也不产生 benchmark 精度成绩。

```bash
pytest tests/unit_tests/robots/test_vln_shared.py \
  tests/unit_tests/robots/test_vln_cli.py \
  tests/unit_tests/robots/test_vln_progress.py \
  tests/unit_tests/robots/test_vln_agent_entries.py \
  tests/unit_tests/robots/test_vln_assets.py \
  tests/unit_tests/robots/test_vln_runtime.py \
  tests/unit_tests/roborsi/benchmarks/test_vln_registration.py -q
ROBORSI_VLN_CONFIG=work/vln-shared-new/inputs/campaign.json \
  pytest tests/integration_tests/robots/test_vln_shared_native.py -v

# 无模型的真实环境检查：LHPR 的 STOP 会依次提交各子任务。
scripts/run_vln_shared.sh smoke --config work/vln-shared-new/inputs/campaign.json \
  --dataset r2r --output work/vln-smoke

# 每个数据集导出含冻结探索记忆的独立 Agent；输出在该预设源码目录外。
python -m roborsi.agents.package --config work/vln-shared-new/agent-input/r2r/agent.yaml \
  --name vln_shared_r2r --output /tmp/vln-shared-r2r-package
python -m build --wheel /tmp/vln-shared-r2r-package
```

交付配方与资产命令见 [VLNCE](../benchmarks/vlnce/README.md)、[LHPR](../benchmarks/lhpr_vln/README.md)。两个服务均已在此公共仓库中登记，无需私有仓库的前置 PR。发布时应保存镜像验收证据、不可变镜像 digest 和 HF commit。合并后的完整模型成绩由操作者重新测评，不能使用历史成绩代替。

安装导出的 Agent wheel 时使用 `python -m pip install --no-compile <wheel>`。
训练记忆含 Python 辅助脚本，默认编译会在冻结资源内生成额外缓存，触发严格文件清单校验。
配置的可执行文件须为 Codex CLI `0.161.0`；Python SDK 的版本与此可执行文件检查分别记录。
