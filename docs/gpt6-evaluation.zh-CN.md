# GPT-6 Astra 全 benchmark 评测

[English](gpt6-evaluation.md) · [全部指南](README.md)

[`evaluate_gpt6.py`](../scripts/evaluation/evaluate_gpt6.py) 按顺序调用各 benchmark
现有的 Agent 和评测入口，默认覆盖 **11 个已发布 benchmark**，其中 VLN-CE 的 R2R、RxR
分开运行。尚未发布的 RoboDojo、SIMPLE 不包含在内。

脚本负责编排，不改变原生动作、终止规则、评分、提示词和冻结 ICL 契约。
提供统一入口不代表已经复现所有论文结果。

## 一次性安装与准备

使用 Linux/WSL、Python 3.10–3.12、Docker，以及各 benchmark 要求的 GPU 和运行环境。
Bench2Dex 的固定依赖需要 Python 3.11 Runner；按对应指南隔离模拟器和 Agent 依赖。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[runtime]'
```

编辑 [`configs/evaluations/gpt6.yaml`](../configs/evaluations/gpt6.yaml)：

- **LIBERO 系列、RoboCasa365：** 复用原有自动准备流程，下载/构建镜像、资产及可用 ICL；
  仍需满足原有 GHCR/HF 访问权限和 GPU 要求。
- **其余 benchmark：** 先按下表的指南准备资产、registry/HTTP 服务和 Agent，
  再填写 `registry_root` 或 `endpoint`、`agent_config`。registry 模式由现有 Runner
  启动环境；HTTP 模式需要服务已经运行。RoboLab 必须按原指南使用 8 GiB `/tmp`，
  不使用通用 Docker executor 的 1 GiB 默认值。
- **CLI 环境：** 各项的 `codex` 指向对应固定版本的可执行文件；依赖不同时，通过
  `python` 指定对应 Runner/Agent Python。`null` 沿用原来的环境解析方式。
  四个固定版本不能依赖同一个全局 Codex 安装。

YAML 中的路径相对该 YAML 文件解析，与当前工作目录无关。
`bundle:NAME` 可选择已经安装的 Agent wheel。密钥保留在环境变量或认证文件中。

| 配置项 | 默认范围和准备入口 |
| --- | --- |
| `libero` | [标准 LIBERO](../benchmarks/libero/README.md)，默认 `libero_spatial`，可用 `ROBORSI_LIBERO_SUITE` 选择其他套件。 |
| `libero_pro` | [PRO](../benchmarks/libero_pro/README.md)，已发布 `migration_subset`，80 个 seed-0 case。 |
| `libero_long_plus_plus` | [Long++](../benchmarks/libero_lilo/README.md)，已发布子集，12 个 seed-0 case。 |
| `libero_ultra_long` | [Ultra-Long](../benchmarks/libero_lilo/README.md)，已发布子集，9 个 seed-0 case。 |
| `robocasa365` | [Target50](model-evaluation.zh-CN.md)，默认 340 个 cell，包含没有发布 memory 的任务。 |
| `robolab` | [标准 RoboLab](../benchmarks/robolab/README.zh-CN.md)，准备 120 任务、seed 0 的 catalog 和 HTTP 服务；不混入历史 120 × 10 app-server 流程。 |
| `bench2dex` | [冻结 ICL](../benchmarks/bench2dex/ICL.zh-CN.md)、[运行环境](../agents/bench2dex_compat/README.md)和 HTTP 服务，使用原有 26 任务 × 3 sample 的 [78-case 计划](../benchmarks/bench2dex/plan-78.json)，逐任务绑定对应 Agent/ICL ZIP。 |
| `dexjoco` | [registry](../benchmarks/dexjoco/README.zh-CN.md) 和 [Agent 资产](../benchmarks/dexjoco/ICL.zh-CN.md)，11 任务 × 5 seed，保留原有 1x→10x 的新会话续跑。 |
| `behavior_1k` | [授权资产与 registry](../benchmarks/behavior_1k/README.zh-CN.md)，然后准备对应任务 Agent。默认已发布 ICL 任务 `sorting_vegetables`、seed 0；其他任务需要各自的学习包与 Agent。 |
| `lhpr_vln` | [LHPR-VLN](../benchmarks/lhpr_vln/README.zh-CN.md)，独立的 114 任务、seed 42 registry。 |
| `vlnce_r2r` | [VLN-CE](../benchmarks/vlnce/README.zh-CN.md)，R2R 100 任务、seed 0 registry。 |
| `vlnce_rxr` | [VLN-CE](../benchmarks/vlnce/README.zh-CN.md)，**另一套独立的** RxR 100 任务、seed 0 registry。 |

“全 benchmark”指上述已发布配置，不代表全部上游任务或扩展后的完整官方测试集。
三项导航配置默认使用源码预设，**不附带 ICL**。如需冻结 memory，按
[发布产物说明](vln_shared_evaluation.zh-CN.md#已提交的交付物)安装对应 wheel，分别将
`agent_config` 改为 `bundle:vln_shared_lhpr_vln`、`bundle:vln_shared_r2r`、
`bundle:vln_shared_rxr`，不能在 R2R、RxR 之间混用。

## 模型认证与 CLI 版本

API Agent 及 RoboLab、BEHAVIOR、导航的 CLI Agent 需要有权限使用模型的
`OPENAI_API_KEY`。API 任务使用独立的
[`configs/models/gpt6.yaml`](../configs/models/gpt6.yaml)，模型固定为 `gpt-6-astra`，
不会读取自部署模型的 `MODEL_NAME`、`OPENAI_BASE_URL`。
可用 `GPT6_BASE_URL` 覆盖 API 任务地址；CLI 任务保留各自预设的 provider/认证配置，
该变量不会改变 CLI provider。

| Agent | 固定 Codex CLI 版本 |
| --- | --- |
| Bench2Dex、DexJoCo | 0.154.0，及其隔离数值环境与配套二进制 |
| RoboLab | 0.160.0 |
| LHPR-VLN、VLN-CE | 0.161.0 |
| BEHAVIOR-1K | 0.162.0 |

按对应指南安装并在 YAML 中指定独立可执行文件。Bench2Dex 仍需安装
`openai-codex-cli-bin`，并通过原有版本、哈希和数值沙箱校验。

Bench2Dex、DexJoCo 还需要原有认证及 policy 环境，填写实际绝对路径，例如：

```bash
export OPENAI_API_KEY='YOUR_API_KEY'
export BENCH2DEX_CODEX_AUTH='/absolute/path/to/bench2dex/auth.json'
export ROBORSI_DEXJOCO_AUTH='/absolute/path/to/dexjoco/auth.json'
export ROBORSI_DEXJOCO_POLICY_PYTHON='/absolute/path/to/separate-policy-venv'
export ROBORSI_DEXJOCO_AGENT_ASSETS='/absolute/path/to/dexjoco-agent-assets'
```

`codex` 为 null 时，原有 `BENCH2DEX_CODEX_BIN`、`ROBORSI_DEXJOCO_CODEX` 设置仍然有效。
原生资产授权、私有 HF 访问需提前完成，脚本不会获取资产许可或生成替代训练知识。

## 启动

先离线查看完整计划，无需密钥，不下载、不创建文件、不启动子进程或调用模型：

```bash
python scripts/evaluation/evaluate_gpt6.py --dry-run
```

完成一次性环境准备与配置后，一条命令启动全部已发布配置：

```bash
python scripts/evaluation/evaluate_gpt6.py
```

正式评测前，所有选中项必须先通过本地前置检查及原有的不调用模型的预检。
环境或 Agent 缺失时，整批评测不会开始。正式执行可能下载大镜像并产生模型调用费用。
各项顺序运行，每项并发数为 1。

也可选择少量 benchmark 检查环境，或只运行尚未执行的项目：

```bash
python scripts/evaluation/evaluate_gpt6.py --benchmark libero_pro --dry-run
python scripts/evaluation/evaluate_gpt6.py \
  --benchmark vlnce_r2r --benchmark vlnce_rxr --run-id gpt6-navigation-001
python scripts/evaluation/evaluate_gpt6.py \
  --config /absolute/path/to/gpt6.yaml --work-dir /data/gpt6 --run-id gpt6-all-001
```

每次使用新的 `--run-id`。脚本不覆盖旧运行，也不自动重试已有评分的失败任务。
任务、seed、超时在对应配置项中设置；Bench2Dex 使用固定 `plan`。
`--work-dir` 只移动自动准备的缓存和结果，不改变 YAML 中已有 registry、Agent、ICL 的路径。

## 结果与验证范围

默认总览为 `work/gpt6/results/<run-id>/summary.json`，旁边保存实际命令计划和各项的
预检、执行日志。逐 case 结果位于 `work/gpt6/native-results/`；Bench2Dex 继续在该次
运行的 `bench2dex/` 下保存原始计划、逐任务结果和严格完整性报告。

各 benchmark 保留独立的原生指标，不计算跨 benchmark 的混合平均分。
有有效评分的 `task_failure` 属于正常完成；Agent、基础设施错误或不完整报告会让整批返回非零。
某项执行失败后继续其他项。返回码：0 为全部正常完成，1 为执行失败/不完整，
2 为配置或前置检查失败，130 为中断。日志会脱敏环境中的 API/服务 token，
Ctrl-C 通过现有清理流程停止当前子进程。

离线测试覆盖命令组合、dry-run 隔离、前置检查阻断、失败报告及结果分离，
不证明真实模拟器成功、付费模型验收或论文分数复现。
