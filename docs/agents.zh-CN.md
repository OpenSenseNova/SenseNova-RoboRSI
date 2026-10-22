# Agent 配置

一键评测通过 [`configs/`](../configs/README.md) 统一选择环境、模型和 Agent。
LIBERO 专用 YAML 与 context map 位于 `configs/agents/libero/`，RoboCasa ICL 预设为
`configs/agents/robocasa365/icl.yaml`。执行前会组合模型 YAML 和 benchmark 的
`agent_overrides`。下方介绍的直接 `bench eval` 配置方式继续可用。

[English](agents.md) · [全部指南](README.md)

Agent 接收任务、当前观测和动作 schema，并返回动作。Benchmark 负责环境推进、
停止和评分。同一份通用 Agent 配置可用于各个受支持的 benchmark 服务。

## 选择配置

| `configs/agents/embodied/` 中的文件 | Planner | 配置内容 |
| --- | --- | --- |
| `default.yaml` | 模型 API | 视觉模型，保留最近八轮反馈 |
| `short_context.yaml` | 模型 API | 继承默认配置，保留四轮反馈 |
| `codex.yaml` | Codex SDK | 模型及 Agent 环境中的 SDK 鉴权 |
| `codex_cli.yaml` | Codex CLI | 可执行文件及精确的 CLI 版本 |
| `claude_code.yaml` | Claude SDK | 模型及 Agent 环境中的 SDK 鉴权 |

运行前替换模型占位符。API 凭据通过 `OPENAI_API_KEY` 提供，自定义兼容服务可使用
`OPENAI_BASE_URL`。SDK 配置使用 Agent 环境中对应提供方的鉴权。Codex CLI 需要另外
安装，且 `--version` 输出必须与配置完全一致。离线测试使用预设响应，不验证真实登录。

## 控制行为

`agent.system_prompt` 描述任务策略。`agent.max_turns` 限制模型轮数，
`agent.max_tokens` 限制单次模型输出，`agent.planner_timeout_s` 限制整段对话，
`agent.action_timeout_s` 限制动作反馈等待时间。Runner 的 `--timeout-s` 和
`--case-timeout-s` 分别限制通信与整个 case。原生物理步数预算由 benchmark manifest 定义。

`runtime.context.strategy` 和 `keep_turns` 控制保留的对话。
`runtime.trace.mode: full` 记录完整模型 trace，供检查调用过程。
Trace 可能包含任务内容和模型响应，请与实验输出一起管理。

可直接覆盖配置项，无需复制 preset：

```bash
bench eval --registry-root /absolute/path/to/registry \
  --results-root /absolute/path/to/results --run-id configured-1 \
  --benchmark libero --task libero:0 --seed 0 \
  --agent-config configs/agents/embodied/default.yaml \
  --agent-set 'runtime.llm.model="YOUR_VISION_TOOL_MODEL"' \
  --agent-set 'agent.max_turns=50' --agent-set 'runtime.context.keep_turns=4'
```

覆盖值使用 JSON：字符串需要在参数内再加引号，其余使用 JSON 数字、`true`、`false`
或 `null`。保存的配置指纹包含展开后的配置及其声明的资源。

## 比较上下文设置

[`configs/evaluations/embodied.yaml`](../configs/evaluations/embodied.yaml)
选择默认与短上下文两个配置。在 `default.yaml` 中设置共享模型，再用
`--agent-matrix configs/evaluations/embodied.yaml` 替换 `--agent-config`。
添加 `--dry-run` 可先检查展开的执行计划。各个 Agent 分别保存 case 和结果。

## Python 工厂与可安装包

[`hold_agent.py`](../examples/roborsi/hold_agent.py) 演示 `reset`、`act` 和 `close`。
它的 `make_agent(case_dir)` 工厂运行于独立 Python 进程。
[`model_agent.py`](../examples/roborsi/model_agent.py) 演示用户自定义上下文回调，
以输入 token、图像数量和反馈轮数作为软预算。

自定义 Python 和资源文件通过配置的 `factory`、`code`、`resources` 声明。
导出时只打包这些声明的输入：

```bash
python -m roborsi.agents.package \
  --config configs/agents/embodied/default.yaml \
  --name my_agent --output ../my-agent-package
```

输出目录必须位于 `source_root` 外。在 Agent 环境中构建并安装生成的包后，使用
`--agent-config bundle:my_agent`。源码导出不会安装模型服务或模拟器依赖。
版本验证见 [评测指南](evaluation.zh-CN.md)。
