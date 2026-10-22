# Agents

For one-command evaluation, environments, models and Agents are selected through
[`configs/`](../configs/README.md). Specialized LIBERO YAML and context maps live
under `configs/agents/libero/`; RoboCasa's ICL preset is
`configs/agents/robocasa365/icl.yaml`. Model YAML and benchmark `agent_overrides`
are composed over the selected Agent before evaluation. The direct `bench eval`
configuration described below remains available.

[简体中文](agents.zh-CN.md) · [All guides](README.md)

An Agent receives the task, current observation and action schema. It returns
actions; the benchmark owns environment stepping, stopping and scoring. The
same Agent preset can target each supported benchmark service.

## Select a preset

| File under `configs/agents/embodied/` | Planner | Configuration |
| --- | --- | --- |
| `default.yaml` | Model API | Vision model, recent eight feedback turns |
| `short_context.yaml` | Model API | Inherits the default, keeps four turns |
| `codex.yaml` | Codex SDK | Model and existing SDK authentication |
| `codex_cli.yaml` | Codex CLI | Executable and exact installed CLI version |
| `claude_code.yaml` | Claude SDK | Model and existing SDK authentication |

Set the model placeholder before use. API credentials come from
`OPENAI_API_KEY`; custom API endpoints can use `OPENAI_BASE_URL`. SDK presets
use their provider's authentication in the Agent environment. The Codex CLI
preset requires a separately installed CLI and an exact `--version` match.
Offline tests use scripted provider responses and do not verify live SDK login.

## Control behavior

`agent.system_prompt` describes the task policy. `agent.max_turns` limits model
turns; `agent.max_tokens` limits a model output; `agent.planner_timeout_s` limits
the conversation and `agent.action_timeout_s` limits action feedback waits.
Runner `--timeout-s` and `--case-timeout-s` set separate transport and case
deadlines. Native physical step limits belong to the benchmark manifest.

`runtime.context.strategy` and `keep_turns` control retained dialogue.
`runtime.trace.mode: full` records the model trace for inspection. Trace files
may contain task content and model responses; store them with the run outputs.

Override individual values without copying a preset:

```bash
bench eval --registry-root /absolute/path/to/registry \
  --results-root /absolute/path/to/results --run-id configured-1 \
  --benchmark libero --task libero:0 --seed 0 \
  --agent-config configs/agents/embodied/default.yaml \
  --agent-set 'runtime.llm.model="YOUR_VISION_TOOL_MODEL"' \
  --agent-set 'agent.max_turns=50' --agent-set 'runtime.context.keep_turns=4'
```

Override values use JSON: quote strings inside the argument, and use JSON
numbers, `true`, `false` or `null` for other values. The saved config fingerprint
includes the resolved preset and declared resources.

## Compare context choices

[`configs/evaluations/embodied.yaml`](../configs/evaluations/embodied.yaml)
selects the default and short-context presets. Set the shared model in
`default.yaml`, then use `--agent-matrix configs/evaluations/embodied.yaml`
instead of `--agent-config`. Add `--dry-run` to inspect the expanded plan before
starting services. Each Agent has separate cases and results.

## Python factories and portable packages

[`examples/roborsi/hold_agent.py`](../examples/roborsi/hold_agent.py) demonstrates
`reset`, `act` and `close`. Its `make_agent(case_dir)` factory runs in a separate
Python process. [`model_agent.py`](../examples/roborsi/model_agent.py) demonstrates
a user-owned context callback with soft input-token, image and feedback budgets.

Declare `factory`, `code` and `resources` in a preset when your Agent has custom
Python or resource files. Export only those declared inputs into a portable
package:

```bash
python -m roborsi.agents.package \
  --config configs/agents/embodied/default.yaml \
  --name my_agent --output ../my-agent-package
```

The output directory must be outside `source_root`. Build and install the
generated package in the Agent environment, then select `--agent-config
bundle:my_agent`. Source export does not install model services or simulator
dependencies. See [Evaluation](evaluation.md) for version verification.
