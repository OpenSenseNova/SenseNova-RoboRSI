# Shared navigation baseline

VLNCE and LHPR-VLN share this policy through YAML inheritance:

| Benchmark | Standard entry | Supported datasets |
| --- | --- | --- |
| VLNCE | `robots/vlnce/agent.yaml` | R2R-CE, RxR-CE |
| LHPR-VLN | `robots/lhpr_vln/agent.yaml` | LHPR-VLN |

Each entry extends its own `agents/baseline/agent.yaml`, which extends this
directory's `agent.yaml`. The old `agents/vln_shared/agent.yaml` remains a thin
compatibility alias. Python dependencies and their installation instructions
remain in [agents/vln_shared](../../../../agents/vln_shared/README.md).

The factory uses `config.embodied_agent(...)`: Codex CLI `0.161.0`, model
`gpt-6-astra`, reasoning `max`, MCP tools and unlimited model turns. It creates a
private workspace and session for each case. Native action budgets belong to
the Benchmark. Only the exploration session receives training tools.

`factory.py`, `prompt.md`, `explore.md` and `tool-policy.toml` are declared
inputs. Exploration reads its prompt from `resources.exploration_prompt`.
`resources.tool_policy` supplies the per-workspace MCP policy;
`resources.exploration`, when supplied by the campaign, is a frozen directory
containing `snapshot.json` and `workspace/`. The factory verifies the recorded
file hashes before copying training memory into a fresh writable workspace.
There are no custom `parameters` or Native Context settings.

The standard entries are usable with `bench eval --agent-config` and the public
Agent exporter. Their `source_root` is the repository root so inherited entries
can share resources; export them to a directory outside the checkout. Only
declared files enter the wheel. On their own they start without frozen
exploration memory.
For the shared-exploration experiment, use `scripts/run_vln_shared.sh`: it
composes the matching Benchmark entry, copies declared inputs into each dataset
preset, then attaches the training snapshot. Export the generated preset to
include that snapshot in its wheel. Credentials remain external to all inputs.

See the [English run guide](../../../../docs/vln_shared_evaluation.md) and
[Chinese run guide](../../../../docs/vln_shared_evaluation.zh-CN.md) for commands,
version/resume behavior and delivery limitations.

Install exported wheels with `python -m pip install --no-compile <wheel>`. Frozen
training memory can contain Python helpers; pip's bytecode compilation would
add files under the immutable resource and make its strict inventory check fail.
The package and its frozen source files must remain unchanged after installation.
