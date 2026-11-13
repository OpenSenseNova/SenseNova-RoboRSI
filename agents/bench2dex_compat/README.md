# Historical CLI Agent

`robots/bench2dex/agent.yaml` selects the shared `codex_cli` backend in external
file-protocol mode. Its factory and prompt live in `robots/bench2dex/agents/baseline`;
`configs/agents/bench2dex/compat.yaml` is a compatibility alias. The original
reference ZIP is immutable and hash checked; its separate license notice applies.
Only its CLI startup and cleanup methods are adapted to the shared backend.
`prompt_file`/inline prompt and ordered `skills` change the actual initial input.
These are distinct experiments; the empty-Skill default matches the historical
prompt exactly. Model calls go through `EmbodiedAgent.start_episode()` using
the external-session API from framework PR #79. MCP tools are not introduced.

Use `benchmarks.bench2dex.agent_bundle` to attach a downloaded, verified task ICL
ZIP, then use `roborsi.agents.package` to create a relocatable wheel. The source
release excludes ICL payloads. Task-specific wheels containing ICL should be
hosted with the ICL assets rather than committed to the source repository.

The pinned executable and companion code-mode host come from
`openai-codex-cli-bin==0.154.0`. Authenticate locally and set
`BENCH2DEX_CODEX_AUTH` to that authentication file if it is outside the default
local Codex directory. Only that single file is copied; user rules, configuration,
sessions, plugins and source repositories are not mounted into the policy.
Credentials are never bundled. Private evaluation output needs separate review
before publication.

The policy namespace uses Ubuntu 22.04 system Python with NumPy 1.21.5, SciPy 1.8.0,
Pillow 9.0.1 and PyYAML 5.4.1. Install distribution packages `bubblewrap`,
`python3-numpy`, `python3-scipy`, `python3-pil`, `python3-yaml` and `ca-certificates`.
Run `python -m benchmarks.bench2dex.check_runtime` before evaluation. The
`agent.lock.txt` file instead pins the Python 3.11 Runner/Agent service packages.
