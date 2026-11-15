# Historical DexJoCo Agent

The canonical preset is `robots/dexjoco/agent.yaml`;
`configs/agents/dexjoco/compat.yaml` is a thin alias. Export it through the existing
RoboRSI Agent packager. All imported local Agent modules and the reference ZIP
are declared; the simulator is not included in the Agent wheel.

`robots/dexjoco/agents/baseline/resources/reference-code.zip` is the byte-identical code-only release from 2026-10-12,
SHA-256 `e883be2bf8f22674b24e56758721f754cce6353a45044f32025e26be7e870674`.
It contains inspectable Python source, prompts, configuration, provenance and
the original MIT/third-party notices. It contains no knowledge assets, model
credentials, raw transcripts or test trajectories. `robots/dexjoco/reference.py`
verifies its hash before extracting to a private temporary source directory.

The trusted bridge uses the archive's original validators, sandbox construction,
action file helper, model identity checks and token accounting. Session execution
uses `FrozenPolicyAgent → EmbodiedAgent → CodexCliPlanner`. The shared
`EmbodiedEpisode` owns the persistent worker, cancellation and close; the
DexJoCo adapter owns file feedback and historical phase/chunk behavior. This
requires [framework PR #79](https://github.com/SenseTime-FVG/RoboRSI/pull/79).
Terminal backend and cleanup failures propagate to the Runner; an end-marker
write failure still cleans up and retains the session for retry.
The policy sandbox uses a separate numerical venv; it cannot read the Agent
wheel, extracted evaluator, Runner files or simulator checkout.

The factory consumes `prompt_file` (or `agent.system_prompt`), phase prompt
resources and full-text `skills` for both sessions. It rejects unsupported model,
CLI isolation, runtime, timeout and working-directory overrides explicitly. Runtime paths and the
operator's authentication file come from explicit `ROBORSI_DEXJOCO_*`
environment variables described in the Benchmark README. The Runner supplies
`ROBORSI_MEDIA_DIR` for transferred camera artifacts.

Dependency regeneration (repository root):

```bash
uv pip compile pyproject.toml agents/dexjoco_compat/requirements.in \
  --extra runtime --python-version 3.10 --generate-hashes \
  -o agents/dexjoco_compat/requirements.txt
```

This lock installs the framework's dependencies, including mandatory SDKs. It
does not install the separately built RoboRSI wheel or this Agent wheel.
The actual policy executable must report `codex-cli 0.154.0`; the framework's
installed SDK versions are not a replacement for that executable.

See [English](../../benchmarks/dexjoco/README.md) and
[中文](../../benchmarks/dexjoco/README.zh-CN.md) for setup, wheel installation,
policy boundaries, supported backend, model-run commands and delivery gaps.
