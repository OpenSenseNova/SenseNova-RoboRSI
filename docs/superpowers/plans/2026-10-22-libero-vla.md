# Configurable evaluation and LIBERO VLA implementation plan

> **For agentic workers:** implement natively in this session with test-first checks.

**Goal:** Automatically prepare and serve OpenPI pi05_libero when a LIBERO Agent
enables VLA, and accept benchmark, Agent and model configuration paths.

**Architecture:** Preserve native environments and scoring. A lightweight Agent
wrapper translates a deferred VLA action into native action vectors using the
current observation. An isolated Docker service loads OpenPI; the Runner owns
its lifecycle. Configuration paths and test selection are resolved before setup.

**Tech stack:** Python, existing AgentConfig/HTTP Runner, Docker, pinned OpenPI
215abfb217dbac7d5f1273282331b9b1866c0479 and its uv.lock; official pi05_libero weights.

## Constraints and decisions

- Keep `--benchmark NAME` sufficient; new config-path arguments are optional.
- A single benchmark accepts a bare path; multiple benchmarks use `NAME=PATH`.
- Custom benchmark YAML may contain only schema_version, benchmark and evaluation;
  it inherits the canonical environment. Explicit references resolve from that YAML.
- `evaluation.tasks`, `evaluation.seeds` and a Python `evaluation.selector` filter
  existing task/seed cells. A selector cannot introduce cells absent from the catalog.
- The selected model YAML contains the main API connection and a nested `vla`
  section. Agent `parameters.enable_vla: true` activates that model. Existing
  external VLA MCP URLs retain their behavior. Released presets disable VLA.
- All VLA weight/tokenizer downloads live in `scripts/evaluation/downloads.py`;
  the serving entry point validates and loads prepared local assets.
- Benchmark YAML `endpoint` optionally selects an existing image's HTTP service.
  Null retains automatic preparation. CLI URL/registry sources override YAML;
  HTTP mode retains its existing requirement for prepared Agent/ICL resources.
- `--prepare-only` builds/downloads without loading a model on a GPU; `--dry-run`
  creates no files and starts no services.
- VLA only predicts; native actions still pass through the Runner and Benchmark.
- Reuse images/download caches; reuse one VLA service for identical profiles;
  remove only containers owned by the current invocation, including failure paths.
- No heavyweight OpenPI dependencies are installed in the Runner.

## Tasks and checks

- [x] Add failing tests for config paths, inherited environments, CLI precedence,
  static/Python case selection and preservation of per-task seed sets.
- [x] Implement configuration/path resolution and selection; connect the same
  resolved configurations to planning, preparation and command generation.
- [x] Add failing tests for opt-in VLA discovery, disabled/external paths, profile
  validation, cache identity, startup readiness, sharing and failure cleanup.
- [x] Add the pinned OpenPI Docker environment, CPU download command, warmed HTTP
  server and Runner orchestration. Honor explicitly selected Agent configs.
- [x] Add failing tests for raw versus already-rotated camera observations, state
  conversion, bounded finite actions, native passthrough and standard LIBERO's
  single-action contract. Implement the Agent wrapper and factories.
- [x] Document custom config examples, Python selectors, VLA enabling, caches,
  GPU selection and the distinction between model API and VLA API.
- [x] Run targeted and available broader checks, Ruff, CLI plans and documentation
  examples. Full GPU integration remains the user's hardware check.

## Review focus

- Disabled VLA must not trigger builds, network calls, startup or new tools.
- Custom Agent selection must drive downloads and retain generated local bindings.
- A selector choosing different seeds for different tasks must not expand a cross product.
- Enhanced LIBERO images are already policy-oriented; standard images need rotation.
- Failed downloads/startup/interruptions must preserve caches and clean owned services.

## Verification on Windows, 2026-10-22

- 129 configuration, evaluation, bootstrap, RoboCasa Agent and VLA tests passed.
  Includes real HTTP inference with a fixture policy, real standard Agent factory
  creation/reset, and mocked external Docker/model-download boundaries.
- All 27 download-cache tests passed in an external temporary directory.
- 66 LIBERO native-adapter and runtime-package tests passed (65 initially, one
  export HTTP test repeated with a Windows extended path after a path-length failure).
- Ruff and formatting passed for all 16 affected Python files. All 41 documented
  evaluation commands and local guide links validated without network/model calls.
- Broader Agent configuration/composition/package tests: 32 passed, 3 failed on
  Windows-only platform limitations in existing code/tests:
  - `tests/unit_tests/roborsi/agents/test_config.py::test_configured_embodied_agent_uses_model_prompt_skills_and_budget`:
    existing `roborsi/llm/retry.py` calls POSIX `os.geteuid`.
  - `tests/unit_tests/roborsi/agents/test_package.py::test_export_preserves_executable_resources`:
    Windows cannot directly execute the exported POSIX shell script (WinError 193).
  - `tests/unit_tests/roborsi/agents/test_package.py::test_wheels_install_and_execute_outside_source_checkout`:
    the Windows filesystem does not preserve the expected Unix executable bits.
- Download tests `test_cache_manifest_cannot_be_rewritten_along_with_its_marker`
  and `test_rechecks_cache_and_preserves_modified_files[same_size]` each encountered
  intermittent WinError 5 renaming newly written directories inside the checkout.
  Both passed with all 27 download tests outside the checkout; downloader code
  was unchanged by this feature.
- Actual OpenPI image build, multi-GB weight download, GPU warmup/inference and
  full native benchmark evaluation were not run on this Windows workstation.

## Unified-model and configured-endpoint follow-up verification

- Moved VLA settings into `configs/models/default.yaml`; removed the standalone
  VLA profile and Agent resource references. Custom `--model-config` paths select
  main and VLA models together; resolved runtime bindings travel with the Agent.
- Centralized VLA download execution/cleanup in `downloads.py`, copied it into
  the OpenPI image and included it in image cache identity. Serving checks cached
  weights/tokenizer before importing the policy implementation.
- Added optional `endpoint` to benchmark YAML, with CLI overrides and the same
  external HTTP catalog, authentication and concurrency constraints.
- 145 configuration/evaluation/bootstrap/VLA/RoboCasa tests passed, including
  independent VLA selections, CLI model-path planning, download failure cleanup,
  prepared-cache resolution and configured HTTP source precedence.
- 27 downloader-cache and 66 LIBERO export/native/runtime tests passed: 238 total.
- Ruff and formatting passed for all 13 changed Python files; 43 documented
  evaluation commands and local guide links validated. GPU integration remains
  untested locally and is reserved for the user's hardware run.
