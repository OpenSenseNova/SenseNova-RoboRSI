# Evaluation usability implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline, with test-first checks.

**Goal:** Implement the six usability improvements approved in this conversation while retaining `evaluate_model.py --benchmark NAME`.

**Architecture:** Separate native simulator preparation from Agent resource preparation. Keep the existing batch engine responsible for scoring, provenance and resume validation; add progress reporting and an evaluation front end for checks, plans and persisted summaries.

**Tech Stack:** Python, YAML, existing HTTP/batch Runner, Docker and OpenPI.

**Spec:** The user's approved six-point proposal in this conversation.

## Global constraints

- Primary model API remains deployed by the user. Download code stays in download helpers.
- Existing HTTP skips simulator startup, but prepares required ICL/tokenizer/VLA resources.
- `--dry-run` performs no network requests, writes or subprocess launches.
- Model YAML owns `vla.endpoint`; benchmark YAML owns `token_env`. CLI/env overrides remain supported.
- Resume uses a new invocation ID and verifies the original locked batch inputs.
- No credentials in logs/config snapshots. Preserve scoring and task/seed semantics.
- Native Windows verification uses fixtures; GPU evaluation remains a user hardware check.

## Tasks

### Task 1: Config and pure plans
Files: configuration.py, vla_client.py, vla.py, evaluate_model.py; tests/test_eval_usability.py.
- [x] RED: nested VLA endpoint, per-benchmark token env, model-free/offline dry-run.
- [x] GREEN: resolve config fields and print one offline plan for all transports.

### Task 2: Resource preparation and early checks
Files: bootstrap.py, new preflight.py, evaluate_model.py; usability/bootstrap tests.
- [x] RED: HTTP prepares ICL without building/starting simulators; failed API check prevents downloads.
- [x] GREEN: add prepare_agents and run cheap image/tool API and remote catalog probes before preparation.

### Task 3: Progress and persisted diagnostics
Files: batch.py, evaluation/cli.py, new reporting.py, bootstrap.py, vla.py.
- [x] RED: batch emits start/finish/reuse events; child output streams before exit and secrets are redacted; owned service logs survive cleanup.
- [x] GREEN: wire optional progress callbacks, streaming runner and saved service logs.

### Task 4: Resume and run artifacts
Files: evaluate_model.py, reporting.py, vla.py/vla_agent.py; usability/batch tests.
- [x] RED: resume maps grouped run IDs, changing inputs are rejected, runtime VLA ports do not alter Agent identity.
- [x] GREEN: forward native resume, bind transient VLA URLs through environment, persist resolved configuration and summary.

### Task 5: Documentation and verification
Files: README EN/ZH, configs/README.md, model-evaluation and libero-vla guides.
- [x] Document remote resource preparation, API probes, offline plan, logs, resume and result summary.
- [x] Run affected tests, appropriate broader checks, Ruff, doc commands and diff checks.
- [x] Review final diff and prepare the commit for direct push to one_cmd_eval (explicit user authorization).

## Review focus

- Missing token env during offline plans must not block planning or expose credentials.
- Different benchmarks may use different HTTP tokens.
- Resume must retain stable Agent identity across managed VLA port changes.
- Interruptions and startup failures must save diagnostics before removing owned containers.
- Existing explicit/custom Agents must retain their policy and prepared resources.

## Ledger

- Ruling: reuse the current one_cmd_eval checkout and execute inline; the user explicitly requested changes and direct push here.
- Pre-flight: Tasks 1/2 share resolved token/profile settings; Tasks 3/4 share run-directory and child progress interfaces. New interfaces will be regression-tested at each boundary.


## Verification record

- Focused script suite: **130 passed**; the added two-HTTP-service token isolation regression subsequently passed (**1 passed**).
- Download/cache suite: **27 passed**, using temporary fake downloads and no real model weights.
- Native batch/progress, HTTP and runtime export/package checks: **43 passed**, **2 deselected**, **3 failed**. The three failures are listed below and reproduce with the unmodified HEAD HTTP module.
- Wider unit-test attempt: **192 passed, 1 skipped, 20 failed**; stopped at the configured failure limit. This is not a passing full repository suite.
- Ruff lint and standard formatter checks pass for all changed Python files. Git whitespace checks pass.
- Validated **49 documented evaluation commands** and local guide links; help and offline plans for all three source modes also work outside the checkout.
- A fresh read-only review found three issues; regression tests reproduced each before fixing: missing native benchmark summary metrics, custom Agent without runtime.llm, and duplicate invocation ID modifying an older directory.
- Live subprocess output, credential redaction, log preservation before cleanup, native resume, changed-model rejection and stable managed VLA identity are covered by fixtures.
- No real GPU evaluation, actual model inference, simulator image download or full Linux signal-lifecycle test was performed here. Those require the user's Linux/GPU run.

### Native HTTP failures reproduced against HEAD

- `test_remote_metadata_checks_stop_waiting_when_cancelled[startup]`: Windows cancellation exceeded the sub-second timing assertion.
- `test_remote_metadata_checks_stop_waiting_when_cancelled[cleanup]`: same existing cancellation limitation.
- `test_remote_registry_and_batch_run_without_local_assets_or_docker`: POSIX `fcntl` unavailable on Windows.

### Wider-suite failure ledger

These are outside the one-command script checks; the run used native Windows and an in-checkout temporary directory, so POSIX/environment export tests are not portable to this setup.

- `test_cli_selects_config_and_records_derived_version[False]`, `[True]`: `os.getuid` unavailable.
- `test_export_keeps_parents_and_baked_overrides`: recursive in-checkout export/path error.
- `test_configured_embodied_agent_uses_model_prompt_skills_and_budget`: `os.geteuid` unavailable.
- `test_bundle_exports_only_declared_inputs_and_runs_from_saved_artifact`: recursive in-checkout export/path error.
- `test_export_preserves_executable_resources`: POSIX executable is not a Win32 executable.
- `test_wheels_install_and_execute_outside_source_checkout`: POSIX executable-bit assertion.
- `test_build_inputs_copy_directories_globs_modes_and_track_changes`: POSIX executable-bit assertion.
- `test_parent_and_candidate_launch_independently_from_pinned_sources`: executable-bit mutation not observed on Windows.
- `test_two_artifact_versions_produce_separate_runner_results`: `fcntl` unavailable.
- `test_offline_bundle_rebuilds_two_independent_environments`: locked environments require POSIX.
- `test_bundle_rejects_missing_transitive_dependencies`: POSIX precondition precedes the expected missing-dependency error.
- `test_setup_process_stops_on_case_cancellation`: `os.killpg` unavailable.
- `test_setup_ignores_pip_configuration`: POSIX environment executable path unavailable.
- `test_cli_selects_saved_source_and_environment`: `os.getuid` unavailable.
- `test_cli_selects_matrix_and_emits_queryable_results`, `test_cli_accepts_an_external_http_agent`: `os.getuid` unavailable.
- `test_interactive_outputs_use_http_and_survive_server_deletion`, `test_complete_test_and_retries_do_not_execute_actions_twice`, `test_replay_conflict_order_and_cross_episode_requests`: native episode execution/publication on the POSIX-only result-store path fails on Windows.

Raw local reports are in ignored `work/eval-ux-*.xml` and `work/eval-baseline-http.xml`.
Delivery target: `one_cmd_eval`; direct push is explicitly authorized by the user.
