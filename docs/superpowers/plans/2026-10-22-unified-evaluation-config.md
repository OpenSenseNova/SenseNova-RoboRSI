# Unified evaluation configuration

Implement the user-approved configuration structure on `one_cmd_eval`, retaining
the `evaluate_model.py --benchmark NAME` entry point. Commit and push after checks.

## Configuration contract

- `configs/benchmarks/NAME.yaml` selects the environment, model profile and Agent.
  Environment settings include pinned images/build arguments, downloads and catalog defaults.
- `configs/models/default.yaml` supplies shared model settings. Credentials are
  referenced by environment variable name and never persisted in generated files.
- LIBERO Agent YAML and context maps move to `configs/agents/libero/`; all inherited
  paths, packaging helpers, combined-image inputs and documentation move with them.
- Keep `configs/evaluations/embodied.yaml` as the existing Agent comparison matrix.
- CLI model values override environment variables, which override model YAML.
  Explicit custom Agent format remains unchanged unless CLI/model YAML selects a format.
- Catalog environment variables override benchmark YAML only for managed services.
- Generated local resource bindings remain under `<work-dir>/presets/`.

## Tasks

1. Add failing configuration tests: all five benchmark references compose; relative
   paths resolve from declaring files; invalid/unknown fields and inline credentials
   fail before downloads; CLI/environment/YAML precedence and per-benchmark models work.
2. Migrate YAML/maps and their references; add model/benchmark YAML and a validated
   loader in `scripts/evaluation/configuration.py`.
3. Wire evaluation and bootstrap to loaded settings, including image build arguments,
   catalog defaults, resource downloads and model credentials scoped to each child.
   Preserve native planner budgets, sealed ICL verification and container cleanup.
4. Update root/config/benchmark documentation in both languages. Verify documented
   commands, targeted evaluation/config/Agent/export tests, Ruff and Git whitespace;
   inspect the final diff, commit and push `one_cmd_eval`.

## Review focus

- Moved presets must still export portable bundles with the same case maps/policy.
- Invalid YAML and missing references must fail before Docker/network mutations.
- Different selected model profiles must not reuse another benchmark's API key.
- Image cache identity must include configurable Docker build arguments.
- Existing HTTP/registry catalogs must not be changed by automatic catalog defaults.

Full GPU simulation remains the user's integration check; local checks use fixture
catalogs, mocked Docker operations and the existing verified ICL data.
