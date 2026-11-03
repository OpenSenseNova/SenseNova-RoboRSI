# LHPR-VLN

[简体中文](README.zh-CN.md) · [All guides](../../docs/README.md) · [Build reference](BUILD.md)

This integration exposes LHPR-VLN through the shared Benchmark service,
Registry and Runner, with isolated native workers and native scoring.
Initial code and usage guides are available in this branch.
Pinned artifacts and installation requirements are documented below and in the
[shared evaluation guide](../../docs/vln_shared_evaluation.md).

## Tasks, actions and scores

The evaluation catalog contains **114 filtered test episodes**, split
`unseen_test_filtered_0708`; the training catalog has 411 episodes. Keep original
annotation order: Registry seed is **42**, while native initialization uses
`42 + original_row_index`. Native source is the AgentCanvas `env_lhvln`
runtime at `6c0a17d3830be5fa9c64e8dec35be29808581f2d`.

Actions are `{"actions": ["move_forward", "turn_left", "stop"]}` with 1–32
entries chosen from `stop`, `move_forward`, `turn_left`, `turn_right`.
Forward is 0.25 m and turns are 30 degrees. STOP submits the current subtask;
a batch ends at a subtask boundary or episode termination. Native recovery
after three blocked actions consumes the same physical-action budget.
Observations expose the instruction, public progress and front/left/right RGB
views (512 rendering, contrast 1.5, resize 366).

Success is native `SR`, with strict 1 m fixed-object and same-region-name rules;
RoboRSI score is `float(success)`. The full native metric record is retained.
Unfinished episodes are unscored, not automatically counted as failures.

The standard Registry below uses 500 physical actions. The separate shared
campaign retries only scored failures with a fresh 5000-action budget; the same
task, original index, seed and frozen memory are used in both stages.
See the [protocol and output guide](../../docs/vln_shared_evaluation.md#protocol-and-outputs).

## Environment and image

Use Linux x86_64, an NVIDIA GPU/driver, Docker with NVIDIA Container Toolkit,
and Python 3.11 for the Agent/evaluator. Validation used an RTX 4090.
The image contains Python 3.9.23 / Habitat-Sim 0.3.0 / NumPy 1.23.5; the HTTP service uses a separate Python 3.11.
Do not install the legacy simulator into the Agent environment.
Reserve space for images, licensed scenes and any staged copies.

From a checkout containing the accepted navigation integration and framework
fix #36, install the pinned host dependencies:

```bash
python3.11 -m venv .venv-vln
source .venv-vln/bin/activate
python -m pip install -r agents/vln_shared/requirements.txt
python -m pip install --no-deps -e .
python -m pip check
```

Before #36 is merged, use the [isolated source installation](../../docs/vln_shared_evaluation.md#published-artifacts)
or the validated framework wheel in the [contributor artifacts](https://github.com/panda7777777/SenseNova-RoboRSI/releases/tag/vln-delivery-20261102).
Install and verify the complete Codex CLI 0.161.0 distribution using the
[pinned CLI instructions](../../docs/vln_shared_evaluation.md#install-the-pinned-cli).
Authenticate that executable in the Agent's account; credentials remain external.

Choose an absolute delivery path without symlink components. If needed, run
`docker login ghcr.io -u YOUR_GITHUB_USERNAME` with authorized pull access.

```bash
export VLN_DELIVERY=/absolute/path/outside/checkout/vln-delivery
export VLN_IMAGE=ghcr.io/opensensenova/sensenova-roborsi-lhvln@sha256:7d72dd910d5c9048ca6ec40ad158214497c3a9ac276641df1429314f2d7a3695
mkdir -p "$VLN_DELIVERY"
docker pull "$VLN_IMAGE"
```

Tag: `v1.0.0-cu120`; evaluation uses the immutable digest above.
Image source commit: `1435a63f853649c9e04fe135b664b9835d10fde7`.
[Dockerfile](Dockerfile), [native recipe](Dockerfile.native),
[release inputs](release-inputs.json), [native inventory](native-environment.json),
[service lock](service-requirements.txt) and [rebuild commands](BUILD.md)
describe the fixed build inputs. Pulling this image does not require rebuilding.

## Assets, catalog and registry

Reuse licensed HM3D scenes. Upstream data and access instructions are linked by
[LH-VLN](https://github.com/HCPLab-SYSU/LH-VLN) and the
[LHPR-VLN dataset](https://huggingface.co/datasets/Starry123/LHPR-VLN).
These sources do not by themselves identify this delivery's exact filtered split.

Required local inputs are the pinned `_runtime.py`, `_metrics.py`, `vendor/`,
`data/unseen_test_filtered_0708.jsonl`, region segmentation/navmeshes, and the
HM3D scene-dataset configuration plus referenced scene files. The filtered
annotation SHA-256 is
`588b4203a7c245364e1dbae36b66706d29a650da85e2853f984c0f93685861cf`.
Do not substitute the unfiltered upstream episodes or regenerate row indices.

The separate [evaluation input package v1.0.0](https://github.com/panda7777777/SenseNova-RoboRSI/releases/tag/vln-evaluation-inputs-v1.0.0) supplies these exact
inputs: 105 original files, about 25.2 MB, plus checksums, source attribution and
license notices. It is distinct from HF training memory. Native files, annotations,
task order and original indices are preserved; no author-specific `gpt6-test`
or historical run directory is needed. Reuse a verified package when available
and skip downloading it again. Provide licensed scenes from an existing local copy.

```bash
export VLN_INPUT_DIR="$VLN_DELIVERY/input-download"
mkdir -p "$VLN_INPUT_DIR"
curl -fL 'https://github.com/panda7777777/SenseNova-RoboRSI/releases/download/vln-evaluation-inputs-v1.0.0/vln-evaluation-inputs-v1.0.0.tar.gz' \
  -o "$VLN_INPUT_DIR/vln-evaluation-inputs-v1.0.0.tar.gz"
printf '%s  %s\n' 'c20368128789105a55b0e0d507c7feaad6a8fbe28819f5d691bc6892450e1f66' \
  "$VLN_INPUT_DIR/vln-evaluation-inputs-v1.0.0.tar.gz" | sha256sum -c -
tar -xzf "$VLN_INPUT_DIR/vln-evaluation-inputs-v1.0.0.tar.gz" -C "$VLN_INPUT_DIR"
export VLN_INPUT_PACKAGE="$VLN_INPUT_DIR/vln-evaluation-inputs-v1.0.0"
export VLN_SCENES=/absolute/path/to/existing/hm3d
export VLN_REGISTRY="$VLN_DELIVERY/lhpr_vln-registry"
python -m robots.vln_shared.public_assets \
  --package "$VLN_INPUT_PACKAGE" --dataset lhpr_vln --scenes-root "$VLN_SCENES" \
  --output "$VLN_REGISTRY" --image "$VLN_IMAGE" --budget 500
```

Preparation verifies every small input and required scene. Missing files or hash
mismatches fail explicitly; no scene is downloaded or substituted. Hard links
are the default: use the same filesystem for scenes and output. Across filesystems,
`--copy-scenes` explicitly opts into copies and requires additional disk space.
Keep source scenes and output snapshots immutable. The output must be a new
directory without symlink components. The manifest is
`$VLN_REGISTRY/lhpr_vln/manifest.json`, with relative `assets` mounted read-only
at `/roborsi/assets/lhpr_vln` inside the container.

For stage two, use the same preparation command with `--budget 5000` and a new
output directory, then evaluate only stage-one scored failures with the same
Agent wheel, task and seed. Errors/timeouts/unscored cases are separate from
native failure and need separate handling. This input package does not supply
the full training catalogs for new exploration; frozen-ICL evaluation needs none.
Historical campaign migration and path mapping remain in [the build reference](BUILD.md).

Discover the Registry, check the image identity and inspect the full catalog:

```python
import os
from roborsi.benchmarks.registry import BenchmarkRegistry

registry = BenchmarkRegistry([os.environ["VLN_REGISTRY"]])
print(registry.discover())
locked = registry.lock("lhpr_vln")
assert locked.image_reference == os.environ["VLN_IMAGE"]
tasks = locked.list_tasks()
assert len(tasks) == 114
print(tasks[:3])
```

## Evaluate with an Agent

The standard entry `robots/lhpr_vln/agent.yaml` inherits its benchmark baseline
and the [shared navigation factory](../../robots/vln_shared/agents/baseline/README.md).
It constructs `EmbodiedAgent` with **Codex CLI 0.161.0**, **gpt-6-astra**, reasoning
**max**, MCP tools and no model-turn limit. This is the only delivered backend;
other backend/model combinations are not claimed as validated.

The standard source preset starts **without ICL**. To reproduce frozen-memory
evaluation, install the matching Agent wheel from the [contributor artifacts](https://github.com/panda7777777/SenseNova-RoboRSI/releases/tag/vln-delivery-20261102)
with `python -m pip install --no-compile /absolute/path/to/AGENT.whl`, then use
`--agent-config bundle:vln_shared_lhpr_vln` in the command below.
The ICL source is [lhvln/v1.0.0](https://huggingface.co/datasets/sensenova/SenseNova-RoboRSI-Packages/tree/495b4e2a9a3652726f52dd6363f7e2c5a77c2b34/lhvln/v1.0.0)
at commit `495b4e2a9a3652726f52dd6363f7e2c5a77c2b34`. Follow the
[download and checksum instructions](../../docs/vln_shared_evaluation.md#published-artifacts)
for the separate HF payload. A package download alone does not attach memory to a preset.

From the repository root, inspect one real task before launching the model:

```bash
export VLN_CODEX=/absolute/path/to/pinned/codex
bench eval --registry-root "$VLN_REGISTRY" --benchmark lhpr_vln \
  --task lhpr_vln:lhpr_vln-batch_1_episode_2 --seed 42 \
  --agent-config robots/lhpr_vln/agent.yaml \
  --agent-set "agent.codex_cli.executable=\"$VLN_CODEX\"" \
  --results-root "$VLN_DELIVERY/results" --run-id navigation-smoke-001 \
  --max-concurrent 1 --timeout-s 900 --case-timeout-s 7200 --dry-run
```

Remove `--dry-run` to evaluate. Use a fresh `--run-id` for every run. Remove
`--task` to evaluate the entire 114-task catalog at seed 42.
The task limit is 500 physical actions; `--timeout-s` and `--case-timeout-s`
control operations and wall time, not the action budget.

The standard command runs one fixed budget only. Use the
[shared campaign guide](../../docs/vln_shared_evaluation.md) for training-only
exploration/import and conditional 500/5000 evaluation. That migration launcher
requires the original campaign inputs; it does not fetch missing public assets.

## Results and validation

Check `<results-root>/<run-id>/summary.json`, each case's outcome, Runner
observations/actions, native metrics and image artifacts. Keep the CLI
`codex-cli/call-*/` runtime/stdin/events/stderr/result records and cleanup status.
Technical errors, timeouts and missing scores must remain distinct from native
task failure. Unavailable token usage is `null`, not zero.

The [release record](../../docs/vln_release_20261102.json) and
[downloadable container summaries](https://github.com/panda7777777/SenseNova-RoboRSI/releases/tag/vln-delivery-20261102)
cover final-image GPU reset, fixed actions, HTTP images/native scores, direct
native parity, final asset verification and resource cleanup. They do not
measure model accuracy. Three Agent wheels passed installation/resource checks
outside the checkout. The only reported real-model case is an earlier-build
R2R native/HTTP run; final-image model acceptance and full benchmark scores
have not been established for all datasets. The owner performs the full rerun
after merge. See the [test guide](../../tests/README.md) for offline checks.

The simulator, native sources and datasets retain their upstream access and
license terms. The public input package prepares the frozen test catalogs; new training
exploration remains a separate workflow with its own training inputs.
