# VLN-CE (R2R / RxR)

[简体中文](README.zh-CN.md) · [All guides](../../docs/README.md) · [Build reference](BUILD.md)

This integration exposes VLN-CE (R2R / RxR) through the shared Benchmark service,
Registry and Runner, with isolated native workers and native scoring.
Initial code and usage guides are available in this branch.
Pinned artifacts and installation requirements are documented below and in the
[shared evaluation guide](../../docs/vln_shared_evaluation.md).

## Tasks, actions and scores

R2R and RxR use **separate 100-task catalogs**, both with seed **0**.
R2R preserves the selected `rand100` annotations, starts and reference paths.
RxR uses the selected `val_unseen` guide episodes with their original languages
and indices. Neither catalog is the entire upstream validation set.
Native source is VLN-CE `729d141b2ee10628061ada74dd3a5b9f70faeba5`.

Actions are `{"actions": ["move_forward", "turn_left", "stop"]}` with 1–32
entries. R2R supports `stop`, `move_forward`, `turn_left`, `turn_right`; RxR
also supports `look_up`, `look_down`. The native configuration supplies motion
increments. Each physical action counts toward the budget; a batch stops when
the episode ends. Observations expose the instruction and one 512×512 RGB view.
Oracle progress and ground-truth paths are not model observations.

Success is native `success`; RoboRSI score is `float(success)`. Native SPL,
nDTW, distance and other returned metrics remain in the result. STOP ends the
episode; reaching the native step limit is truncation. Unfinished episodes
remain unscored rather than being counted as task failures.

The standard Registry below uses 500 physical actions. The separate shared
campaign retries only scored failures with a fresh 5000-action budget; the same
task, original index, seed and frozen memory are used in both stages.
See the [protocol and output guide](../../docs/vln_shared_evaluation.md#protocol-and-outputs).

## Environment and image

Use Linux x86_64, an NVIDIA GPU/driver, Docker with NVIDIA Container Toolkit,
and Python 3.11 for the Agent/evaluator. Validation used an RTX 4090.
The image contains Python 3.8.20 / Habitat-Lab and Habitat-Sim 0.1.7 / NumPy 1.23.5; the HTTP service uses a separate Python 3.11.
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
export VLN_IMAGE=ghcr.io/opensensenova/sensenova-roborsi-vln-ce@sha256:f54607562ca465101d6e1091f97d776755c2b00517f1852643bd8e0161c0e787
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

Reuse licensed Matterport3D scenes. Obtain source and dataset instructions from
the [pinned VLN-CE README](https://github.com/jacobkrantz/VLN-CE/blob/729d141b2ee10628061ada74dd3a5b9f70faeba5/README.md).
Scene files follow `mp3d/<scene>/<scene>.glb`; the configured scene root must
match the paths resolved from the annotation. Retain upstream license notices.

Required inputs include `habitat_extensions/` from that commit, R2R
`rand100.json.gz` and `rand100_gt.json.gz`, RxR `val_unseen_guide.json.gz` and
`val_unseen_guide_gt.json.gz`, plus the exact selected episode IDs/order.
R2R annotation SHA-256 is
`b563a76dbbdec5da61ee656ccf5e096b62070423e633987fb437ca1003b4fc67`;
RxR annotation SHA-256 is
`93abddce58e8232867a8738daa8107eb82953693a9d5e70779d1eb00386b222a`.

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
export VLN_SCENES=/absolute/path/to/existing/scene_datasets
export VLN_REGISTRY="$VLN_DELIVERY/r2r-registry"
python -m robots.vln_shared.public_assets \
  --package "$VLN_INPUT_PACKAGE" --dataset r2r --scenes-root "$VLN_SCENES" \
  --output "$VLN_REGISTRY" --image "$VLN_IMAGE" --budget 500
```

Preparation verifies every small input and required scene. Missing files or hash
mismatches fail explicitly; no scene is downloaded or substituted. Hard links
are the default: use the same filesystem for scenes and output. Across filesystems,
`--copy-scenes` explicitly opts into copies and requires additional disk space.
Keep source scenes and output snapshots immutable. The output must be a new
directory without symlink components. The manifest is
`$VLN_REGISTRY/vlnce/manifest.json`, with relative `assets` mounted read-only
at `/roborsi/assets/vlnce` inside the container.

For stage two, use the same preparation command with `--budget 5000` and a new
output directory, then evaluate only stage-one scored failures with the same
Agent wheel, task and seed. Errors/timeouts/unscored cases are separate from
native failure and need separate handling. This input package does not supply
the full training catalogs for new exploration; frozen-ICL evaluation needs none.
Historical campaign migration and path mapping remain in [the build reference](BUILD.md).

For RxR, use the same package/scenes and change `--dataset` to `rxr` with a
separate `$VLN_DELIVERY/rxr-registry` output. Both catalogs use benchmark ID
`vlnce`, so do not put them under the same Registry search root. Each has 100 tasks.

Discover the Registry, check the image identity and inspect the full catalog:

```python
import os
from roborsi.benchmarks.registry import BenchmarkRegistry

registry = BenchmarkRegistry([os.environ["VLN_REGISTRY"]])
print(registry.discover())
locked = registry.lock("vlnce")
assert locked.image_reference == os.environ["VLN_IMAGE"]
tasks = locked.list_tasks()
assert len(tasks) == 100
print(tasks[:3])
```

## Evaluate with an Agent

The standard entry `robots/vlnce/agent.yaml` inherits its benchmark baseline
and the [shared navigation factory](../../robots/vln_shared/agents/baseline/README.md).
It constructs `EmbodiedAgent` with **Codex CLI 0.161.0**, **gpt-6-astra**, reasoning
**max**, MCP tools and no model-turn limit. This is the only delivered backend;
other backend/model combinations are not claimed as validated.

The standard source preset starts **without ICL**. To reproduce frozen-memory
evaluation, install the matching Agent wheel from the [contributor artifacts](https://github.com/panda7777777/SenseNova-RoboRSI/releases/tag/vln-delivery-20261102)
with `python -m pip install --no-compile /absolute/path/to/AGENT.whl`, then use
`--agent-config bundle:vln_shared_r2r` in the command below.
The ICL source is [vln-ce/v1.0.0](https://huggingface.co/datasets/sensenova/SenseNova-RoboRSI-Packages/tree/495b4e2a9a3652726f52dd6363f7e2c5a77c2b34/vln-ce/v1.0.0)
at commit `495b4e2a9a3652726f52dd6363f7e2c5a77c2b34`. Follow the
[download and checksum instructions](../../docs/vln_shared_evaluation.md#published-artifacts)
for the separate HF payload. A package download alone does not attach memory to a preset.

From the repository root, inspect one real task before launching the model:

```bash
export VLN_CODEX=/absolute/path/to/pinned/codex
bench eval --registry-root "$VLN_REGISTRY" --benchmark vlnce \
  --task vlnce:r2r-7 --seed 0 \
  --agent-config robots/vlnce/agent.yaml \
  --agent-set "agent.codex_cli.executable=\"$VLN_CODEX\"" \
  --results-root "$VLN_DELIVERY/results" --run-id navigation-smoke-001 \
  --max-concurrent 1 --timeout-s 900 --case-timeout-s 7200 --dry-run
```

Remove `--dry-run` to evaluate. Use a fresh `--run-id` for every run. Remove
`--task` to evaluate the entire 100-task catalog at seed 0.
The task limit is 500 physical actions; `--timeout-s` and `--case-timeout-s`
control operations and wall time, not the action budget.

For RxR, use its separate Registry, `--task vlnce:rxr-198`, seed 0 and
`--agent-config bundle:vln_shared_rxr` for frozen ICL. Do not use the R2R wheel
for RxR. Run the two catalogs separately and retain their individual summaries.

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
