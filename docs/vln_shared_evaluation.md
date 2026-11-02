# Shared VLN exploration and conditional evaluation

Initial code and usage guides for LHPR-VLN and VLN-CE (R2R / RxR)
are available in this branch.
The [frozen evaluation input package](https://github.com/panda7777777/SenseNova-RoboRSI/releases/tag/vln-evaluation-inputs-v1.0.0) and local scene preparation are
documented in [LHPR-VLN](../benchmarks/lhpr_vln/README.md) and
[VLN-CE](../benchmarks/vlnce/README.md). The historical importer below supports
new exploration/migration from an existing project's training inputs. It is
not needed to evaluate the public frozen test package with the released Agent wheels.

`scripts/run_vln_shared.sh` runs one training-only exploration per dataset, freezes
its workspace, evaluates every test at 500 physical actions, then restarts only
scored failures with a fresh 5000-action budget. Both runs use the same snapshot,
original annotation index and seed. A case may consume 5500 actions in total.
Timeouts, agent errors and missing native scores remain unscored. Dataset selection,
per-dataset test limits and a 500-only mode are available; two stages remain the default.

## Standard Agent entries

| Benchmark | Entry for `bench eval --agent-config` | Datasets |
| --- | --- | --- |
| VLNCE | `robots/vlnce/agent.yaml` | R2R, RxR |
| LHPR-VLN | `robots/lhpr_vln/agent.yaml` | LHPR |

Each entry extends its own `agents/baseline/agent.yaml`; both reuse the factory,
prompts and tool policy in `robots/vln_shared/agents/baseline/`. The old
`agents/vln_shared/agent.yaml` remains a compatibility alias. The shared Python
dependency lock stays in `agents/vln_shared/requirements.txt`.

The launcher composes the matching standard entry for both exploration and
evaluation, then creates self-contained presets under `agent-input/`. Exploration
uses the declared `exploration_prompt` resource, which is included in exports.
The standard entries alone start without frozen training memory; use the
generated `agent-input/<dataset>/agent.yaml` or its wheel for shared-exploration
evaluation. No Agent implementation is duplicated between Benchmarks.
Standard and compatibility entries share the repository as `source_root`;
export them outside the checkout. Only declared files are packaged.

Implementation and directory changes alter code/config fingerprints. Use a new output
directory for new runs; existing frozen presets, wheels and historical results
remain unchanged. Resume still requires the original code and input fingerprints.

Service manifests and runtime `adapter_version` now use
`benchmark_source_digest` over the same registered server files, so the full
checkout and exported service report the same version. Agent, exploration and
campaign edits do not change that version. The campaign separately fingerprints
all navigation modules and declared Agent resources for resume checks.
After this update, start a new output directory and regenerate delivery manifests
against the matching service source; existing results remain historical evidence.

## Setup and launch

Use Python 3.11 for RoboRSI, with separate native interpreters. Validated native
versions are Python 3.8.20 / Habitat 0.1.7 for VLNCE and Python 3.9.23 /
Habitat-Sim 0.3.0 for LHPR. Package inventories are recorded in each benchmark's
`native-environment.json`.

Install the complete pinned Codex CLI distribution (default `0.161.0`). When
Code Mode is enabled, its matching `codex-code-mode-host` executable must remain
beside the native CLI executable. Copying only the main binary omits this
required component. The campaign records companion hashes in its preflight.
`--dry-run` validates campaign inputs, while the CLI preflight checks the pinned
binary version; neither proves that model access or account quota is available.
A successful `codex login status` only confirms the configured login.

```bash
uv venv --python 3.11 .venv-vln
uv pip install --python .venv-vln/bin/python -r agents/vln_shared/requirements.txt
uv pip install --python .venv-vln/bin/python --no-deps -e .
export ROBORSI_PYTHON="$PWD/.venv-vln/bin/python"
scripts/run_vln_shared.sh run --source-project /path/to/gpt6-test \
  --history /path/to/navigation_shared_explore_20261006T161135Z \
  --codex /path/to/codex --output work/vln-shared-new
scripts/run_vln_shared.sh run --config work/vln-shared-new/inputs/campaign.json \
  --output work/vln-preflight --dry-run
scripts/run_vln_shared.sh run --config work/vln-shared-new/inputs/campaign.json \
  --output work/vln-shared-new --resume
```

Use `--reuse-exploration` when importing new inputs to verify and copy historical
training snapshots for the selected datasets. This never imports historical test
outcomes: selected tests still run at 500 first (314 tests by default). The default
performs new exploration. With an existing `--config`, its `import_snapshot` fields
control exploration reuse. For another host,
prepare a new config with its input paths and hashes before starting a campaign.

## Dataset selection and small checks

With a prepared local `campaign.json`, evaluate one LHPR, R2R and RxR episode
with a maximum of 500 physical actions each:

```bash
scripts/run_vln_shared.sh run --config /path/to/campaign.json \
  --datasets lhvln r2r rxr --episodes-per-dataset 1 --only-500 \
  --output work/vln-500-check
```

| Option | Behavior |
| --- | --- |
| `--datasets lhvln r2r rxr` | Select one or more space-separated names. `lhvln` aliases `lhpr_vln`; outputs use the canonical name. Defaults to every configured dataset. |
| `--episodes-per-dataset N` | Take the first N test episodes from each selected dataset, preserving catalog order, original indices and seeds. N must be positive and no larger than any selected catalog. Defaults to all tests; training tasks are unchanged. |
| `--only-500` | Stop after the 500-action stage. Native termination may occur earlier. Both scored successes and failures are final; no case enters the 5000-action stage. |
| `--dry-run` | Validate inputs and report selected datasets, counts, budgets and exploration sources without starting a model or simulator. |
| `--progress-interval 30` | Positive, finite heartbeat interval in seconds. Defaults to 30; starts, step changes and completion are reported immediately. |

Remove `--only-500` to use conditional 500-to-5000 evaluation on the same subset.
Also remove the dataset and episode options to use the full default campaign.
Use a new `--output` for each experiment. To resume, repeat the same command with
`--resume`, keeping its dataset, count and stage options. Changing those inputs
in an initialized directory is rejected. Input configs remain unchanged; the
selection applies to the current run. Preparation, validation and exploration
only require selected datasets, including their complete training catalogs.

Defaults: `gpt-6-astra`, reasoning `max`, Codex CLI `0.161.0`, concurrency 4,
one additional technical retry per invocation, 7200 seconds per test and 14400
seconds per exploration. Controls are `--concurrency`, `--retries`,
`--case-timeout`, and `--exploration-timeout`. Action budgets are independent of
model turns; technical retries retain the current stage's budget. Exit code 2
means incomplete scoring. Native task failure is a valid completed evaluation.

The declared `tool-policy.toml` pre-approves only local navigation and memory
tools within each workspace. Shell commands retain `workspace-write` sandboxing.
The working project must allow project configuration; restrictive managed policy
remains authoritative. See the [official MCP settings](https://learn.chatgpt.com/docs/extend/mcp?surface=cli).

## Live terminal progress

`run` reports progress immediately on **stderr**, with no extra switch or `python -u`.
The final summary JSON stays on **stdout**. Preparation, validation, historical
memory import and training exploration show their current phase. Evaluation
lines identify the dataset, original episode ID and attempt number. Example format:

```text
[09:00:00] event=stage_start budget=500 cases=3
[09:00:00] dataset=rxr episode=198 attempt=attempt-0001 retry=0/1 event=start steps=?/500 elapsed=0.0s idle=0.0s artifacts=...
[09:00:04] dataset=rxr episode=198 attempt=attempt-0001 retry=0/1 event=step steps=0/500 elapsed=4.0s idle=0.0s
[09:00:12] dataset=rxr episode=198 attempt=attempt-0001 retry=0/1 event=step steps=32/500 elapsed=12.0s idle=0.0s
[09:00:30] dataset=rxr episode=198 attempt=attempt-0001 retry=0/1 event=heartbeat steps=32/500 elapsed=30.0s idle=18.0s
[09:00:40] dataset=rxr episode=198 attempt=attempt-0001 retry=0/1 event=attempt_end steps=32/500 elapsed=40.0s idle=28.0s status=agent_error
[09:00:40] dataset=rxr episode=198 attempt=attempt-0002 retry=1/1 event=retry steps=?/500 elapsed=0.0s idle=0.0s artifacts=...
```

- `steps` counts physical actions, including native recovery actions, and updates
  after each returned batch. `?` means reset has not returned yet; the initial
  observation reports 0. These are neither model turns nor RPC batch counts.
- `heartbeat` repeats the latest count at the configured interval. `elapsed` is
  attempt duration; `idle` is time since the last step update. A heartbeat means
  the launcher is waiting, not that the model connection is healthy. It does not
  change timeouts or scoring.
- `retry=1/1` starts the one allowed extra technical retry for this invocation,
  resetting its count. The 5000-action stage has its own budget and initial count.
  Attempt numbering continues on resume.
- `attempt_end` distinguishes `success`, `task_failure` and technical errors.
  Error details remain in `outcome.json` and child logs under the attempt's
  `artifacts` directory. Only scored native failures enter stage two.
- `reuse` indicates an already scored result. `stage_progress processed=N/M`
  counts handled episodes, including unscored cases after exhausted retries.
  `dataset_end` reports scored, successful and incomplete counts; the final
  `run_end status=complete/incomplete` describes scoring completion.

To save both streams, append `> summary-console.json 2> progress.log` to the
command and run `tail -f progress.log` in another terminal. These files are
relative to the invoking shell's directory; full results remain in
`<output>/evaluation/summary.json`. To keep live progress visible while saving
only the final JSON, append just `> summary-console.json`.

This progress change also alters code fingerprints. Use a new output directory,
such as `work/vln-500-progress` above. Existing `work/vln-500-cli-check` results
remain intact and cannot be resumed directly with the changed code.

## Protocol and outputs

- Catalogs retain 114 LHPR, 100 R2R and 100 RxR tasks, with disjoint training and
  test scenes. R2R preserves the historical `rand100` starts and ground truth;
  RxR preserves the selected guide episodes.
- Batches contain at most 32 actions and stop at termination or an LHPR subtask
  boundary. Native force-escape actions consume the same physical budget. RPC
  step indices and physical action counts are recorded separately.
- Stage two starts after every first-stage case completes or exhausts technical
  retries. Its size is computed from this run, never hard-coded to 128.
- `evaluation/summary.json` retains both stages and selects an entire scored
  run. Its `budgets` field is `[500]` or `[500, 5000]`. In 500-only mode each
  `stage_5000` result is `null`; `stage-5000.json` has `enabled: false` and an empty
  queue. Rate denominators use the selected episode counts.
  It reports native metrics, action counts, wall time and available token
  usage; unavailable usage and incomplete full-dataset rates remain `null`.
  Scored-only rates and incomplete counts are explicit. Exploration and technical
  retry costs are separate; imported memory does not invent exploration cost.
- Each attempt has request/result records, stdout/stderr, public Runner traces,
  CLI runtime/stdin/events/result, HTTP images and native evidence. Snapshots and
  committed results are hashed. Locks prevent duplicate launches; resume can
  recover committed `bench` results. SIGINT/SIGTERM clean up owned processes.
- Snapshot export removes generated caches and empty directories from the new
  copy. Original projects and historical results remain unchanged.

The local bridge checks native code and annotation inputs. Delivery additionally
requires a complete scene asset manifest. Observation artifacts are retained;
this change does not add per-physical-action videos. Historical LHPR videos
remain sampled-observation videos.

## Validation and delivery

Run offline checks in the RoboRSI service environment. Native checks additionally
require the licensed scenes, pinned native source and isolated simulator environments.
They compare fixed actions through the HTTP service with the native adapter directly;
they do not call a model or produce benchmark accuracy results.

```bash
pytest tests/unit_tests/robots/test_vln_shared.py \
  tests/unit_tests/robots/test_vln_cli.py \
  tests/unit_tests/robots/test_vln_progress.py \
  tests/unit_tests/robots/test_vln_agent_entries.py \
  tests/unit_tests/robots/test_vln_assets.py \
  tests/unit_tests/robots/test_vln_runtime.py \
  tests/unit_tests/roborsi/benchmarks/test_vln_registration.py -q
ROBORSI_VLN_CONFIG=work/vln-shared-new/inputs/campaign.json \
  pytest tests/integration_tests/robots/test_vln_shared_native.py -v
scripts/run_vln_shared.sh smoke --config work/vln-shared-new/inputs/campaign.json \
  --dataset r2r --output work/vln-smoke
python -m roborsi.agents.package --config work/vln-shared-new/agent-input/r2r/agent.yaml \
  --name vln_shared_r2r --output /tmp/vln-shared-r2r-package
python -m build --wheel /tmp/vln-shared-r2r-package
```

See [VLNCE packaging](../benchmarks/vlnce/README.md) and
[LHPR packaging](../benchmarks/lhpr_vln/README.md). Both service wrappers are registered in this public repository; the
frozen evaluation inputs are distributed separately from the training-memory packages.
Keep image acceptance evidence, immutable image digests and HF revisions with the
release. Report full-model scores only after a fresh run of the merged public commit.

Install exported Agent wheels with `python -m pip install --no-compile <wheel>`.
This preserves the frozen resource inventory when training memory includes Python
helpers. Keep Codex CLI `0.161.0` available at the configured executable path;
the separately pinned Python SDK is not the executable version check.

## Published artifacts

[Release identities and checks](vln_release_20261102.json) pin both GHCR images,
the frozen ICL commit and the independent framework dependency. Image references
are in the [LHPR recipe](../benchmarks/lhpr_vln/README.md) and
[VLN-CE recipe](../benchmarks/vlnce/README.md). GHCR visibility is pending; the
HF resources are in [PR #1](https://huggingface.co/datasets/sensenova/SenseNova-RoboRSI-Packages/discussions/1), awaiting maintainer merge. All 119 submitted
files were re-downloaded at the following commit and matched the approved bytes.

Reuse an already verified local copy. For a new machine, download and verify:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="sensenova/SenseNova-RoboRSI-Packages",
    repo_type="dataset",
    revision="495b4e2a9a3652726f52dd6363f7e2c5a77c2b34",
    allow_patterns=["lhvln/v1.0.0/*", "vln-ce/v1.0.0/*", "scripts/verify_package.py"],
    local_dir="/path/to/roborsi-icl",
)
```

```bash
python /path/to/roborsi-icl/scripts/verify_package.py /path/to/roborsi-icl/lhvln/v1.0.0 \
  --manifest-sha256 16e34a59a0d806135926945be328251b4790b7fc0dbfefa0733e0911311423dc
python /path/to/roborsi-icl/scripts/verify_package.py /path/to/roborsi-icl/vln-ce/v1.0.0 \
  --manifest-sha256 c13ee26190ad749fc362d657e5e2e57cb54f00a6719efa29300b26b17df1b044
```

Set each prepared campaign dataset's `import_snapshot` to the corresponding
absolute `payload` directory: `lhvln/v1.0.0/packages/lhvln/payload`,
`vln-ce/v1.0.0/packages/r2r/payload`, or `vln-ce/v1.0.0/packages/rxr/payload`.
Do not pass the aggregate repository root. Do not overwrite a running campaign.
The original HF payload remains frozen; the Agent exporter creates its separate
canonical snapshot as described below.

Before framework PR #36 is merged, reproduce the validated Docker host in an
isolated worktree; `origin` here is the public OpenSenseNova repository:

```bash
git fetch origin pull/33/head
git worktree add --detach /path/to/vln-delivery 1435a63f853649c9e04fe135b664b9835d10fde7
cd /path/to/vln-delivery
git fetch origin pull/36/head
git cherry-pick 8e4894abffc6f94e4bf614ac1af45ae93303368a
```

Install the pinned host requirements from the installation section. Export and
install the generated dataset Agent wheel using `--no-compile`. Prepare the
licensed scene assets and their manifest using the relevant benchmark recipe.
For a 5000-step Registry, call `robots.vln_shared.assets.manifest(assets, output,
image, budget=5000)` with `Path` arguments and a new output manifest. Budgets are
part of the manifest, not an Agent turn limit. Keep the two stage outputs separate.

Inspect the Registry and validate the standard installed Agent entry:

```python
from roborsi.benchmarks.registry import BenchmarkRegistry

locked = BenchmarkRegistry(["/path/to/r2r-registry"]).lock("vlnce")
print(locked.list_tasks()[:3])
```

```bash
bench eval --registry-root /path/to/r2r-registry --benchmark vlnce \
  --task vlnce:r2r-7 --seed 0 --agent-config bundle:vln_shared_r2r \
  --agent-set 'agent.codex_cli.executable="/path/to/pinned/codex"' \
  --results-root /path/to/new-results --run-id r2r-check \
  --max-concurrent 1 --dry-run
```

Remove `--dry-run` only when ready to run the model. Use a fresh results/run ID
for each run. This command evaluates one 500-step R2R task; it does not implement
the conditional 5000-step campaign. Use the campaign entry below for that protocol.

Final image acceptance compared HTTP and direct native execution inside each
image, including initial image hashes, actions, native scores and cleanup.
Navigation unit tests: 1613 passed; standalone framework fix: 1540 passed.
Both PRs passed Python 3.11/3.12 CI and pre-commit. Installed Agent resources
and host dependencies were verified outside the checkout. These checks do not
claim a new full-model benchmark score; the owner will rerun after merge.

Download the three Agent wheels, the validated framework wheel, host dependency lock,
component license notices and container summaries from the [contributor artifact release](https://github.com/panda7777777/SenseNova-RoboRSI/releases/tag/vln-delivery-20261102).
Verify its `SHA256SUMS` before installation. This prerelease accompanies PRs #33/#36;
it is not an OpenSenseNova organization release. The framework wheel already contains
the separate timeout fix, and the Agent wheels preserve their frozen resource fingerprints.

## Install the pinned CLI

Use the verified official npm Linux x64 artifact below and retain its complete
distribution, including companion programs. Reuse an existing copy if its
hashes match. The task-specific directory leaves a global CLI unchanged.

```bash
export VLN_CLI_DIR=/absolute/path/outside/checkout/codex-0.161.0
mkdir -p "$VLN_CLI_DIR"
python - "$VLN_CLI_DIR" <<'PY'
import base64
import hashlib
from pathlib import Path
import sys
import urllib.request

archive = Path(sys.argv[1]) / "codex-0.161.0-linux-x64.tgz"
url = "https://registry.npmjs.org/@openai/codex/-/codex-0.161.0-linux-x64.tgz"
if not archive.exists():
    with urllib.request.urlopen(url) as source, archive.open("xb") as target:
        import shutil
        shutil.copyfileobj(source, target)
expected = "AVkNzeJWyCHHPyg5vxD8UE6aFsYaOwq0/x6mtmkOKy6LdjvsyxLr7Q7UULlk+Ma3+PnrN8uIxG2/XGPQ936hgA=="
assert base64.b64encode(hashlib.sha512(archive.read_bytes()).digest()).decode() == expected
print("Verified official 0.161.0 Linux x64 archive")
PY
tar -xzf "$VLN_CLI_DIR/codex-0.161.0-linux-x64.tgz" -C "$VLN_CLI_DIR"
export VLN_CODEX="$VLN_CLI_DIR/package/vendor/x86_64-unknown-linux-musl/bin/codex"
"$VLN_CODEX" --version
sha256sum "$VLN_CODEX" "$(dirname "$VLN_CODEX")/codex-code-mode-host"
```

Expected version: `codex-cli 0.161.0`. The two expected SHA-256 values, in order:

```text
9a820c17865fa825d04db416818679a9d63bd72e50835c396f496e5684626c9c
e87096087c02aa6ab90a5a85031659665aa0bf3ee20c07ca0b8ab701c83fcd5b
```

Set `agent.codex_cli.executable` to this absolute path and authenticate using
`"$VLN_CODEX" login` in the Agent account. The preset uses native CLI provider/account
configuration; keep credentials out of YAML and images. `login status` confirms
authentication, not quota or a successful request to the target model.
