# Validation scope

## SenseNova-RoboRSI migration — 2026-11-18

The SIMPLE source, recipes and tests were imported from the local NewRPent
checkout. All later sections retain upstream validation history, including
original commit identities, image/source digests and PR status. Those records
do not certify a build against this repository's shared framework. The current
checkout must be exported, rebuilt and accepted independently before deployment.

The branch starts directly from fetched `origin/main`
`9e7e7ad17b8957a5b8a530b8f4a0bbc11bb00555`.

### Commit and file coverage

- `2d06bcf1417f779f68611ab3671cd0e13ee63056`: SIMPLE runtime registration;
  its eleven-file inventory is identical to the migrated registration.
- `237484ebf1d9863fdcb9694e4f4e329876d3730a`: native adapter, Agent preset,
  recipes, guides and tests, including the entry in `benchmarks/README.md`.
- The nine later SIMPLE commits through `02dafd79` are included using the
  source checkout's final file contents, including the sixth task, HTTP
  lifecycle checks, native termination checks and cleanup retry repair.

All 46 paths touched by these eleven commits are present. The 44 SIMPLE-owned
files include 38 byte-identical files and six guides adapted only for this
repository's layout and upstream-history provenance. The shared registration
and benchmark index are integrated with this repository. The full commit/file
inventory and original hashes are in
[`docs/simple-port-sources.json`](../../docs/simple-port-sources.json).

### Checks run for this migration

Linux checks used the existing WSL Ubuntu Python 3.12 environment and Linux
`/tmp` for permission and symlink fixtures:

```bash
pytest tests/unit_tests/robots/simple \
  tests/unit_tests/roborsi/benchmarks/test_runtime_package.py \
  tests/unit_tests/roborsi/benchmarks/test_build_inputs.py \
  tests/integration_tests/robots/simple -q
```

The first run passed 72 checks and skipped the opt-in native smoke; its only
failure was the Agent preset check because the test environment lacked the
project-declared `pydantic_ai` dependency. After installing that dependency in
the isolated test venv, the focused rerun passed **45 checks in 64.27 seconds**:

```bash
pytest tests/unit_tests/robots/simple \
  'tests/unit_tests/roborsi/benchmarks/test_runtime_package.py::test_bundle_imports_and_service_help_without_agent_code[simple]' -q
```

Together the runs cover **73 distinct passing offline checks and one skipped
native smoke**. A preliminary Windows run encountered Linux-only permission
APIs and sandboxed socket restrictions; the Linux rerun preserves those checks.
No tests were weakened for the host platform.

The command `python -m benchmarks.package --benchmark simple --output <new-dir>`
also exported 44 runtime files, with matching checkout/export source SHA256
`d911592b8b79c2183b8eacb2cb01be4ec34100091a9f58eb4b21296cc941f101`.
The exported runtime excludes Agent presets. Focused Ruff lint and formatting
checks passed for all 24 changed/imported Python files; `git diff --check`
passed. These checks do not establish a new Docker build, GPU task success or
equivalence to the historical published images.

## Imported upstream validation history

This PR prepares a native SIMPLE integration candidate against RoboRSI `main`.
A service candidate is now published; it is not full task acceptance or a
reproduction of historical scores. The sections below retain chronological
validation history. For the current repair candidate, see the 2026-10-19 section and
[DELIVERY.md](DELIVERY.md). The v0.1.0 GHCR image is historical and lacks the
cleanup retry fixes; earlier sections describe their original artifacts only.

## Cleanup retry repair and main synchronization — 2026-10-19

Rebased without conflicts onto main `e357c6a5ae41878ad839d34547d37b0c5f0fbfac`,
which includes merged registration #86 and shared facade retry fix #88.
`SimpleAdapter.closed` continues to reject new work immediately; a separate
cleanup-complete flag is set only after worker stop, Kit permission restoration
and RPC cleanup return successfully. A failed close can therefore be retried.

Three regressions use real CPU workers and the current AdapterFacade. Each
injects one failure (worker stop, Kit cleanup, RPC close), checks that new work
is rejected after failure, retries close, and checks process exit, restored
screenshot permissions and idempotent completed facade close. All three failed
on the old adapter and passed after repair. Each fixture forcibly cleans its
worker and directory even on assertion failure. Focused SIMPLE/packaging/HTTP/
service checks: **99 passed** (33.77 s). These are scripted-physics CPU tests,
not native successful-task evidence.

The standard exporter directly produced 44 registered runtime files whose bytes
match this checkout. Relative to v0.1.0, exactly three runtime files changed:
`runtime_source.py`, shared `adapter_service.py` (#88), and SIMPLE's
`roborsi_adapter.py` (this repair). In particular this update is not just a source
inventory change; the historical service image must not be called fixed.

A new local service image was built using the original native base by digest.
Dependency checking, the unchanged standalone runtime verifier and exclusion of
Agent SDKs passed. Its exported source SHA256 is
`3833e3e3dc272828e9a7bc5070b65e1735127a7faee6dd07eba68e5ea16a848c`.
Published as `v0.1.1-cu128` from adapter commit
`a1cfbfc802c6df8f1e5db8c5e2c96ec559275255`, with fixed registry reference:
`ghcr.io/opensensenova/sensenova-roborsi-simple@sha256:4b6f9eb28fa87aebba7a3d46fcf0498753d81a47719dadf30a78c22d1f208213`.
Authenticated push, registry manifest digest verification, digest-pinned pull and
runtime verification passed on the build host. Temporary GHCR authentication was
removed after logout. This publication does not add an independent-host check.
The prior native simulator, controller weights, actions and scoring conditions
are retained. The new image passed real BendPick L0/E00 (AMO, H200 GPU5):
**400/400 original rows, 469.99 s**, normal step-limit truncation, native finish
`success=false`, `score=0.0`, no first-success step, matching 400-row action trace,
46 downloaded artifacts including 21 RGB observations, case close/delete and
owned container cleanup. The standalone source verifier and a 46-file source/
recipe plus known-credential-location audit passed. No model calls were made.
This covers native failure termination and resource cleanup, not a successful
manipulation or independent recipient-host test.

An initial host full-suite run had 2,756 passes, 3 skips and 8 failures, all from
Bench2Dex's `/usr/bin/python3` sandbox lacking NumPy. The unmodified main
reproduced the same eight failures. System dependencies were then provided only
inside a temporary mount namespace; no host Python, benchmark tests or production
sandbox were changed. The final full suite on `a1cfbfc8` passed **2,764 tests,
3 skipped**, with two deprecation warnings in **359.59 s**. All-files pre-commit
passed. This subsequent publication record changes documentation only.

Actual GitHub CI was attempted on `a1cfbfc8`, but both
[Unit tests](https://github.com/SenseTime-FVG/RoboRSI/actions/runs/39242815255) and
[Pre-commit](https://github.com/SenseTime-FVG/RoboRSI/actions/runs/39242815309)
were rejected before any step due to failed account payments or the spending
limit, as recorded in job annotations. No GitHub test executed and no passing
current-head CI is claimed. Draft was restored; an organization administrator
must resolve Actions billing before the latest commit can be checked.

## Executed checks

- Fresh independent development venv: Python 3.11.15; repository `.[test]` and
  pre-commit 4.6.2 installed. No existing SIMPLE or global Python was modified.
- Focused offline suite: 32 passed. The opt-in native smoke was explicitly run
  and passed separately; it is skipped in ordinary CPU-only invocations.
  The SIMPLE directory remains a namespace module, matching the other new
  Benchmark adapters rather than registering an unsupported legacy RobotSpec.
- Full repository unit suite: **2601 passed, 3 skipped** (314.28 seconds).
  The 32 focused SIMPLE checks were rerun after the final support for already-filled
  reset instruction templates; both real 30-case catalogs were checked offline.
- All-repository Ruff lint and formatting passed. A real pinned Tabletop archive
  was exported and loaded successfully with all 30 test cases.
- Focused offline checks cover exact six-task 1×/10× budgets; nonfinite, boolean,
  malformed and excess actions; private reset hashes and task membership; early
  success inside a chunk; exact physical-row timeout; no privileged observation
  fields; process/RPC cleanup and model-credential isolation.
- Real local child process, HTTP RPC, Benchmark HTTP API, PNG media hashes,
  result/artifact downloads and deletion use the actual framework implementation.
  The test native physics is scripted and framework source version discovery is
  mocked because registration belongs to the separate maintainer PR.
- The standard `EmbodiedAgent` preset is loaded without a model request and its
  Agent source bundle was exported. No installed-wheel relocation is claimed.
  Complete repository checks are recorded in the PR summary.

## Native smoke executed on 2026-10-15

`G1WholebodyTabletopGraspMP-v0`, released L0 episode 00, reset seed 20261002,
AMO with continuous Dex3 targets: reset, original 60-step stand warmup, then three
state-hold policy rows. The opt-in integration test passed in 292.06 seconds.
Four original 640x360 PNG observations were retained (initial and steps 1/2/3),
together with three action records and reset/device evidence. The owned native
process exited. No model API call or full-task success is claimed.

The test used a separate clean pinned SIMPLE source checkout with the three
original AMO checkpoints checked against their committed LFS hashes. Assets were
mounted outside the checkout. Native dependencies reused an existing installation:
Python 3.10.20, Isaac Sim/Kit distribution 4.5.0.0, MuJoCo 3.3.6,
Torch 2.7.0+cu128, NumPy 1.26.4, Gymnasium 1.3.0; H200 GPU 6,
NVIDIA driver 570.195.03, kernel 6.8.0-51-generic. This is not a clean-container
installation or a complete native dependency lock.

Host-provided GLU/Vulkan libraries and NVIDIA ICD were required. Torch/Isaac/EGL
selected GPU 6; CUDA_VISIBLE_DEVICES was not remapped. Two traced AMO CUDA device
constants were relocated in memory; checkpoint bytes and rendering quality
settings were unchanged. Visual inspection found recognizable hands/table/objects
but substantial grain and bright specular highlights. This does not establish
RTX-equivalent rendering or benchmark scores.

Earlier failed attempts exposed a missing graphics-library path and an incomplete
external asset-root override. The override now covers both native asset resolvers;
startup errors are persisted before Isaac's close can exit the process. Failed
attempts were preserved outside the source PR.

## Local native container built on 2026-10-15

`benchmarks/simple/native/` builds a separate simulator base from the pinned public
SIMPLE/submodule revisions and the digest-pinned CUDA 12.8 Ubuntu 22.04 base.
Docker reports 46,242,168,155 bytes. The local image ID is
`sha256:781f291b5b52e801314f2fc0411d372f5d333b93e5efa127e0e51419a6ff6e69`;
this is not a published GHCR service digest. The image records its resolved
Python/system packages and source revisions under `/opt/simple-build`.

`pip check` passed. The build audit checked 2,157 source/recipe text files and
known credential locations, rejected model SDK dependencies, and found no
credential-pattern matches. The image configuration/history scan also had no
matches. Only the six allowlisted recipe inputs entered the build context;
no private source checkout, model key, SSH agent, experiment log or home directory
was supplied. Assets and catalog data were mounted read-only for smoke tests.

Container Tabletop L0/E00 completed reset and three state-hold rows in 249.48
seconds, with four 640x360 PNGs, finite 32-value state, action records, reset
identity and actual Carb settings. Both container and launcher exited with code 0.
It ran without network access or model calls, as a non-root user on physical H200
GPU 6 (container GPU 0), with the unchanged 570.195.03 driver. The actual renderer
was `RayTracedLighting`, AA `2`, limitedOps `false`, indirect diffuse `true` and
OptiX denoiser `true`. The images retain strong grain/specular highlights.

Initial container attempts exposed an asset mount blocked by root-squashed
storage, Git LFS temporary-storage permissions, and a missing implicit
`huggingface_hub` dependency. These were corrected without modifying host drivers
or private directory permissions. One completed Tabletop run had a launcher error
after its script was edited while running; its Kit log/images were retained and
the clean run above replaced it for validation.

An XMove native attempt exposed disagreement between `episodes.jsonl` task text
and the reset language template. All 30 XMove cases in the pinned archive share
this discrepancy; all 30 Tabletop cases match. The adapter now checks the native
instruction against the unmodified reset, preserves the Agent's episode text,
and records both. A regression check covers differing and already-filled
templates; no mismatch is silently rewritten or treated as task success.

After that correction, container XMovePick L0/E00 also completed reset,
stabilization and three state-hold rows in 285.17 seconds on physical H200 GPU 5.
Its two original ONNX balance/walk policies loaded, four PNGs and action/state
records were retained, and both launcher and container exited with code 0. The
saved reset evidence explicitly records the prompt/template disagreement. Visual
inspection found the hands, table and target visible with substantial noise and
specular highlights. This covers one decoupled WBC case and one AMO case; it
does not establish full task success or six-task/controller coverage.

## Review follow-up: registered export and bounded HTTP verification

Framework registration is proposed separately in
[PR #86](https://github.com/SenseTime-FVG/RoboRSI/pull/86), commit
`c9bf2290bd32d55cb9b993ae0e38ca2c2cc093d1`, based on `main`. Its full local
framework suite passed (2,569 passed, 3 skipped), and GitHub actually executed
Python 3.10/3.11/3.12 tests, minimum-runtime checks, Dashboard tests and lint.
Those checks all passed. This is registration CI, not CI for the still-Draft #85.

An isolated combined checkout used this adapter and exactly the proposed
registration. The standard exporter produced the service bundle; its source
digest matches the checkout, and standalone imports/CLI help passed with Agent
imports blocked. A local shared HTTP service candidate was built with the
unchanged source-boundary verifier, no Agent/model SDKs and the eleven-package
hashed `requirements-http.lock`. No published registry digest is claimed.

The first actual HTTP reset exposed a difference from the earlier direct-runtime
smoke: arbitrary `GIT_CONFIG_*` variables are intentionally not inherited by the
child. A non-root worker therefore rejected the root-owned image checkout. The
worker now scopes `safe.directory` to its explicitly configured source and LFS
scratch to its own episode. A real-Git CPU regression simulates different checkout
ownership and verifies that wrong revisions and tracked-source modifications are
still rejected; it never loads Isaac.

The next HTTP check executed three native Tabletop rows, downloaded feedback, and
closed its worker, but case deletion failed: Kit created its private `screenshots`
directory with mode 000. After worker exit, the adapter now adds only owner access
to that exact owned directory. It does not follow directory links or change global
permissions. CPU regressions cover actual worker close and an external symlink.
Both unsuccessful checks are retained as diagnostic evidence, not passing runs.

`benchmarks.simple.smoke_http` has a fixed three-row execution ceiling and never
constructs a model client. It verifies real service identity, keeps downloaded
RGB/state/action evidence, cancels an unfinished case without assigning a task
score, and deletes only its own case. CPU HTTP regressions cover both natural
termination/finish and nonterminal cancellation. Native `finish` cannot be claimed
from a nonterminal three-row check; full task success remains untested.

The corrected local HTTP image
`simple-service:candidate03-cu128-20261015` passed the real Tabletop L0/E00 check
in **257.55 seconds**: reset, three native rows, four RGB/state observations,
eight downloaded artifacts, cancellation, worker cleanup and case deletion.
The non-root container exited afterwards. This used physical H200 GPU 6, the
original 800-row budget, and zero model calls. `native_finish_exercised=false`:
the case was deliberately cancelled after three rows, not scored as task failure.
The image's local ID is
`sha256:75126479f2116d15747e85fbc67f04d3a2f4fd522a3d7a185f3470f98479061c`
and its registered server-source SHA-256 is
`4ed1e7d2fcf26137b00f3a5d433bfe9fccb26b013d2dcdb16492de3afc44919f`.
Neither value is a published GHCR digest. Rendering still has H200 grain.

Final local adapter checks: **36 focused tests passed**, **2,605 full unit tests
passed, 3 skipped** (310.59 seconds), and all-files pre-commit passed. The combined
checkout's package/build-input/SIMPLE suite passed **57 tests**. Credential/private
path scans found no matches in changed delivery files or the exported context;
the image's native dependency/credential audit also passed.

A separate earlier native-base test used the historical external Astra/file bridge:
XMove executed one 50-row model-generated chunk and returned RGB/state; Tabletop
stopped before action execution. Both were intentionally stopped. That was not the
standard `EmbodiedAgent`, not the shared HTTP image, and not a complete task test.
It does not certify the example Agent or reproduce any paper score. The review
follow-up itself makes **zero model requests**.

## Sixth task restoration and service candidate (2026-10-15)

The pinned upstream checkout had deleted XMoveBendPick. The service now fetches
the unchanged task and MIT license from upstream commit
`c0a01cea95780a7eb56141686b7f25d38713e19a`, verifies the published hashes, and
restores its original registration outside the clean checkout. The task SHA256
is `51931cdd096ccd9953138cd52f4f1fe7cded70ea20990c827bf4339e948ae9d3`.
Its hardcoded reward >= 0.8 predicate is preserved. The earlier missing-environment
failure is resolved; no substitute environment or altered scoring was used.

Two subsequent native HTTP checks used `candidate07`, without model calls:

| Task | Physical H200 | Rows | RGBs | Downloaded artifacts | Elapsed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tabletop L0/E00 | 4 | 3 | 4 | 8 | 312.81 s |
| XMoveBendPick L0/E00 | 5 | 3 | 4 | 8 | 345.16 s |

Both retained the official row budget, verified reset/state/action transport,
cancelled the unfinished case, closed the worker, downloaded artifacts and
deleted the case. These are nonterminal checks, not successful task episodes.
Earlier `candidate03` checks covered XMovePick, BendPick, Handover and
PickBetweenTables. Together these cover all six families across candidate
versions; this is not a claim that all six were rerun on the final image.

The final `candidate08` build combines adapter commit
`0573b5a3a0c4c39e1ed7e425357997eb46db8c80` and registration proposal
`a793c8fc00f85503ee748ab28b0036fa4ff1d2f8`, after rebasing onto main
`519e15ac`. Its exported source SHA256 is
`bd14e521269bc2c9df01eefdad913700173737ab4ed374710ac2a80aee029867`.
Byte comparison against `candidate07` found only the shared source inventory
changed, incorporating main's DexJoCo registration; the native runtime files are
identical. This final image separately passed real HTTP source-identity checks,
discovery of 180 tasks and a writable `/tmp` probe under a read-only root.
The launcher supplies a 4 GB `/tmp` tmpfs for native scratch, with main Kit caches
and outputs on the writable output mount. No additional physics or model calls
were made during that metadata check.

After rebase, the full local unit suite passed **2,619 tests, 3 skipped** in
315.30 seconds; two upstream deprecation warnings remain. SIMPLE's focused suite
passed **39 tests** before rebase, the registration suite passed **41 tests**
after rebase, and all-files pre-commit passed. The final image verified its
exported source and scanned 56 source/build text files plus known credential
locations: no credential-pattern matches or prohibited Agent/model SDKs.
This supplements the native base's earlier audit; it is not a claim of a new
exhaustive scan of every binary layer. Both new catalogs export 180 cases,
each with verified reset bytes, for 1x and 10x respectively.

Publication identity and Registry checks are recorded in [DELIVERY.md](DELIVERY.md).
The newest #85 remains Draft, so draft-skipped GitHub jobs are not represented
as executed CI; the earlier executed adapter CI and local checks remain distinct.

## GHCR and Registry receipts (2026-10-15)

Published candidate `v0.1.0-cu128` to
`ghcr.io/opensensenova/sensenova-roborsi-simple` with registry digest
`sha256:c96dfdea288b31f8b56662f106a48c81c1457c3d815506019d45ef9f89cdefa3`.
The tag did not exist before publication. Push completed at 11:44:11 UTC;
registry manifest bytes matched the returned digest, and authenticated pull by
that digest passed on the build host. Temporary registry credentials were
removed after logout. The unpacked local size is 46,242,770,811 bytes.

The pulled image then passed the standard HTTP Registry/runtime identity check
with 180 tasks, a read-only root and writable temporary directory. No physics or
model request was needed for this check. The filesystem Registry manifest uses
the actual published reference, verifies both materialized asset trees and all
reset bytes, and reproduces all 180 catalog task messages exactly. Verification
took 11.65 seconds. This does not claim default Docker-executor native acceptance.

Code, recipes and docs are in the PR; generated manifests, PNGs, raw logs and
private filesystem paths are retained separately by the benchmark owner, not
committed to the source repository. These receipts establish build-host delivery;
an independent recipient-host pull/run remains pending.

## Native terminal/finish acceptance (2026-10-15)

The published `v0.1.0-cu128` image was exercised through its digest-pinned HTTP
service with `benchmarks.simple.check_terminal`. BendPick L0/E00 on physical
H200 GPU 5 completed its unchanged **400-row 1x budget** in **343.12 seconds**.
Each action held the initial joint/body targets with zero locomotion commands,
in blocks of at most 20 rows. No model, task policy or budget override was used.

The native response was `terminated=false`, `truncated=true`,
`termination_reason=native_step_limit`. The real `finish` call returned
`success=false`, `score=0.0`, `first_success_step=null`. The diagnostic verified
the downloaded result SHA256, all 400 contiguous native action records, absence
of success in that trace, the official budget and agreement with the returned
binary score. It downloaded **46 artifacts**, including 21 RGB observations,
the native result, reset provenance and actions. The HTTP case reached
`completed`, remained completed after close (the framework preserves that
outcome), and was deleted successfully. The owned container exited cleanly.

This establishes **native budget-exhaustion, failure scoring, finish, downloads
and cleanup** for this one AMO case. It does not establish native successful
manipulation scoring, every task's termination behavior or paper-score parity.
CPU HTTP tests separately cover both an early success and full-budget failure,
using real child-process/RPC/HTTP transport with scripted physics. The focused
SIMPLE suite now passes **41 tests**; the full local suite passed **2,622 tests,
3 skipped**, in 342.93 seconds, with two upstream deprecation warnings. All-files
pre-commit and the staged credential/private-path scan passed. This follow-up
made **zero model requests**.

The branch was rebased onto main `de243009` before validation. Comparing its
44 registered runtime files with the published image export found only the
common registration inventory differs (main added Bench2Dex; the image includes
the pending SIMPLE registration). No native execution, action or scoring file
changed. The new diagnostic is an external client, so the published image and
digest remain unchanged. Current-head GitHub CI receipts are recorded in PR #85;
its final Draft status must not be used to count later skipped runs as execution.

## Not executed / remaining gates

- No standard-Agent live-model task, native successful task or independent
  recipient-host validation has been performed. One fixed-action native failure
  episode now covers its full budget and finish/scoring. See DELIVERY.md for
  publication/Registry scope.
- Registration PR #86 is merged and this branch includes it. The published
  v0.1.0 image predates the adapter/facade retry fixes and must not be claimed
  as the fixed delivery. Use v0.1.1 and its matching source identity in DELIVERY.md.
- Current GitHub CI is blocked by account billing/spending limits before job
  execution. The latest commit still requires executed CI and maintainer review.
- The original working SIMPLE checkout had unrelated source changes and unresolved
  controller/mesh LFS resources. Host smoke used a separate clean checkout; the
  container fetched its pinned public sources and required LFS resources. The
  adapter rejects unresolved resources and source modifications before simulation.
- Official published scoring overrides, exact historical Agent request extensions,
  baseline training equivalence and complete six-task scoring remain unverified.
  `official_score` is false in saved native evidence.
- Native execution through the default Docker executor remains unverified: its
  1 GB `/tmp` is smaller than observed Isaac shader caches. Use the documented
  external HTTP service. The dependency lock covers HTTP packages, not a full
  native/system transitive artifact hash lock.

All tests are offline unless an operator explicitly enables the separate native
integration check and supplies a clean pinned installation/catalog. No API key,
filled environment file, experiment output, log, video, dataset or checkpoint is
part of this source PR.
