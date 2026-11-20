# SIMPLE native evaluation integration candidate

[中文](README.zh-CN.md) · [build inputs](BUILD.md) · [validation](VALIDATION.md)

## Migration into SenseNova-RoboRSI — 2026-11-18

The SIMPLE benchmark sources, recipes, Agent preset and tests were copied from
the local `NewRPent` checkout onto the `SIMPLE` branch. The standard exporter
accepts `--benchmark simple`. The image digests, PR status and native validation
below are imported records from that checkout, not verification of an image
built with this repository's shared framework. Rebuild and verify the paired
image/source digests before deploying this checkout; see [VALIDATION.md](VALIDATION.md).

This branch adds a SIMPLE Provider/Adapter, a main-thread native worker, private
reset catalog export, and a standard external `EmbodiedAgent` API preset. It does
not claim reproduction of the historical Astra scores or full task acceptance.

## Status and dependencies

- Framework source registration [PR #86](https://github.com/SenseTime-FVG/RoboRSI/pull/86)
  merged as `2d06bcf1417f779f68611ab3671cd0e13ee63056`. This branch includes it
  and the shared facade cleanup retry fix (#88) from main. Standard source
  export and service discovery are available; see [FRAMEWORK_REQUEST.md](FRAMEWORK_REQUEST.md).
- Published repair candidate: `ghcr.io/opensensenova/sensenova-roborsi-simple:v0.1.1-cu128`. The verified registry reference is
  `ghcr.io/opensensenova/sensenova-roborsi-simple@sha256:4b6f9eb28fa87aebba7a3d46fcf0498753d81a47719dadf30a78c22d1f208213`.
  It includes both adapter and facade cleanup retry fixes. Authenticated push,
  digest-pinned pull and runtime verification passed on the build host. See
  [DELIVERY.md](DELIVERY.md) for commands, source identity and validation limits;
  recipient-host and successful-task scoring acceptance remain pending. Historical
  v0.1.0 lacks these repairs. Current GitHub CI is blocked before job execution by
  account billing/spending limits; the PR remains Draft.
- A separate [native base recipe](native/README.md) installs the pinned simulator
  and controller stack without Agent SDKs or credentials. Catalogs, assets and
  historical ICL references remain external artifacts.
- The example API Agent is a new scaffold. It does not include historical frozen
  ICL packages, the archived numerical/FK/IK sandbox, or the original max/pro
  request extension. It is not a replacement for archived paper experiments.
- This work changes no shared framework source or global dependencies.

## Candidate delivery workflow

See [DELIVERY.md](DELIVERY.md) for the candidate image, complete launch commands,
Registry verification, and the scope of the short checks. Prepare scene/robot
assets using [ASSETS.md](ASSETS.md). The CLI `benchmarks.simple.serve` runs the
shared HTTP service with one explicit GPU and a writable output/cache directory.
The optional `benchmarks.simple.check_terminal` client holds a fixed action until
native termination or the original 1x budget, then checks finish/scoring/download
and cleanup without a model. It is separate from the three-row smoke and does
not evaluate an Agent policy; commands and evidence are in DELIVERY/VALIDATION.

`xmove_bendpick` requires the unmodified original upstream task from commit
`c0a01cea95780a7eb56141686b7f25d38713e19a`, removed by upstream commit
`eef474fe0676057efc25c40d0a2dc33f1afb41e6`. The service build downloads that one
file and its MIT license with SHA256 checks. The worker loads it outside the
clean native checkout and restores the original environment registration.
The original reward/check-success code, geometry, budget and instructions are
not rewritten. Its original `check_success` uses reward >= 0.8, even though the
historical worker also assigns a `success_criteria` attribute. Compatibility
source identity is retained in each native reset record.

## Native conditions

SIMPLE commit `6d10628794d9c7de4596b4f2afb2c054a637c2bc`; data revision
`e78fb93cc28912a3031a10b8656d32d7f0a2b867`, repository
`USC-PSI-Lab/psi-data`. Exact archive bytes and SHA256 are in
`robots/simple/sources.json`. Catalog export does not execute downloaded code.

| Alias | 1× rows | Controller |
| --- | ---: | --- |
| xmove | 800 | decoupled WBC |
| bendpick | 400 | AMO |
| handover | 800 | decoupled WBC |
| tabletop | 800 | AMO |
| mobile_pick | 1200 | decoupled WBC |
| xmove_bendpick | 800 | decoupled WBC |

Each task has all ten released cases in each of levels 0/1/2. Explicit 10× catalogs
use ten times the corresponding row budget. Native action rows execute at nominal
50 Hz. HTTP step indices count submitted chunks; `native_env_steps` counts executed
rows. A chunk is at most 512 rows to fit the shared HTTP request boundary and may
never exceed the remaining physical budget. Nothing is clipped or silently dropped;
native termination can stop a chunk early.

Scoring deliberately declares the historical worker condition: AMO 0.9, non-mobile
WBC 0.5, mobile's task metadata. These are **not certified official leaderboard
CLI overrides**. Native `task.check_success` owns scoring (including the restored task’s hardcoded threshold); neither model finish nor
a synthetic hold-action example implies success. Physics pauses between requests;
WBC's existing wall-clock controller behavior is unchanged. Preparation, Robot
resources, hand profiles and real G1/Wuji binary control are separate experiments.

## Install independent environments

Use Python 3.10–3.12 for RoboRSI, and a separate native Python 3.10 environment for
the Isaac 4.5 reference stack. Follow `CONTRIBUTING.md` for the control/Agent venv.
Install SIMPLE and its submodules per upstream instructions in the native venv;
materialize required LFS checkpoints and G1 meshes. The native checkout must match
the pinned commit and have no tracked changes under `src/simple`; unsupported local
patches require a separately validated profile. This candidate does not install or
modify drivers, graphics capabilities, Isaac, global packages or upstream source.
Install and initialize Git LFS in that checkout (`git lfs install --local`);
otherwise materialized checkpoints can appear as source modifications. The native
stack also needs `libGLU.so.1`, `libvulkan.so.1` and working NVIDIA graphics libraries.
Set graphics library paths on the host if they are supplied outside system paths.

Download the selected test archive with its exact revision, for example:

```bash
hf download USC-PSI-Lab/psi-data \
  simple-eval/G1WholebodyTabletopGraspMP-v0.zip --repo-type dataset \
  --revision e78fb93cc28912a3031a10b8656d32d7f0a2b867 --local-dir /data/psi-data
python -m benchmarks.simple.prepare --archives /data/psi-data \
  --tasks tabletop --budget-multiplier 1 --output /data/simple-catalog-1x
```

The output must be new. The exporter validates the pinned ZIP, preserves all
30 cases, applies the historical scene UID conversion and hashes every reset.
Reset geometry stays in the Benchmark process; public tasks carry only identities
and a reset checksum. Seed base `20261002` is an explicit reset RNG condition,
not a claim that every historical run used that same seed or hardware.

The Agent instruction is the archive's `episodes.jsonl` task text. Some released
resets contain a different language template (XMove L0/E00 says "bend" in the
reset and "move forward" in the episode text). Native initialization validates
its generated instruction against the reset template and target name, and records
both texts plus `instruction_match` in private reset evidence. Neither text nor
the reset is silently rewritten. This prompt choice is not certified equivalent
to the upstream evaluation CLI, which reads the native generated instruction.

## Serve and select an Agent

With the merged framework registration included in this branch:

```bash
export ROBORSI_SIMPLE_CATALOG=/data/simple-catalog-1x/catalog.json
export ROBORSI_SIMPLE_ROOT=/data/SIMPLE
export ROBORSI_SIMPLE_PYTHON=/data/simple-native-venv/bin/python
export ROBORSI_SIMPLE_ASSETS=/data/simple-assets
export ROBORSI_SIMPLE_GPU_INDEX=0
# Set OMNI_KIT_ACCEPT_EULA=Y only after accepting NVIDIA's EULA.
python -m roborsi.evaluation.service_host --kind benchmark \
  --factory robots.simple.benchmark_service:make_simple_benchmark \
  --host 127.0.0.1 --port 8000
```

The native interpreter must have the Benchmark-only dependencies; use the same
installed service source via `PYTHONPATH`, not a model/Agent environment. Allocate
one GPU and one active native case per service. `ROBORSI_SIMPLE_GPU_INDEX` selects
the device for both Torch and Isaac; their device numbering must agree. Do not
remap `CUDA_VISIBLE_DEVICES` independently of Isaac's graphics devices. Set
`MUJOCO_EGL_DEVICE_ID` consistently when using EGL. Kit caches and logs are kept
inside each case directory, and AMO's traced CUDA allocation constants are moved
in memory to the selected GPU without changing checkpoint bytes. Use a unique
`ROBORSI_SIMPLE_DDS_DOMAIN` for each concurrent native allocation. Headless still
requires the host graphics libraries/devices. Current graphics driver support must
be validated on the destination; prior H200 diagnostics do not certify this adapter.

The public API is the existing RoboRSI Benchmark HTTP API. Use the existing Registry
and Runner deployment instructions after an actual image/service acceptance run;
this PR does not fabricate a manifest/image digest for an unverified deployment.
Use the standard preset at `robots/simple/agent.yaml`, configure the endpoint/model
and credentials only in the external Agent environment, and retain full traces.
The example preset declares a 1000 model-turn cap, no per-response token cap and
no planner timeout. This is separate from native robot-row budgets. Raw credentials
are not inherited by the simulator worker. Do not put filled envs, dialogs, videos,
resets, weights, generated manifests or results into the source PR.

Export/install the Agent with the existing `roborsi.agents.package` command. The
preset declares its prompt/config only; it must not carry the simulator or private
catalog. Frozen ICL references and exact historical Agent migration are future
work, with their own hashes and validation.

## Checks

For a running service with registered sources, this client makes no model request
and stops after at most three native action rows:

```bash
python -m benchmarks.simple.smoke_http --endpoint http://127.0.0.1:8000 \
  --task-id tabletop/L0/E00 --output /data/new-simple-http-smoke \
  --source-sha256 SHA256_FROM_EXPORTED_SOURCE_JSON
```

Run it in the external client environment. The output must be a new directory.
The source checksum must match the build's `source.json`, not an image digest.
It downloads RGB/state evidence and closes/deletes only its own case. An early
stop is not scored as task failure; `finish` requires genuine native termination.
No complete-task run or paid model call is needed for this short interface check.

```bash
pytest tests/unit_tests/robots/simple -v
pre-commit run --all-files
pytest tests/unit_tests -v
```

See `VALIDATION.md` for checks actually executed and the current native/image gaps.
All model inputs are RGB, the original 32-value state and budget counters. Reward,
object poses, reset geometry and native success are withheld until termination.
The current scaffold returns the latest image at each chunk boundary; it does not
recreate the archived file-tool history or claim historical evaluation equivalence.

To check native rendering and controller wiring without model calls, configure the
native variables above, install `.[test]` in the control environment, then run:

```bash
export ROBORSI_RUN_SIMPLE_NATIVE=1
export ROBORSI_SIMPLE_EVIDENCE_DIR=/data/new-simple-native-smoke
pytest tests/integration_tests/robots/simple/test_native.py -v
```

The evidence directory must be new. This executes at most three native rows for
the first catalog case and preserves its PNGs and worker/Kit logs. It bypasses
neither reset validation nor action validation, and does not require framework
source registration because it checks the Adapter directly. It is not a task
success rate, a model test, or validation of all six tasks.
