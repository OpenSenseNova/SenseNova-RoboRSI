# RoboDojo build and installation

[中文](BUILD.zh-CN.md) · [usage](README.md) · [source identity](PROVENANCE.md)

The benchmark image and Agent environment are independent. Native code, scene
assets, model checkpoints and Codex binaries are not vendored into this release.
The submitted skill/tool sources and their provenance are included.

## Benchmark image

A [public Native base image](https://github.com/orgs/OpenSenseNova/packages/container/package/sensenova-roborsi-robodojo)
is available as `ghcr.io/opensensenova/sensenova-roborsi-robodojo:v1.0.0-cu128-native`,
pinned by digest:

```text
ghcr.io/opensensenova/sensenova-roborsi-robodojo@sha256:e82c95f893834ebe16a65441a071a6b0b2fb6dbfe3b59d846fbc8cee7cb2bfda
```

This is the Native environment; build the current Benchmark service overlay
separately. The base's native Python is `/root/miniconda3/envs/RoboDojo/bin/python`.
Select it through the base's `PATH` for build and runtime commands: the overlay
clears the inherited entrypoint, so entrypoint-only Conda activation is insufficient.

Prepare the [official RoboDojo native stack](https://github.com/RoboDojo-Benchmark/RoboDojo)
using its upstream installation instructions at the revision in
`robots/robodojo/native_files.json` (default
`726e9aabfaa642203722eb126f5eaf0f37f3e1ad`). The source inventory also lists
explicitly supported revision profiles. Use the same profile for catalog export
and evaluation; set `ROBORSI_ROBODOJO_NATIVE_COMMIT` before starting either process.
Install its matching Isaac Sim, IsaacLab, CUDA PyTorch and cuRobo dependencies
using the upstream native environment. These dependencies are imported only by
the native owner. Put the verified checkout at `/workspace/RoboDojo`, and link
its `Assets` directory to `/roborsi/assets/robodojo` as expected by that checkout's
configuration. Retain the original native task, controller and scorer files.

Seal that environment as a base image, record its actual digest, and export the
server-only context:

```bash
python -m benchmarks.package --benchmark robodojo --output /data/robodojo-context
docker build /data/robodojo-context \
  --build-arg ROBODOJO_BASE_IMAGE=YOUR_NATIVE_IMAGE@sha256:ACTUAL_DIGEST \
  -t YOUR_REGISTRY/roborsi-robodojo:local
```

For another supported profile, also pass
`--build-arg ROBODOJO_NATIVE_COMMIT=SUPPORTED_COMMIT` to `docker build` and use
the same `ROBORSI_ROBODOJO_NATIVE_COMMIT` for catalog export. At runtime the
case-sensitive native `Assets` path must resolve to the advertised snapshot;
the service checks that binding before opening a case.

The overlay verifies the pinned native file inventory and rejects Agent SDKs or
undeclared source in the image. It contains no policy, prompts, skills, model
credentials or weights. The base is an explicit input: this release does not
claim to build Isaac/cuRobo from a generic Python image. Record the final image
digest after a successful build and use that digest in the generated manifest.
The service image has its own digest, separate from the public Native base above.

## Task assets and service

Run in the prepared native environment from the repository root. Snapshot the
official assets, then use the snapshot path printed by the first command:

```bash
python -m benchmarks.robodojo.prepare assets \
  --source /data/RoboDojo-assets --output-dir /data/registry/robodojo
export ROBORSI_ROBODOJO_NATIVE_ROOT=/workspace/RoboDojo
export ROBORSI_ROBODOJO_ASSETS=/data/registry/robodojo/snapshots/ACTUAL_SNAPSHOT_SHA256
```

Configure `/workspace/RoboDojo/Assets` to resolve to that snapshot. Export the
task catalog and start the service:

```bash
python -m benchmarks.robodojo.prepare catalog --all-tasks --layout 0 --eval-seed 0 \
  --output /data/registry/robodojo/catalog.json --work-dir /data/robodojo-inspection
export ROBORSI_ROBODOJO_CATALOG=/data/registry/robodojo/catalog.json
python -m roborsi.evaluation.service_host --kind benchmark \
  --factory robots.robodojo.benchmark_service:make_robodojo_benchmark \
  --host 127.0.0.1 --port 18800
```

## Agent

Use Linux x86-64 and the submitted policy's Python 3.10.12 environment. From this
repository, with that interpreter selected:

```bash
python3.10 -m venv .venv-robodojo
source .venv-robodojo/bin/activate
python -m pip install uv==0.8.9
python -m pip install --require-hashes \
  -r robots/robodojo/agents/submission/requirements.lock
python -m pip install --no-deps --no-build-isolation -e .
python -m benchmarks.robodojo.install_agent --vla-backend simate_beta
```

The installer checks protected resources, authenticates the fixed Codex 0.160.1
artifact and companion binaries, builds the original separate tool environment,
and installs the selected VLA. It makes no model API request. Downloads can be
large. Checkpoint/source/tokenizer hashes and dependency locks stay in the
submission's `pi05/`, `simate/`, `tools/` and `vendor/` directories. Explicit
`--checkpoint-dir` selects weights outside the source tree; pass the same absolute
path with `--agent-set parameters.checkpoint_path='"/data/checkpoint"'` at evaluation.

The Simate installer prepares its model environment and checkpoint. If using
an external checkpoint, select the same path for installation and evaluation.

## Verification

```bash
python -m benchmarks.robodojo.provenance
pytest tests/unit_tests/robots/robodojo -q
python -m benchmarks.robodojo.prepare --help
python -m benchmarks.robodojo.install_agent --help
```

Offline tests establish protocol/resource contracts. Native acceptance still
requires a verified image, licensed assets, supported GPU/driver and model
service, followed by actual catalog and task execution.
