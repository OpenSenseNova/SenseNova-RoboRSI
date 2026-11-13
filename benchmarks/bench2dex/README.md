# Bench2Dex benchmark

[简体中文](README.zh-CN.md)

This integration serves 26 pinned Bench2Dex tasks through RoboRSI's common
benchmark HTTP interface. It preserves native actions, camera calibration,
control budgets, sample-index mapping, stopping, scoring and cleanup. Agent
policies are supplied separately through the standard evaluation interface.

Native execution requires Linux x86-64, a supported NVIDIA RTX GPU, its driver,
Docker with GPU support, and access to the pinned Isaac Sim image and upstream
repositories. The image uses Isaac Sim, Isaac Lab and Bench2Dex; their licenses
and access requirements apply independently. Assets are downloaded separately;
installing RoboRSI does not provide the native runtime or assets.
The custom reference harness is licensed under
[Apache-2.0](../../robots/bench2dex/reference/LICENSE_NOTICE.md).
This migration does not establish native task success or model performance.

## Prepare external assets

Run these commands from the repository root after installing the framework.
Install `huggingface_hub` for the asset download command.

```bash
python -m pip install huggingface_hub
python -m benchmarks.bench2dex.assets --download \
  --source work/bench2dex/download --output work/bench2dex/registry/bench2dex/assets
python -m benchmarks.bench2dex.prepare catalog \
  --assets work/bench2dex/registry/bench2dex/assets --sample 5 \
  --output work/bench2dex/registry/bench2dex/catalog.json
```

The asset helper selects files from the official `Bench2Dex/Assets` dataset at
revision `bf65215f844d1fca30750e4cc70d7266650abf0a`. It verifies Git/LFS object
identities before copying, checks each copied file, and writes an adjacent
inventory and directory checksum. Omit `--download` to verify an existing
download. Keep the sealed assets and catalog read-only during evaluation.

Repeat `--sample` to select additional native episode indices and `--task` to
select task IDs from `robots/bench2dex/tasks.json`. `--seed` in evaluation maps
to this episode index; the native RNG seeds remain fixed by `protocol.json`.
A catalog with `--smoke-steps 4` is a diagnostic run and is explicitly ineligible
for scored evaluation.

## Build and register

Use the published `v1.0.4-cu128` image with this source checkout. Its benchmark
runtime matches commit `b513a1c7289ab11eb844856588b64c378705c405`. After preparing
the assets and catalog above, register it by its immutable digest:

```bash
export ROBORSI_BENCH2DEX_IMAGE=ghcr.io/opensensenova/sensenova-roborsi-bench2dex@sha256:edb0701c6ea3904c53a64820775336b998e6a3d0a9ac8b3b5be2caf824db3a1b
python -m benchmarks.bench2dex.prepare manifest \
  --image "$ROBORSI_BENCH2DEX_IMAGE" --assets work/bench2dex/registry/bench2dex/assets \
  --catalog work/bench2dex/registry/bench2dex/catalog.json \
  --output work/bench2dex/registry/bench2dex/manifest.json
```

To build and publish an image from a changed checkout, use your own registry:

```bash
python -m benchmarks.package --benchmark bench2dex \
  --output work/bench2dex/build
python -m benchmarks.bench2dex.build \
  --context work/bench2dex/build --tag YOUR_REGISTRY/bench2dex:BUILD \
  --evidence work/bench2dex/build-evidence --publish
export ROBORSI_BENCH2DEX_IMAGE=$(docker image inspect YOUR_REGISTRY/bench2dex:BUILD \
  --format '{{index .RepoDigests 0}}')
python -m benchmarks.bench2dex.prepare manifest \
  --image "$ROBORSI_BENCH2DEX_IMAGE" --assets work/bench2dex/registry/bench2dex/assets \
  --catalog work/bench2dex/registry/bench2dex/catalog.json \
  --output work/bench2dex/registry/bench2dex/manifest.json
```

Replace `YOUR_REGISTRY` and `BUILD` with your registry and tag. The Dockerfile
pins the simulator base and native source revisions. The exported context
contains the declared benchmark dependency closure and build inputs. The build
helper checks dependencies and source imports, and requires the image's source
digest to match the exported context before publication. It then pulls the
published image by digest and repeats these checks. The verified source digest
and adapter version are recorded in `build-evidence/image.json`.
Omit `--publish` for a local build;
the manifest requires a repository digest. These checks do not execute a GPU
episode.

When using a prebuilt image, generate the manifest from the matching source
checkout. Changing the Bench2Dex adapter requires rebuilding and publishing the
image, updating its digest reference, and regenerating the manifest. A stale
image is rejected with `runtime adapter_version differs from Benchmark manifest`;
do not override the manifest's adapter version or disable runtime verification.

Place the assets and catalog beneath the manifest directory; the generator
records relative paths required by the Registry and rejects external paths or
symlinks within those paths. The registry directory itself may be reached through
a symlink. For an existing layout, move the assets and catalog beneath that
directory before regenerating the manifest. Keep manifests,
downloaded data, credentials and generated evidence outside published source.

## Evaluate with an Agent

Configure your model and credentials using the [Agent guide](../../docs/agents.md),
then use a standard preset or your own Agent:

```bash
bench eval --registry-root work/bench2dex/registry \
  --benchmark bench2dex --task bench2dex:26 --seed 5 \
  --agent-config configs/agents/embodied/default.yaml \
  --results-root work/bench2dex/results --run-id bench2dex-1 \
  --timeout-s 900 --case-timeout-s 86400 --max-concurrent 1
```

The preset is a generic interface example and does not reproduce historical
scores. The original task ICL Agent, package staging and construction sources
are available separately: see [frozen ICL](ICL.md) and
[learning from demonstrations](../../docs/bench2dex-icl.md). These retain their
original model, CLI and package contracts. The fixed-action
LIBERO example in [Getting started](../../docs/getting-started.md) applies only
to LIBERO. Set the model, context and decision budgets for your run.

The service exposes Bench2Dex's native action schema and scoring. Actions
contain finite absolute joint-position targets: `{"actions": [[...], ...]}` with one value per
active joint. Use `{"continue": true}` while a previously submitted chunk still
has actions. RGB images and calibrated robot information are policy-visible;
native success and detailed scoring are returned only when finishing a case.

For a separately managed native environment, configure `BENCH2DEX_ROOT`,
`ISAACLAB_ROOT`, `ISAACLAB_PYTHON`, `BENCH2DEX_ASSETS` and `BENCH2DEX_CATALOG`.
The `dex2bench_dataset` sibling link beside the native Bench2Dex checkout must
point to the sealed assets. Start the common service:

```bash
python -m roborsi.evaluation.service_host --kind benchmark \
  --factory robots.bench2dex.benchmark_service:make_bench2dex_benchmark \
  --host 127.0.0.1 --port 8000
```

Then replace `--registry-root` with
`--benchmark-url bench2dex=http://127.0.0.1:8000` in the evaluation command.
See [Evaluation](../../docs/evaluation.md) for saved results, status and resume.
Offline tests exercise synthetic native providers, HTTP transport, build
boundaries and retryable cleanup; they do not validate this simulator image on
a GPU.
