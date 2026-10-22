# Deploy LIBERO-PRO, Long++ and Ultra-Long

[简体中文](README.zh-CN.md)

The three benchmark services retain native simulation and scoring. Specialized
Agents bind a verified ICL package to each task, preserve one dialogue throughout
the episode and use a locked tokenizer for context management. The combined
image pairs the Agent with LiLo or PRO for schedulers that accept one image per
job. Long++ and Ultra-Long share the LiLo composition, with separate Agent and
native Python package directories inside the image.

Use Linux x86_64, Docker, an NVIDIA GPU/driver and the NVIDIA Container Toolkit
for EGL rendering. Source builds and the host Runner use Python 3.11. Allow
space for native images, the Agent environment, build contexts and results.
The model runs through an OpenAI Responses-compatible endpoint.

Set `RUN_ROOT` to an absolute output directory and `MODEL_API_KEY_FILE` to an
existing private key file. Export `OPENAI_BASE_URL` for your endpoint. Each
single-case output directory must be new.

## Run the published combined images

These immutable `v1.0.1-egl` artifacts were published by
[upstream PR 93](https://github.com/SenseTime-FVG/RoboRSI/pull/93) from commit
`5b8c7f7aa6e1b67119ef3e887625601387ce917e`. They retain that revision's embedded
presets, model routing and directory layout. Use an endpoint compatible with
those embedded presets, or follow the rebuild steps below to select your model.
The combined CLI accepts an endpoint and key file; model selection is configured
in its installed bundles.

```bash
export LONGPP_COMBINED_IMAGE='ghcr.io/opensensenova/sensenova-roborsi-libero-long-plus-plus@sha256:814139261e914b114cb89a359c6a39d4f409a556f79928e986b6ade3dfe7dc7a'
export ULTRA_COMBINED_IMAGE='ghcr.io/opensensenova/sensenova-roborsi-libero-ultra-long@sha256:814139261e914b114cb89a359c6a39d4f409a556f79928e986b6ade3dfe7dc7a'
export PRO_COMBINED_IMAGE='ghcr.io/opensensenova/sensenova-roborsi-libero-pro@sha256:a83417f1459a4c555c457caa82ef00a5de50191e3d8deb943894fa9285cff172'
docker pull "$LONGPP_COMBINED_IMAGE"
docker pull "$ULTRA_COMBINED_IMAGE"
docker pull "$PRO_COMBINED_IMAGE"
```

Long++ and Ultra-Long use separate GHCR package names with identical LiLo image
content. Each image includes its Benchmark, all three Agent bundles, sealed
ICL packages and tokenizer. Model credentials are supplied at runtime.

Run one Long++ case:

```bash
mkdir -p "$RUN_ROOT/results"
docker run --rm --gpus all \
  --mount "type=bind,src=$MODEL_API_KEY_FILE,dst=/run/secrets/model_api.key,readonly" \
  --mount "type=bind,src=$RUN_ROOT/results,dst=/results" \
  "$LONGPP_COMBINED_IMAGE" \
  --benchmark libero_long_plus_plus --task t00-i01 \
  --run-id longpp-one --results-root /results/longpp-one \
  --api-key-file /run/secrets/model_api.key --base-url "$OPENAI_BASE_URL"
```

For Ultra-Long use `$ULTRA_COMBINED_IMAGE`, `--benchmark libero_ultra_long` and
`--task t00-i01`. For PRO use `$PRO_COMBINED_IMAGE`, `--benchmark libero_pro` and
`--task libero_spatial_task-t00-i03`.

Run the fixed subset with the subset runner. This command uses the original
script path embedded in the published images above:

```bash
docker run --rm --gpus all --entrypoint python \
  --mount "type=bind,src=$MODEL_API_KEY_FILE,dst=/run/secrets/model_api.key,readonly" \
  --mount "type=bind,src=$RUN_ROOT/results,dst=/results" \
  "$LONGPP_COMBINED_IMAGE" \
  -B /opt/roborsi/scripts/libero_three/run_combined_subset.py \
  --benchmark libero_long_plus_plus --run-id longpp-subset \
  --results-root /results/longpp-subset --max-concurrent 16 --gpu-count 1 \
  --api-key-file /run/secrets/model_api.key --base-url "$OPENAI_BASE_URL"
```

The equivalent Ultra-Long and PRO commands select their image and benchmark;
the maps plan 12, 9 and 80 cases. Repeated `--task` selects fewer cases.
Concurrency defaults to 16 with one shared GPU; choose capacity for your host.
The runner preserves each attempt. Reuse requires a matching image/source,
Agent, ICL, endpoint and timeout fingerprint, zero process exit, matching case
identity and an intact native score digest. It exits nonzero while any case
lacks a verified native terminal. Its service uses the fixed subset at 1×.

The combined image shares a container between Agent and Benchmark. Independent
containers below give each case a separate Agent filesystem and network.

## Evaluation contracts

| Benchmark | Official set | Fixed subset | Official 1× budget | Primary score |
| --- | --- | --- | --- | --- |
| LIBERO-PRO Task/Swap | 8 suites × 10 tasks × 10 states = 800 | 80 tasks, state 3 | 10,000 physical steps | Final-goal success |
| LIBERO-Long++ | 12 configurations × 10 bare-reset trials = 120 | 12 configurations, state 1 | 200 × ordered skills | Ordered-prefix fraction; all skills for success |
| LIBERO-Ultra-Long | 9 configurations × 10 bare-reset trials = 90 | 9 configurations, state 1 | 200 × ordered skills | Ordered-prefix fraction; all skills for success |

Long++ and Ultra-Long trial labels are not random seeds. The official profile
repeats the released bare random reset; `migration_subset` explicitly uses
fixed state 1. PRO's subset uses fixed state 3. The supplied ICL maps cover
these 12/9/80 subset cases. Full-set evaluation needs an Agent whose task
coverage matches the chosen manifest.

An Agent may submit 7-DoF Panda `OSC_POSE` actions or the declared feedback
primitives, including `move_to`, `move_pose`, wrist rotation and gripper control.
The Benchmark compiles primitives to physical actions, and every physical step
uses the same budget. `rotate_wrist` rotates in place; its semantics differ
from `move_pose.target_yaw`.

For the ordered benchmarks, `lilo-instance-pose-feedback-v1` supplies registered
instance names, live world positions, collision-geometry heights, exact gripper
contacts, held objects and current skill progress. Reading it does not advance
the scorer. `libero-robot-state-feedback-v1` exposes `robot0_eef_pos`, xyzw
quaternion and `robot0_gripper_qpos`, alongside the compact eight-dimensional
`states` input. These feedback channels are part of the named evaluation
condition and must be retained when comparing results.

`plus9_on_failure` defaults to false. When enabled, a case that fails at 1× may
continue in the same environment and dialogue up to 10× total steps.
`result.json` and aggregates keep the immutable 1× score;
`native/score.json` additionally records cumulative recovery and physical and
decision step counts. Ordered scoring follows the uninterrupted released skill
order; it does not substitute final-goal success.

## Agent presets and inputs

| Preset under `configs/agents/libero/` | Installed bundle | ICL packages |
| --- | --- | ---: |
| `long_plus_plus.yaml` | `bundle:libero_long_plus_plus_v13` | 12 |
| `ultra_long.yaml` | `bundle:libero_ultra_long_v13` | 9 |
| `pro.yaml` | `bundle:libero_pro` | 40 shared across 80 cases |

`base.yaml` and `v13.yaml` supply shared settings. All three leaf presets use
max reasoning and the token-accurate context policy: 1,050,000-token window,
128,000-token output reserve, 32,768-token overhead reserve, 0.75 summary
trigger, eight complete recent exchanges and two image groups. History remains
complete below the trigger. PRO permits 100 model turns, a 5,000-second planner
budget and no model retries; the ordered presets permit 10,000 turns, a
43,200-second planner budget and three bounded transport retries.

The current preset uses model ID `gpt-6-astra`. Select an ID supported by your
endpoint in the preset or with `roborsi.agents.package --set`; an independent
Docker Agent also accepts `bench eval --agent-set`. Tokenizer paths must point
to the verified in-image file when exporting a portable bundle.

The [asset lock](assets.lock.json) identifies the 61 sealed ICL packages and
tokenizer. The [context maps](../../configs/agents/libero/context_maps/)
bind tasks to packages.
`prepare_agent_assets.py` verifies inventories and hashes before copying them;
`verify_agent_image.py` repeats the checks against installed bundles during the
Agent image build. The [rebuild procedure](#rebuild-the-agent-from-public-inputs) obtains
these inputs from the published GHCR image.

## Rebuild the Agent from public inputs

Run from this checkout. Set `LIBERO_BUILD_ROOT` to a directory outside the
checkout. Its `extracted-assets`, bundle and build subdirectories must be new;
use a fresh `dist/agent/` for this build. Set `AGENT_BASE_TAG` to a new local
Agent base image tag. Set `AGENT_IMAGE_TAG`, `LILO_IMAGE_TAG` and `PRO_IMAGE_TAG`
to new tags in your container registry and authenticate before publishing.

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[runtime]' build uv tokenizers==0.22.2
mkdir -p "$LIBERO_BUILD_ROOT"
```

Pull the Long++ combined image above, then copy its assets from a stopped
container. This does not run the simulator or Agent. Verify and stage the
three package families and tokenizer with the current asset lock:

```bash
ASSET_CONTAINER=$(docker create "$LONGPP_COMBINED_IMAGE")
docker cp "$ASSET_CONTAINER:/opt/roborsi-agent-assets" "$LIBERO_BUILD_ROOT/extracted-assets"
docker rm "$ASSET_CONTAINER"
python benchmarks/libero_three_combined/prepare_agent_assets.py \
  --longpp-root "$LIBERO_BUILD_ROOT/extracted-assets/longpp/packages" \
  --ultra-root "$LIBERO_BUILD_ROOT/extracted-assets/ultra/packages" \
  --pro-root "$LIBERO_BUILD_ROOT/extracted-assets/pro/packages" \
  --tokenizer "$LIBERO_BUILD_ROOT/extracted-assets/tokenizer.json" \
  --output dist/agent/libero-three-assets
```

The inventory has 12 Long++ packages, 9 Ultra-Long packages and 40 PRO packages
covering 80 cases. The preparer rejects changed or extra package entries.

Export the three bundles with the in-image tokenizer path and hash. Replace
`gpt-6-astra` consistently if your endpoint uses another model ID:

```bash
python -m roborsi.agents.package \
  --config configs/agents/libero/long_plus_plus.yaml \
  --name libero_long_plus_plus_v13 \
  --set 'runtime.llm.model="gpt-6-astra"' \
  --set 'runtime.context.tokenizer_file="/opt/roborsi-agent-assets/tokenizer.json"' \
  --set 'runtime.context.tokenizer_sha256="0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"' \
  --output "$LIBERO_BUILD_ROOT/bundle-longpp"
python -m roborsi.agents.package \
  --config configs/agents/libero/ultra_long.yaml \
  --name libero_ultra_long_v13 \
  --set 'runtime.llm.model="gpt-6-astra"' \
  --set 'runtime.context.tokenizer_file="/opt/roborsi-agent-assets/tokenizer.json"' \
  --set 'runtime.context.tokenizer_sha256="0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"' \
  --output "$LIBERO_BUILD_ROOT/bundle-ultra"
python -m roborsi.agents.package \
  --config configs/agents/libero/pro.yaml --name libero_pro \
  --set 'runtime.llm.model="gpt-6-astra"' \
  --set 'runtime.context.tokenizer_file="/opt/roborsi-agent-assets/tokenizer.json"' \
  --set 'runtime.context.tokenizer_sha256="0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"' \
  --output "$LIBERO_BUILD_ROOT/bundle-pro"
python -m build --wheel --outdir dist/agent "$LIBERO_BUILD_ROOT/bundle-longpp"
python -m build --wheel --outdir dist/agent "$LIBERO_BUILD_ROOT/bundle-ultra"
python -m build --wheel --outdir dist/agent "$LIBERO_BUILD_ROOT/bundle-pro"
```

Resolve all three wheel requirements together with the Agent base requirements.
`tokenizers==0.22.2` is declared by the presets. SciPy is also included because
the combined image checks `scipy.spatial.transform.Rotation`. Version 1.15.3
aligns with the native base; the generated lock fixes all versions and hashes.
The local framework wheel supplies metadata for this checkout, and the Dockerfile
installs that checkout's source.

```bash
python -m build --wheel --outdir "$LIBERO_BUILD_ROOT/framework-wheel" .
python -c 'from pathlib import Path; import sys; print("roborsi @ " + next(Path(sys.argv[1]).glob("roborsi-*.whl")).resolve().as_uri())' \
  "$LIBERO_BUILD_ROOT/framework-wheel" > "$LIBERO_BUILD_ROOT/roborsi-override.txt"
printf 'scipy==1.15.3\n' > "$LIBERO_BUILD_ROOT/combined-runtime.in"
uv pip compile \
  "$LIBERO_BUILD_ROOT/bundle-longpp/requirements.in" \
  "$LIBERO_BUILD_ROOT/bundle-ultra/requirements.in" \
  "$LIBERO_BUILD_ROOT/bundle-pro/requirements.in" \
  agents/python/requirements.in "$LIBERO_BUILD_ROOT/combined-runtime.in" \
  --overrides "$LIBERO_BUILD_ROOT/roborsi-override.txt" --no-emit-package roborsi \
  --python-version 3.11 --generate-hashes --output-file dist/agent/requirements.txt
docker build --target configured -f agents/python/Dockerfile \
  --build-arg AGENT_REQUIREMENTS=dist/agent/requirements.txt \
  --build-arg 'AGENT_WHEEL=dist/agent/roborsi_agent_bundle_libero_*.whl' \
  -t "$AGENT_BASE_TAG" .
docker build -f benchmarks/libero_three_combined/Agent.Dockerfile \
  --build-arg AGENT_BASE_IMAGE="$AGENT_BASE_TAG" \
  --build-arg LIBERO_THREE_ASSETS=dist/agent/libero-three-assets \
  -t "$AGENT_IMAGE_TAG" .
docker push "$AGENT_IMAGE_TAG"
export AGENT_IMAGE=$(docker image inspect "$AGENT_IMAGE_TAG" --format '{{index .RepoDigests 0}}')
```

The image build verifies bundle task maps, all package hashes, inventories and
tokenizer identity, then runs dependency checks. Preserve the generated lock,
bundle manifests, source revision and actual repository digest.

## Rebuild and register the Benchmarks

Prepare the complete PRO evaluation tree as described in the
[PRO guide](../libero_pro/README.md), and set `PRO_EVALUATION_SOURCE`
to that tree. Its required SHA-256 is
`04ae2020acf89315108a68743f99f8fe9c4cb9fc47e67712298a25a7179d616a`.
Follow the [LiLo recipe](../libero_lilo/README.md) for the pinned
native source and asset details. Build and publish both current contexts:

```bash
python -m benchmarks.package --benchmark libero_lilo --output "$LIBERO_BUILD_ROOT/build-lilo"
python -m benchmarks.package --benchmark libero_pro --output "$LIBERO_BUILD_ROOT/build-pro"
python -m benchmarks.libero_pro.assets \
  --source "$PRO_EVALUATION_SOURCE" --output-dir "$LIBERO_BUILD_ROOT/pro-assets" \
  --build-context "$LIBERO_BUILD_ROOT/build-pro"
docker build -t "$LILO_IMAGE_TAG" "$LIBERO_BUILD_ROOT/build-lilo"
docker build -t "$PRO_IMAGE_TAG" "$LIBERO_BUILD_ROOT/build-pro"
docker push "$LILO_IMAGE_TAG"
docker push "$PRO_IMAGE_TAG"
export LILO_IMAGE=$(docker image inspect "$LILO_IMAGE_TAG" --format '{{index .RepoDigests 0}}')
export PRO_IMAGE=$(docker image inspect "$PRO_IMAGE_TAG" --format '{{index .RepoDigests 0}}')
```

Generate manifests inside those same images so their code, embedded assets and
runtime identities agree. The following selects the fixed 12/9/80 subsets:

```bash
mkdir -p "$RUN_ROOT/registry/libero_long_plus_plus" \
  "$RUN_ROOT/registry/libero_ultra_long" "$RUN_ROOT/registry/libero_pro"
docker run --rm --entrypoint python \
  --mount "type=bind,src=$RUN_ROOT/registry,dst=/output" "$LILO_IMAGE" \
  -m benchmarks.libero_lilo.manifest --image "$LILO_IMAGE" \
  --benchmark libero_long_plus_plus --profile migration_subset \
  --output /output/libero_long_plus_plus/manifest.json
docker run --rm --entrypoint python \
  --mount "type=bind,src=$RUN_ROOT/registry,dst=/output" "$LILO_IMAGE" \
  -m benchmarks.libero_lilo.manifest --image "$LILO_IMAGE" \
  --benchmark libero_ultra_long --profile migration_subset \
  --output /output/libero_ultra_long/manifest.json
docker run --rm --entrypoint python \
  --mount "type=bind,src=$RUN_ROOT/registry,dst=/output" "$PRO_IMAGE" \
  -m benchmarks.libero_pro.manifest --image "$PRO_IMAGE" --profile migration_subset \
  --output /output/libero_pro/manifest.json
```

The manifests request one GPU for EGL. `official` selects the full benchmark
sets; the supplied specialized context maps cover the fixed subsets. Enable
`--plus9-on-failure` only for a separately recorded continuation condition.

## Evaluate with separate containers

Use the rebuilt Agent digest and installed bundle. Credentials are passed only
to the Agent. First inspect the expanded Long++ plan:

```bash
export OPENAI_API_KEY="$(cat "$MODEL_API_KEY_FILE")"
bench eval --registry-root "$RUN_ROOT/registry" \
  --results-root "$RUN_ROOT/results" --run-id longpp-plan \
  --benchmark libero_long_plus_plus --seed 0 \
  --agent-image "$AGENT_IMAGE" --agent-config bundle:libero_long_plus_plus_v13 \
  --agent-credential-env OPENAI_API_KEY --agent-credential-env OPENAI_BASE_URL \
  --agent-set 'runtime.llm.model="gpt-6-astra"' \
  --max-concurrent 1 --timeout-s 43200 --case-timeout-s 14400 --dry-run
```

Remove `--dry-run`, choose a fresh run ID and keep the selected model ID to
execute. Add `--task libero_long_plus_plus:t00-i01` for one case. For the other
subsets use these benchmark/bundle pairs:

| Benchmark | Bundle | Example task |
| --- | --- | --- |
| `libero_ultra_long` | `bundle:libero_ultra_long_v13` | `libero_ultra_long:t00-i01` |
| `libero_pro` | `bundle:libero_pro` | `libero_pro:libero_spatial_task-t00-i03` |

Use `--resume-from OLD_RUN_ID` with a new run ID and unchanged locked inputs.
See [evaluation](../../docs/evaluation.md) for result inspection. Report scored
terminals, Agent/infrastructure failures and native evidence separately.

## Compose your rebuilt images

Set `NEW_LILO_COMBINED_TAG` and `NEW_PRO_COMBINED_TAG` to new tags. Build from the
repository root using the actual Agent and Benchmark digests above:

```bash
docker build -f benchmarks/libero_three_combined/Dockerfile \
  --build-arg AGENT_IMAGE="$AGENT_IMAGE" --build-arg BENCHMARK_IMAGE="$LILO_IMAGE" \
  --build-arg ROBORSI_REVISION="$(git rev-parse HEAD)" \
  -t "$NEW_LILO_COMBINED_TAG" .
docker build -f benchmarks/libero_three_combined/Dockerfile \
  --build-arg AGENT_IMAGE="$AGENT_IMAGE" --build-arg BENCHMARK_IMAGE="$PRO_IMAGE" \
  --build-arg ROBORSI_REVISION="$(git rev-parse HEAD)" \
  -t "$NEW_PRO_COMBINED_TAG" .
docker push "$NEW_LILO_COMBINED_TAG"
docker push "$NEW_PRO_COMBINED_TAG"
docker image inspect "$NEW_LILO_COMBINED_TAG" "$NEW_PRO_COMBINED_TAG" \
  --format '{{json .RepoDigests}}'
```

Long++ and Ultra-Long share the rebuilt LiLo combined image. Record the current
source revision, both input digests and the new output digest. Its default
entrypoint accepts the same single-case arguments shown above. To run the fixed
subset, use the new script path in the rebuilt image:

```bash
export NEW_LILO_COMBINED_IMAGE=$(docker image inspect "$NEW_LILO_COMBINED_TAG" --format '{{index .RepoDigests 0}}')
export NEW_PRO_COMBINED_IMAGE=$(docker image inspect "$NEW_PRO_COMBINED_TAG" --format '{{index .RepoDigests 0}}')
docker run --rm --gpus all --entrypoint python \
  --mount "type=bind,src=$MODEL_API_KEY_FILE,dst=/run/secrets/model_api.key,readonly" \
  --mount "type=bind,src=$RUN_ROOT/results,dst=/results" \
  "$NEW_LILO_COMBINED_IMAGE" \
  -B /opt/roborsi/benchmarks/libero_three_combined/run_combined_subset.py \
  --benchmark libero_long_plus_plus --run-id rebuilt-longpp-subset \
  --results-root /results/rebuilt-longpp-subset --max-concurrent 16 --gpu-count 1 \
  --api-key-file /run/secrets/model_api.key --base-url "$OPENAI_BASE_URL"
```

For Ultra-Long, keep that image and select `--benchmark libero_ultra_long`. For
PRO, use `$NEW_PRO_COMBINED_IMAGE` and `--benchmark libero_pro`. Choose a separate
run ID and output directory for each run. This new path belongs to images built
from the current source; the immutable published images above retain their
original `/opt/roborsi/scripts/libero_three/` path.

## Results and provenance

Each completed case records `result.json`, native score, available native video,
content-addressed observations, Runner envelopes and the Agent trace/config
fingerprint. Keep the image digests, source revision, manifest, model endpoint
identity and budgets with the run. `success` and native `task_failure` are
scored terminals; infrastructure and Agent errors require separate reporting.
