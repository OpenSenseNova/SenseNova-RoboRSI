# LIBERO Long++ and Ultra-Long

One Linux x86-64 image serves `libero_long_plus_plus` and `libero_ultra_long`.
It pins LiLo-VLA revision `64aa29d77f5457afb3d1642cfc84534dc1e81fad` and LIBERO
revision `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`, including native task assets.
Source archives are checked against the hashes in the Dockerfile.

Long++ has 12 configurations and Ultra-Long has 9. The `official` profile uses
ten bare random resets per configuration. `migration_subset` uses fixed init
state 1 once. The 1x budget is 200 physical steps per ordered skill, the score
is ordered-prefix completion, and success requires the complete sequence.
The live instance-pose feedback channel is part of this adapter's observation
contract and does not change native skill predicates or scoring.

## Build and register

Run from the installed framework checkout. Docker and network access to the
pinned source archives and Python packages are required.

```bash
export ROBORSI_WORKSPACE=/absolute/path/to/libero-lilo-work
python -m benchmarks.package --benchmark libero_lilo \
  --output "$ROBORSI_WORKSPACE/build"
docker build -t YOUR_REGISTRY/roborsi-libero-lilo:BUILD "$ROBORSI_WORKSPACE/build"
docker push YOUR_REGISTRY/roborsi-libero-lilo:BUILD
export ROBORSI_IMAGE=$(docker image inspect YOUR_REGISTRY/roborsi-libero-lilo:BUILD \
  --format '{{index .RepoDigests 0}}')
mkdir -p "$ROBORSI_WORKSPACE/registry/libero_long_plus_plus" \
  "$ROBORSI_WORKSPACE/registry/libero_ultra_long"
```

Replace `YOUR_REGISTRY` and `BUILD` before running. Generate manifests inside
the image, where pinned runtime metadata is available:

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$ROBORSI_WORKSPACE/registry,dst=/config" \
  "$ROBORSI_IMAGE" python -m benchmarks.libero_lilo.manifest \
  --image "$ROBORSI_IMAGE" --benchmark libero_long_plus_plus \
  --profile official --output /config/libero_long_plus_plus/manifest.json
docker run --rm --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$ROBORSI_WORKSPACE/registry,dst=/config" \
  "$ROBORSI_IMAGE" python -m benchmarks.libero_lilo.manifest \
  --image "$ROBORSI_IMAGE" --benchmark libero_ultra_long \
  --profile official --output /config/libero_ultra_long/manifest.json
```

The manifests request one GPU for EGL rendering, two CPUs and 6144 MiB memory.
They record the selected profile; changing it changes the evaluation condition.
Runtime scene assets are embedded in the image. Mutable path configuration and
Numba caches use writable `/tmp`.

## Evaluate

Set the model in the general Agent preset, then run:

```bash
bench eval --registry-root "$ROBORSI_WORKSPACE/registry" \
  --results-root "$ROBORSI_WORKSPACE/results" --run-id lilo-1 \
  --agent-config configs/agents/embodied/default.yaml \
  --benchmark libero_long_plus_plus --seed 0 \
  --timeout-s 600 --case-timeout-s 43200
```

Select task IDs from the generated manifest and add `--task NAME:TASK_ID` to
run fewer cases. Use `libero_ultra_long` to select the other family. The
`hold_agent.py` factory also accepts both names for transport checks.

`--plus9-on-failure` is an explicitly named diagnostic that continues failed
1x episodes; it never changes the recorded 1x score. General Agent use is documented in
[English](../../docs/agents.md) and [Chinese](../../docs/agents.zh-CN.md).

See [Evaluation](../../docs/evaluation.md), [tests](../../tests/README.md), and
[third-party notices](THIRD_PARTY_NOTICES.md).
For a scheduler that accepts only one job image, use the
[three-LIBERO single-image target](../libero_three_combined/README.md).
