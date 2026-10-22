# LIBERO-PRO

This Linux x86-64 benchmark exposes eight Task/Swap suites, each with ten tasks
and ten initial states: 800 `official` cases. `migration_subset` selects fixed
init state 3 for all 80 tasks. Native final-goal success is measured at the
10,000-physical-step 1x boundary.

The runtime pins LIBERO-PRO revision `d1e11fb181b8544487d27742c0caa3a4d46452ad`.
The Dockerfile records source hashes and package versions, including Python
3.11, MuJoCo 3.3.0, robosuite 1.5.2 and NumPy 1.26.4.

## Required evaluation assets

The required Task/Swap BDDL and init files are public in the official
[`zhouxueyang/LIBERO-Pro`](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro)
dataset. The build pins revision
`c86fc3b8293185a6f373677018ff3e37f8391602` and selects only the eight suites
used by this evaluation: 80 BDDL files and 80 init files. Unrelated files from
an existing LIBERO checkout never enter the image.

The asset preparer and Docker build require directory SHA-256:

```text
d4a0a1ddbe8137cbcb410a41d9ec0296f27cb709e03f0ae6f9bd3fb11da7b59d
```

A different or partial tree is rejected. Do not replace this digest merely to
make a build pass: different assets define a different evaluation condition.

## Build and register

Run from the installed framework checkout with Docker available:

```bash
export ROBORSI_WORKSPACE=/absolute/path/to/libero-pro-work
python -m benchmarks.package --benchmark libero_pro \
  --output "$ROBORSI_WORKSPACE/build"
python scripts/libero_three/download_assets.py \
  --output "$ROBORSI_WORKSPACE/public-assets" \
  --pro-build-context "$ROBORSI_WORKSPACE/build"
docker build -t YOUR_REGISTRY/roborsi-libero-pro:BUILD "$ROBORSI_WORKSPACE/build"
docker push YOUR_REGISTRY/roborsi-libero-pro:BUILD
export ROBORSI_IMAGE=$(docker image inspect YOUR_REGISTRY/roborsi-libero-pro:BUILD \
  --format '{{index .RepoDigests 0}}')
mkdir -p "$ROBORSI_WORKSPACE/registry/libero_pro"
```

Replace `YOUR_REGISTRY` and `BUILD` before running. The downloader also stages
the public ICL packages and tokenizer under `public-assets/agent-assets` for the
Agent image. The PRO image embeds only the verified eight evaluation suites
under `/opt/libero-pro-evaluation`; execution needs no separate asset mount.
Mutable path configuration uses `/tmp`.

Generate the manifest inside the built image, which reports the pinned source
revision and embedded asset identity:

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$ROBORSI_WORKSPACE/registry/libero_pro,dst=/config" \
  "$ROBORSI_IMAGE" python -m benchmarks.libero_pro.manifest \
  --image "$ROBORSI_IMAGE" --profile official --output /config/manifest.json
```

The manifest requests one GPU for EGL rendering, two CPUs and 6144 MiB memory.
Use `--profile migration_subset` explicitly when selecting that smaller condition.

## Evaluate

Set the model in the general Agent preset, then run:

```bash
bench eval --registry-root "$ROBORSI_WORKSPACE/registry" \
  --results-root "$ROBORSI_WORKSPACE/results" --run-id pro-1 \
  --agent-config configs/agents/embodied/default.yaml \
  --benchmark libero_pro --seed 0 --timeout-s 600 --case-timeout-s 43200
```

Select task IDs from the manifest and add `--task libero_pro:TASK_ID` to run
fewer cases. `--plus9-on-failure` is a separate diagnostic that may continue a
failed 1x episode without changing its recorded 1x score.

Use [Agents](../../docs/agents.md) / [Agent 配置](../../docs/agents.zh-CN.md)
for the public presets and [Evaluation](../../docs/evaluation.md) for outputs.
Offline adapter checks do not establish native task success. See
[tests](../../tests/README.md) and [third-party notices](THIRD_PARTY_NOTICES.md).
For a scheduler that accepts only one job image, use the
[three-LIBERO single-image target](../libero_three_combined/README.md).
