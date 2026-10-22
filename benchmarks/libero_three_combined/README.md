# Three-LIBERO single-image deployment

This deployment target combines one locked Agent image with one native
Benchmark image. A job therefore needs one image, an API endpoint and key, and
a writable results directory. Model weights are not included.

Long++ and Ultra-Long use the same LiLo native image but may be tagged as two
delivery images. LIBERO-PRO uses its own native image. Each resulting image
contains the appropriate simulator, the three Agent bundles, all sealed ICL
packages, and the pinned context tokenizer.

## Evaluation profiles

The default `official` profile is the complete released evaluation. The
explicit `migration_subset` profile is the fixed, inexpensive reproduction
subset.

| Benchmark | `official` | `migration_subset` |
| --- | ---: | ---: |
| LIBERO-PRO | 80 tasks × 10 init states = 800 cases | 80 tasks at init state 3 |
| Long++ | 12 configurations × 10 random resets = 120 cases | 12 configurations at init state 1 |
| Ultra-Long | 9 configurations × 10 random resets = 90 cases | 9 configurations at init state 1 |

The ICL package is task-level: every official reset of the same task resolves
to the same sealed package. Reset identity and native scoring remain owned by
the Benchmark.

## Public build inputs

All non-model inputs are publicly downloadable and revision-pinned:

- ICL packages: `sensenova/SenseNova-RoboRSI-Packages` at
  `e977b5ad667b4ce753ee947cbb0b95d4fc0fe98c`;
- PRO BDDL/init: `zhouxueyang/LIBERO-Pro` at
  `c86fc3b8293185a6f373677018ff3e37f8391602`;
- tokenizer: `Qwen/Qwen3.8-Flash-Next/tokenizer.json` at
  `34567a4712bc9766c4449e2e98e4468bfa24d915`.

The downloader deliberately disables Hugging Face authentication. Building the
image therefore needs network access but no Hugging Face account or access
token. Model-service credentials are deployment secrets and never build inputs.

Download and verify them with the command below. If a PRO build context already
exists, add `--pro-build-context /path/to/build-pro` to stage it at the same
time:

```bash
python scripts/libero_three/download_assets.py \
  --output dist/agent/public-assets
```

The command verifies every ICL tree, selects only the eight required PRO
Task/Swap suites, and checks the tokenizer SHA-256. It produces
`agent-assets/`, `pro-assets/`, and `sources.json`. No AFS or m-cluster path is
used.

## Build

First export the PRO context and acquire all public assets:

```bash
python -m benchmarks.package --benchmark libero_pro \
  --output ../work/libero-three/build-pro
python scripts/libero_three/download_assets.py \
  --output dist/agent/public-assets \
  --pro-build-context ../work/libero-three/build-pro
```

Then build the portable Agent image according to
[`agents/python/README.md`](../../agents/python/README.md), using
`dist/agent/public-assets/agent-assets` as `LIBERO_THREE_ASSETS`. Build the LiLo and staged
PRO Benchmark contexts normally.

Build a combined image from the repository root:

```bash
docker build -f benchmarks/libero_three_combined/Dockerfile \
  --build-arg AGENT_IMAGE=repository/agent@sha256:DIGEST \
  --build-arg BENCHMARK_IMAGE=repository/libero-lilo@sha256:DIGEST \
  --build-arg ROBORSI_REVISION="$(git rev-parse HEAD)" \
  -t roborsi-libero-long-plus-plus:local .
```

Use the same LiLo parent for the Ultra-Long delivery image and the PRO parent
for the PRO delivery image. Before publishing, replace every parent tag with a
`repository@sha256:digest` identity.

## Run the complete evaluation

The batch runner defaults to `official`. The example explicitly names it so
the recorded condition is obvious:

```bash
docker run --rm --gpus all --entrypoint python \
  -v /path/to/model_api.key:/run/secrets/model_api.key:ro \
  -v /path/to/results:/results \
  roborsi-libero-long-plus-plus:local \
  -B /opt/roborsi/scripts/libero_three/run_combined_batch.py \
  --benchmark libero_long_plus_plus --profile official \
  --run-id longpp-full --results-root /results/longpp-full \
  --max-concurrent 16 --gpu-count 1 \
  --api-key-file /run/secrets/model_api.key \
  --base-url https://your-responses-endpoint.example/v1 \
  --model gpt-6-astra/YOUR_DEPLOYMENT
```

Select `libero_ultra_long` or `libero_pro` in its corresponding image.

## Run the fixed subset

Use the compatibility wrapper for the 12/9/80 subset:

```bash
docker run --rm --gpus all --entrypoint python \
  -v /path/to/model_api.key:/run/secrets/model_api.key:ro \
  -v /path/to/results:/results \
  roborsi-libero-long-plus-plus:local \
  -B /opt/roborsi/scripts/libero_three/run_combined_subset.py \
  --benchmark libero_long_plus_plus \
  --run-id longpp-subset --results-root /results/longpp-subset \
  --max-concurrent 16 --gpu-count 1 \
  --api-key-file /run/secrets/model_api.key \
  --base-url https://your-responses-endpoint.example/v1 \
  --model gpt-6-astra/YOUR_DEPLOYMENT
```

Both runners preserve attempts and only reuse results whose complete image,
Agent, asset, endpoint, timeout, task, and native-score fingerprints match.
