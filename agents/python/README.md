# Docker Agent runtime

The Dockerfile provides three stages:

| Stage | Contents |
| --- | --- |
| `runtime` | Framework source and the selected hash-locked dependencies |
| `configured` | Runtime plus an installed Agent bundle wheel; checks dependencies |
| `libero_three` | Configured runtime plus all three LIBERO bundles, ICL, and tokenizer |
| `example` | Runtime plus the fixed-action LIBERO Agent; default build target |

The Agent runs separately from the native benchmark. `DockerAgentLauncher`
creates a container and bridge network per case. Only the Agent workspace is
writable; the current case's images and container root are read-only. The
launcher sets CPU, memory and process limits and removes the container/network
on completion or failure. The Docker daemon and host administrator remain
trusted. A host `PythonAgentLauncher` runs trusted code without these container
file-access restrictions.

## Build the fixed-action example

Run from the repository root:

```bash
docker build --target example -f agents/python/Dockerfile \
  -t YOUR_REGISTRY/roborsi-agent:example .
docker push YOUR_REGISTRY/roborsi-agent:example
docker image inspect YOUR_REGISTRY/roborsi-agent:example \
  --format '{{json .RepoDigests}}'
```

Replace `YOUR_REGISTRY` and use the returned `repository@sha256:digest` below.
The launcher requires a repository digest, not a mutable tag.

```bash
bench eval --registry-root /absolute/path/to/libero-registry \
  --results-root /absolute/path/to/results --run-id container-agent-1 \
  --agent-image YOUR_REGISTRY/roborsi-agent@sha256:DIGEST \
  --agent-factory roborsi_hold_agent:make_agent \
  --agent-cpus 1 --agent-memory-mb 2048 \
  --benchmark libero --task libero:0 --seed 0 \
  --timeout-s 180 --case-timeout-s 900
```

The fixed-action Agent reads observations and keeps the gripper open. A task
failure is expected. Its runtime needs no simulator or model checkpoint.
The CLI derives the Agent version from the image, factory and preset overrides.

## Build a configured model Agent

Choose a model in the [general Agent preset](../../configs/agents/embodied/default.yaml).
Declare any custom factory code and resources in that preset. Export a package
outside the repository and build its wheel. Run these commands with `build`
and `uv` installed, using fresh temporary directories and a fresh `dist/agent/`:

```bash
python -m roborsi.agents.package \
  --config configs/agents/embodied/default.yaml \
  --name embodied_default --output /tmp/embodied-agent-build
python -m build --wheel --outdir /tmp/roborsi-wheels .
python -m build --wheel --outdir dist/agent /tmp/embodied-agent-build
python -c 'from pathlib import Path; print("roborsi @ " + next(Path("/tmp/roborsi-wheels").glob("roborsi-*.whl")).as_uri())' \
  > /tmp/roborsi-override.txt
uv pip compile /tmp/embodied-agent-build/requirements.in agents/python/requirements.in \
  --overrides /tmp/roborsi-override.txt --no-emit-package roborsi \
  --python-version 3.11 --generate-hashes --output-file dist/agent/requirements.txt

docker build --target configured -f agents/python/Dockerfile \
  --build-arg AGENT_REQUIREMENTS=dist/agent/requirements.txt \
  --build-arg 'AGENT_WHEEL=dist/agent/roborsi_agent_bundle_embodied_default-*.whl' \
  -t YOUR_REGISTRY/roborsi-agent:configured .
docker push YOUR_REGISTRY/roborsi-agent:configured
```

The local framework wheel supplies dependency metadata for lock resolution;
Docker installs framework source from the same checkout. The default
`agents/python/requirements.txt` supports only the fixed-action runtime. A model
Agent needs the complete generated lock. Keep the source revision, lock, bundle
manifest and image digest with the evaluation record.

Select the installed bundle and explicitly pass credential variable names:

```bash
bench eval --registry-root /absolute/path/to/libero-registry \
  --results-root /absolute/path/to/results --run-id configured-agent-1 \
  --agent-image YOUR_REGISTRY/roborsi-agent@sha256:DIGEST \
  --agent-config bundle:embodied_default \
  --agent-credential-env OPENAI_API_KEY \
  --benchmark libero --task libero:0 --seed 0 \
  --timeout-s 180 --case-timeout-s 3600
```

Only an installed `bundle:name` is accepted with `--agent-image`; a host preset
path is not available inside the image. Imports use installed Python packages
through `python -I -B`; host `PYTHONPATH` does not supply Agent code. Rebuild the
wheel and image after changing declared source or resources. `--agent-set`
applies runtime overrides and participates in the recorded Agent version.

Set credentials in the Runner environment and select only their names with
`--agent-credential-env`. Unselected variables are not inherited. For a custom
endpoint, configure the bundle or explicitly pass `OPENAI_BASE_URL` as well.
Do not bake credentials into the image. When combining a Docker Agent with a
remote benchmark, supply `--benchmark-token-env` for the benchmark's separate
Bearer token; do not pass that token to the Agent.

See [Agents](../../docs/agents.md) / [Agent 配置](../../docs/agents.zh-CN.md),
[Evaluation](../../docs/evaluation.md) / [评测](../../docs/evaluation.zh-CN.md),
and [native opt-in tests](../../tests/README.md#libero-cpu-evaluation-check).

## Build the three-LIBERO Agent

Acquire the public, revision-pinned ICL packages and tokenizer first:

```bash
python scripts/libero_three/download_assets.py \
  --output dist/agent/public-assets
```

Export one bundle wheel per benchmark. The tokenizer path is the final in-image
path, and its digest is checked again during the Docker build:

```bash
TOKENIZER_SHA=0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3
mkdir -p ../work/libero-three/bundles dist/agent

python -m roborsi.agents.package \
  --config configs/agents/libero_three/long_plus_plus_v13.yaml \
  --name libero_long_plus_plus_v13 \
  --set 'runtime.context.tokenizer_file="/opt/roborsi-agent-assets/tokenizer.json"' \
  --set "runtime.context.tokenizer_sha256=\"$TOKENIZER_SHA\"" \
  --output ../work/libero-three/bundles/longpp
python -m roborsi.agents.package \
  --config configs/agents/libero_three/ultra_long_v13.yaml \
  --name libero_ultra_long_v13 \
  --set 'runtime.context.tokenizer_file="/opt/roborsi-agent-assets/tokenizer.json"' \
  --set "runtime.context.tokenizer_sha256=\"$TOKENIZER_SHA\"" \
  --output ../work/libero-three/bundles/ultra
python -m roborsi.agents.package \
  --config configs/agents/libero_three/pro.yaml --name libero_pro \
  --set 'runtime.context.tokenizer_file="/opt/roborsi-agent-assets/tokenizer.json"' \
  --set "runtime.context.tokenizer_sha256=\"$TOKENIZER_SHA\"" \
  --output ../work/libero-three/bundles/pro

python -m build --wheel --outdir dist/agent ../work/libero-three/bundles/longpp
python -m build --wheel --outdir dist/agent ../work/libero-three/bundles/ultra
python -m build --wheel --outdir dist/agent ../work/libero-three/bundles/pro
```

Resolve a Python 3.11 hash-locked requirements file from all three generated
`requirements.in` files using the same local RoboRSI-wheel override shown in
the configured-Agent section. Then build:

```bash
docker build --target libero_three -f agents/python/Dockerfile \
  --build-arg AGENT_REQUIREMENTS=dist/agent/requirements.txt \
  --build-arg 'AGENT_WHEEL=dist/agent/roborsi_agent_bundle_libero_*.whl' \
  --build-arg LIBERO_THREE_ASSETS=dist/agent/public-assets/agent-assets \
  -t roborsi-libero-three-agent:local .
```

The image contains neither model weights nor credentials. Override the model
and endpoint at deployment when the default preset name is not available in
your API service.
