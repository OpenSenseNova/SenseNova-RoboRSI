# VLNCE delivery recipe

Runtime: Python 3.8.20, Habitat-Lab / Habitat-Sim 0.1.7, VLN-CE source
`729d141b2ee10628061ada74dd3a5b9f70faeba5`. R2R and RxR use separate catalogs.
The adapter loads only official evaluation modules; it does not load training
baselines. R2R uses the selected `rand100` annotations and reference paths.

`native-environment.json` records the validated target native package inventory.
`provenance.json` identifies the source binding. Preserve MP3D licensing and
distribute assets and native source externally, not in Git.

## Agent entry

Use `bench eval --agent-config robots/vlnce/agent.yaml` with the Registry and
task selection arguments for R2R or RxR. This entry inherits
`robots/vlnce/agents/baseline/agent.yaml` and the common
[navigation baseline](../../robots/vln_shared/agents/baseline/README.md).
The factory, prompts and tool policy are shared with LHPR-VLN. Dependencies stay
in `agents/vln_shared/requirements.txt`.

The standard entry starts without frozen training memory. For the shared
exploration and conditional 500/5000 experiment, follow the
[run guide](../../docs/vln_shared_evaluation.md) and use the generated dataset
preset or its exported wheel. The one-click launcher selects this standard
entry before attaching the dataset's exploration snapshot.

## Build contract

The Dockerfile extends a **digest-pinned native runtime image** that supplies
`/opt/native/bin/python` with the recorded native environment, and a separate
Python 3.11 interpreter at `/opt/roborsi-service/bin/python`. It validates the
native package versions and installs the pinned HTTP service dependencies.
Do not mix Habitat 0.1.7 into the RoboRSI Agent interpreter.

Public runtime-source registration includes `vlnce`, its service factory/native
adapter and four shared service modules: `benchmark.py`, `native_process.py`,
`native_worker.py`, and `storage.py`. Agent, exploration, campaign code and the
LHPR adapter are excluded. The exported service and manifest use the same
registered source digest as `adapter_version`.

```bash
python -m benchmarks.package --benchmark vlnce --output /tmp/vlnce-context
docker build --build-arg NATIVE_RUNTIME_IMAGE=REGISTRY/IMAGE@sha256:ACTUAL_DIGEST \
  -t YOUR_VLNCE_IMAGE /tmp/vlnce-context
```

Use the released native base by its immutable digest. A build recipe alone is
not acceptance evidence: verify the exported source, native environment, GPU
reset/step/finish and Registry deployment before publishing a benchmark digest.

## Stage assets and seal the Registry manifest

First prepare the local campaign as described in the
[Chinese run guide](../../docs/vln_shared_evaluation.zh-CN.md) or
[English run guide](../../docs/vln_shared_evaluation.md).
Create a JSON mapping of explicit source directories to disjoint relative asset
directories. For example:

```json
{
  "/path/to/gpt6-test/third_party/benchmarks/vln-ce": "native/vln-ce",
  "/path/to/gpt6-test/data": "data"
}
```

Staging copies these complete directories, including licensed scenes; allocate
enough storage or prepare a smaller complete input root first. The source roots
remain unchanged. Source links are dereferenced into regular copied files.

```bash
python -m robots.vln_shared.assets stage --config /path/to/campaign.json \
  --dataset r2r --mapping /path/to/path-map.json --output /delivery/r2r/assets
python -m robots.vln_shared.assets manifest --assets /delivery/r2r/assets \
  --image YOUR_VLNCE_IMAGE@sha256:ACTUAL_DIGEST --output /delivery/r2r/manifest.json
```

The script rebases catalog paths, checks copied input hashes, and seals the
whole copied directory using the standard Registry hash. Assets must be proper
relative subdirectories of the manifest and have no symlink components. The
container mounts the asset directory at `/roborsi/assets/vlnce`. Repeat with
`--dataset rxr` and a separate output directory. A manifest is for one fixed
budget; the local conditional launcher independently constructs 500 and 5000
Providers. Asset export does not claim that an image has been built or tested.

Local tests cover native reset, RGB transport, fixed actions, original scoring
and direct/native parity. The one-click campaign currently uses local isolated
native workers behind the common HTTP service; image deployment is a separate
delivery validation step.

## Native dependency and graphics constraints

The delivery includes `future==1.0.0`, required by pyglet `1.5.0`; the wheel hash
is in `native-fixes.txt`. `Dockerfile.native` applies it to the pinned source
environment. Both the native base and benchmark build must pass `pip check`.

The image prioritizes the system GLVND/EGL libraries explicitly to avoid mixing
old Conda graphics libraries with the NVIDIA driver. For an isolated host
environment, install `python-system-gl` alongside its `bin/python` and use the
launcher as the campaign's native Python executable. This does not modify
system libraries or the existing Python executable.
