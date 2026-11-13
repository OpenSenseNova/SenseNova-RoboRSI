# DexJoCo benchmark

[简体中文](README.zh-CN.md)

The DexJoCo adapter exposes the pinned upstream simulator through RoboRSI's
Benchmark HTTP service. It discovers 11 `rand_full` tasks with seeds
`100000`–`100004`: 55 task/seed cases. The simulator revision is
`8d23b0fab23b17a58c4b55f3942e17013aaf8267`.

This release includes the benchmark, native process wrapper, action and
observation contracts, and container recipe. Choose your own Agent through the
shared evaluation interface.

## Build and register

Prepare the native runtime and assets using this recipe; installing the framework
does not supply these dependencies. Read the [license and attribution notices](#validation-and-licenses)
before redistributing the runtime or assets.

Use Linux x86-64 for the native runtime. The Dockerfile pins a Python 3.10 base
image, installs hash-locked dependencies, and checks out the pinned simulator.
Rendering uses CPU OSMesa; no GPU or model is needed by the benchmark service.
Allow about 4 GiB RAM per simulator plus storage for per-step image artifacts.

From the repository root, after installing the framework:

```bash
python -m benchmarks.package --benchmark dexjoco --output ../dexjoco-context
docker build -t dexjoco-benchmark:local ../dexjoco-context

git clone https://github.com/brave-eai/dexjoco.git ../dexjoco-upstream
git -C ../dexjoco-upstream checkout 8d23b0fab23b17a58c4b55f3942e17013aaf8267
```

Build only the exported context. Its `source.json` and `build.json` record the
runtime and build input hashes. The upstream checkout used for discovery must
be clean at the pinned revision.

Publish the built image through your registry and set `DEXJOCO_IMAGE_WITH_DIGEST`
to its actual `repository@sha256:digest`. A local tag alone is insufficient for
a registry manifest; this repository supplies no published image digest.

```bash
python -m benchmarks.dexjoco.manifest --upstream ../dexjoco-upstream \
  --image "$DEXJOCO_IMAGE_WITH_DIGEST" \
  --output ../benchmark-registry/dexjoco/manifest.json
```

The command creates missing parent directories for `--output` automatically.
The generated manifest describes the default 10x service, all 55 cases, native
scoring, and in-image assets. No separate host asset mount is required.

## Evaluate an Agent

Set your model provider credentials in the environment and select a vision and
tool-calling model. Use the shared Agent configuration with the DexJoCo
benchmark name and task IDs shown below.
The service exposes the native action schema and scoring rules. The fixed-action
LIBERO example in [Getting started](../../docs/getting-started.md) only supports
LIBERO. The original ICL construction sources, frozen-package preparation and
two-session task Agent are described in [the ICL guide](ICL.md).
Historical ICL packages are not
included; the generic configuration does not reproduce historical scores or
claim task success.

```bash
bench eval --registry-root ../benchmark-registry \
  --benchmark dexjoco --task dexjoco:water_plant --seed 100000 \
  --agent-config configs/agents/embodied/default.yaml \
  --agent-set 'runtime.llm.model="YOUR_VISION_TOOL_MODEL"' \
  --results-root ../dexjoco-results --run-id water-pilot \
  --timeout-s 3600 --case-timeout-s 21600 --dry-run
```

Remove `--dry-run` to execute the case and make model calls. Omit `--task` and
`--seed` for all 55 cases; use `--max-concurrent 1` to run sequentially. Runner
timeouts bound wall-clock execution and do not replace physical action budgets.
See [Agents](../../docs/agents.md) and [Evaluation](../../docs/evaluation.md)
for custom Agent factories, context, tool budgets, results and resume.

For a direct local service, use a separate Python 3.10 environment with
`pip install --require-hashes -r benchmarks/dexjoco/requirements.txt`, then
run from the repository root:

```bash
export ROBORSI_DEXJOCO_UPSTREAM="$(realpath ../dexjoco-upstream)"
export ROBORSI_DEXJOCO_MULTIPLIER=10
export MUJOCO_GL=osmesa
python -m roborsi.evaluation.service_host --kind benchmark \
  --factory robots.dexjoco.benchmark_service:make_dexjoco_benchmark \
  --host 127.0.0.1 --port 8000
```

Install `git`, `libosmesa6`, `libgl1` and `libglib2.0-0` in that host environment.
Use `--benchmark-url dexjoco=http://127.0.0.1:8000` in place of the registry
option. Start a fresh service for an independent run of the same task/seed,
because the service retains episode IDs. Set the multiplier to `1` before
starting the service for the original horizon; only `1` and `10` are supported.

## Actions, stopping and scoring

Each action is `{"vector": [...]}` with 22 finite numeric targets for one arm or
44 for two arms: absolute xyz in metres, a rotation vector in radians, and
16 hand targets per arm, right arm first. The adapter converts rotations to
scalar-first quaternions and preserves native hand ordering. One action is one
native 30 Hz control step, with no clipping or interpolation. Click Mouse
requires the exact `ALIGNMENT` vector from
[`contracts.py`](../../robots/dexjoco/contracts.py) for its first 30 steps.

The public observation contains the task instruction, robot proprioception and
configured RGB cameras. Native rewards, object state, score, and private
checkpoints remain evaluation artifacts.

The original per-task horizon is 1,000–1,500 steps. At 10x, only the native
terminal step comparison is extended in memory; upstream files and other
failure or success predicates stay unchanged. An active episode continues
through the 1x boundary without resetting the simulator. The adapter saves a
private checkpoint and reports 1x and cumulative 10x outcomes. Before that
boundary, `remaining_steps` reports the original budget. Agent session changes
or replanning at 1x remain the Agent's responsibility.

Native termination determines success. An explicit policy stop is a scored
`policy_failure` and grants no extra steps. Infrastructure errors remain errors.
The primary score is native success, aggregated as a task-weighted success rate
with the expected case denominator. Timeouts, retries and model/tool budgets
must not be treated as permission to exceed native physical limits.

## Validation and licenses

```bash
pytest tests/unit_tests/robots/dexjoco -q
```

These offline tests use fake simulators and check discovery, actions,
information boundaries, stopping, score artifacts, and cleanup. They establish
interface behavior, not native simulator execution or model task success.

The framework license does not replace upstream code, robot, mesh or other
asset licenses. Preserve the notices in the
[upstream simulator](https://github.com/brave-eai/dexjoco) when redistributing
its checkout or container assets. The horizon adaptation retains its MIT notice
in [LICENSE.reference](LICENSE.reference).
