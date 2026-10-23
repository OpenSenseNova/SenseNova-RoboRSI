# RoboCasa365 Target50 benchmark

[简体中文](README.zh-CN.md)

This integration serves RoboCasa365 Target50 through RoboRSI's common Benchmark
HTTP interface. One action invokes one native RoboCasa tool, including navigation,
camera inspection and RLDX-1 manipulation. The simulator's final `state.success`
determines the native verdict. RLDX-1 is an execution dependency of the published
tool contract and runs in a separate process.

| Item | Value |
| --- | --- |
| Benchmark ID | `robocasa365` |
| Simulator | `RLinf/robocasa@2692d8fc5fd86708a1b2028dcbce892ec418a9e7`, `RLinf/robosuite@97cfbde4b68d8ec43dad20cf4747297866a6ca2e`, MuJoCo 3.3.1 |
| RLDX runtime | `RLinf/RLDX-1@ebcfd13df5177e4b3e574bdf5a3b427c7c4a1e8a` |
| RLDX checkpoint | `RLWRLD/RLDX-1-FT-RC365@587e9ecdcc5e7184fcc17f58713908edff5af041` |
| Cells | 340: 18 atomic tasks × seeds 1–10, 16 composite seen and 16 composite unseen tasks × seeds 1–5 |
| Score | Mean of the 50 per-task success rates; missing or invalid cells count as failures |

The task list, split budgets and dependency revisions are recorded in
[`target50.json`](../../robots/robocasa/eval/target50.json). Each episode exposes
live task language, public state and camera views. `finish` and an explicit
episode stop score the current native state. Invalid tool arguments return tool
feedback; the action must still select a supported primitive.

## One-command evaluation

From the repository root, set the model API and run:

```bash
export OPENAI_BASE_URL='http://127.0.0.1:8000/v1'
export OPENAI_API_KEY='YOUR_API_KEY'   # EMPTY if the server needs no key
export MODEL_NAME='YOUR_SERVED_MODEL'

# One cell, to check the setup
python scripts/evaluation/evaluate_model.py --benchmark robocasa365 \
  --task robocasa365:OpenDrawer --seed 1 --run-id robocasa-smoke

# The 50 seed-1 cells
ROBORSI_ROBOCASA_CELLS=seed1 python scripts/evaluation/evaluate_model.py \
  --benchmark robocasa365 --run-id robocasa-seed1

# All 340 cells
ROBORSI_ROBOCASA_CELLS=all python scripts/evaluation/evaluate_model.py \
  --benchmark robocasa365 --run-id robocasa-full
```

The script builds the service image from this checkout, downloads the Target50
ICL package and runs the Target50 Agent
([`configs/agents/robocasa365/icl.yaml`](../../configs/agents/robocasa365/icl.yaml)).
Add `--openai-format responses` for a Responses API model. To use a Benchmark
service that is already running, add
`--benchmark-url robocasa365=http://127.0.0.1:18800`.

Results are written to `work/model_eval/results/`, one directory per split
group. Score a seed-1 run with:

```bash
python -m benchmarks.robocasa365.score \
  work/model_eval/results/robocasa-seed1-robocasa365-* --cells seed1
```

Omit `--cells seed1` when scoring all 340 cells. See the
[model evaluation guide](../../docs/model-evaluation.md) for all options.

## Build and run

Follow [BUILD.md](BUILD.md) to export a context and build your image. The original
digest-pinned GHCR base requires pull permission. This repository does not
provide a complete recipe to recreate that base from public inputs. Its native
environment includes kitchen assets and RLDX-1 weights; installing RoboRSI on the
host does not install them. Upstream simulator, asset and checkpoint terms apply.

For native execution, use Linux x86-64, Docker, the NVIDIA Container Toolkit and
a driver supporting CUDA 12.6. The inherited environment uses Python 3.10.21 and
Torch 2.7.0+cu126. RLDX-1 needs a GPU (the source delivery budgets about 16 GB
VRAM); kitchens render with OSMesa on the CPU. Reserve space for the approximately
25.94 GB compressed base plus extraction, build layers and results.

Set `ROBOCASA365_IMAGE` to the image you built. The image's default service starts
its own RLDX-1 process from `RLDX_CHECKPOINT` when the first episode begins:

```bash
docker run --rm --gpus all -p 127.0.0.1:18800:8000 \
  -e ROBORSI_ROBOCASA_CELLS=seed1 "$ROBOCASA365_IMAGE"
```

`seed1` exposes 50 cells; omit the variable or set it to `all` for 340 cells.
Each episode owns an environment subprocess and closes its RLDX session.
The Benchmark service pins 40 RLDX chunks, settle patience 999 and eight action
steps per chunk. The owned RLDX server exits with the service.

## Evaluate and score

Install the framework in your Runner environment and configure a model using the
[Agent guide](../../docs/agents.md). From the repository root:

```bash
bench eval --benchmark-url robocasa365=http://127.0.0.1:18800 \
  --benchmark robocasa365 --task robocasa365:OpenDrawer --seed 1 \
  --agent-config configs/agents/embodied/default.yaml \
  --agent-set 'runtime.llm.model="YOUR_VISION_TOOL_MODEL"' \
  --agent-set 'agent.max_turns=100' \
  --agent-set 'agent.planner_timeout_s=1800' \
  --results-root work/robocasa365/results --run-id open-drawer-1 \
  --max-concurrent 1 --timeout-s 1800 --case-timeout-s 4500
```

Target50 assigns 1800 seconds to atomic tasks and 3600 seconds to
composite tasks; select the matching Agent planner budget for each run. Runner
timeouts are separate transport and case limits. The service caps native tool
actions at 400; this does not replace the Target50 planner limits.

Score saved runs with:

```bash
python -m benchmarks.robocasa365.score \
  work/robocasa365/results/open-drawer-1 --cells seed1
```

`--cells seed1` uses all 50 seed-1 cells as the denominator, including cells not
run in the example. Omit it for the complete 340-cell protocol. The scorer reports
missing or invalid cells and allows later valid attempts to replace earlier
invalid attempts. A planner timeout counts as a valid task failure.

## Registry or a shared RLDX service

For Registry-managed containers, push your built image to your registry and use
its actual repository digest. Set `ROBOCASA365_IMAGE_DIGEST` to that
`repository@sha256:digest` and generate the manifest inside that same image:

```bash
mkdir -p work/robocasa365/registry
docker run --rm \
  --mount type=bind,src="$PWD/work/robocasa365/registry",dst=/registry \
  "$ROBOCASA365_IMAGE_DIGEST" \
  python -m benchmarks.robocasa365.manifest --image "$ROBOCASA365_IMAGE_DIGEST" \
  --output /registry/robocasa365/manifest.json --cells seed1
```

The manifest requests one GPU, eight CPUs, 32 GB memory and a 900-second startup
limit. Assets and weights remain in the image, so the asset list is empty. Replace
`--benchmark-url` with `--registry-root work/robocasa365/registry` when evaluating.
Generating it inside the delivery image keeps the adapter identity aligned with
the running service. A manifest generated from this checkout must not be paired
with an earlier published image whose source digest differs.

To share an existing RLDX process across Benchmark services, the original native
entry point is available in the image:

```bash
docker run --rm --gpus all --network host "$ROBOCASA365_IMAGE" \
  python robots/robocasa/vla_server.py --model-path /opt/rldx/RLDX-1-FT-RC365 \
  --transport http --host 127.0.0.1 --port 18710
```

Start each Benchmark service with `--network host` and
`-e ROBORSI_ROBOCASA_VLA_ENDPOINT=http://127.0.0.1:18710`, using a different HTTP
port for each service. This endpoint is an alternative to the service-owned
RLDX process; the task actions and scoring stay the same.

Offline tests cover catalog and manifest agreement, tool dispatch, native score
publication, cleanup and source exports. They do not establish a successful
native image build, simulator episode or task success for this migration.
