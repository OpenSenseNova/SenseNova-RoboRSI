# BEHAVIOR-1K

[简体中文](README.zh-CN.md) · [All guides](../../docs/README.md)

This integration exposes BEHAVIOR-1K through the shared Benchmark service and
Runner. It provides native task inspection, an isolated simulator worker, the
official 21D action interface, RGB-D observations and native scoring.

## Tasks, actions and scores

The native source is BEHAVIOR-1K v3.9.2, commit
`b1979916ec1549b10a4e65e630bc6504a9af1b00`. The
[task selection](official_tasks.json) contains 100 tasks. Catalog preparation
resets each selected task and reads its actual instruction and step limit.
The service exposes the tasks in that catalog using `public_test` instance
301 (public-test index 0), seed 0.

Each action is `{"action": [...]}` with exactly 21 finite values, in this order:

| Values | Control |
| --- | --- |
| 3 | Base velocity |
| 4 | Torso joint targets |
| 6 | Left end-effector position and rotation vector in the base frame |
| 1 | Left gripper |
| 6 | Right end-effector position and rotation vector in the base frame |
| 1 | Right gripper |

Actions must satisfy the native bounds; the adapter rejects out-of-range
commands without clipping or normalization. Observations contain full-resolution
head and wrist RGB/depth PNGs, public proprioception and action bounds. Depth
images encode inverse log intensity over 0.01–10 m with a 3.5 m shift; pixel
values are display intensities, not metric distances.

Success comes from native `task.success`; score comes from the official
aggregated `q_score.final`. An explicit stop scores the current state without
advancing physics. Native step or metric failures are reported as infrastructure
errors. Each episode owns an isolated simulator process and bounded cleanup.

## Environment and image

Use Linux x86_64, Python 3.11, Docker, a supported NVIDIA driver and the NVIDIA
Container Toolkit. Simulation requires an RTX-class GPU; 24 GB VRAM is
recommended. Provide 32 GB RAM and disk space for the image, licensed assets
and their snapshot. Rebuilding from source additionally requires at least
80 GB free build disk.

Install the Agent and evaluator from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[runtime]'
```

Set `BEHAVIOR_DELIVERY` to an absolute directory outside the checkout, without
symlink components. Pull the published Benchmark image and resolve its digest.
If the GHCR package requires authentication, first run
`docker login ghcr.io -u YOUR_GITHUB_USERNAME` with an account that has pull access.

```bash
mkdir -p "$BEHAVIOR_DELIVERY"
BEHAVIOR_IMAGE_TAG=ghcr.io/opensensenova/sensenova-roborsi-behavior-1k:v1.0.0-cu128
docker pull "$BEHAVIOR_IMAGE_TAG"
BEHAVIOR_IMAGE=$(docker image inspect "$BEHAVIOR_IMAGE_TAG" --format '{{index .RepoDigests 0}}')
```

Use the resolved `repository@sha256:digest` in `BEHAVIOR_IMAGE` throughout the
following commands, and retain it with your deployment records. Evaluating with
this published image requires neither a new build nor registry write access.
The [Dockerfile](Dockerfile), [build lock](build-lock.json) and
[build script](build_image.py) remain available for source rebuilds.

## Assets, catalog and registry

### Download the datasets and obtain the key

Evaluation requires simulator assets, task instances and `omnigibson.key`. The official download repository is
[behavior-1k/zipped-datasets](https://huggingface.co/datasets/behavior-1k/zipped-datasets/tree/main).
The native source pinned above uses these archives:

| Asset directory | Archive in the official repository |
| --- | --- |
| `behavior-1k-assets` | `behavior-1k-assets-3.9.0.zip` |
| `omnigibson-robot-assets` | `omnigibson-robot-assets-3.8.2.zip` |
| `2026-challenge-task-instances` | `2026-challenge-task-instances.zip` |

**1. Use the provided image.** Pull
`ghcr.io/opensensenova/sensenova-roborsi-behavior-1k:v1.0.0-cu128`
and set `BEHAVIOR_IMAGE` as shown above. The image includes OmniGibson; no local
native environment installation or activation is needed.

**2. Set the data directory and download.** Choose a writable absolute directory
outside the checkout. Reserve space for archives, extracted data and the snapshot
copy created below. Mount that directory into the container to keep downloads
on the host:

```bash
export BEHAVIOR_DATA=/absolute/path/outside/checkout/behavior-data
mkdir -p "$BEHAVIOR_DATA"
docker run --rm -it --gpus all \
  --mount type=bind,src="$BEHAVIOR_DATA",dst=/roborsi/assets/behavior_1k \
  -e OMNIGIBSON_DATA_PATH=/roborsi/assets/behavior_1k \
  "${BEHAVIOR_IMAGE:?Pull the provided image and resolve its digest first}" \
  python -m omnigibson.utils.asset_utils \
  --download_behavior_1k_assets \
  --download_omnigibson_robot_assets \
  --download_2026_challenge_task_instances
```

**3. Obtain the key.** On first use, read the license and enter `y` if you agree.
The command saves the key to `$BEHAVIOR_DATA/omnigibson.key` and continues
downloading the datasets.

The resulting local directory is:

```text
$BEHAVIOR_DATA/
  behavior-1k-assets/
  omnigibson-robot-assets/
  2026-challenge-task-instances/
  omnigibson.key
```

**4. Check the downloaded paths.** After the command finishes, check the required
paths without displaying the key:

```bash
test -s "$BEHAVIOR_DATA/omnigibson.key" &&
test -s "$BEHAVIOR_DATA/behavior-1k-assets/VERSION" &&
test -s "$BEHAVIOR_DATA/omnigibson-robot-assets/VERSION" &&
test -d "$BEHAVIOR_DATA/2026-challenge-task-instances/metadata" &&
printf 'Key and required asset paths are present.\n'
```

This is a presence check, not a complete integrity check. The upstream downloader
skips any existing dataset directory, even if a previous extraction was
interrupted. For an incomplete installation, move the affected directory aside
or select a fresh data directory before retrying. The downloader does not pin
the HF repository revision; retain the complete snapshot and its SHA-256 below
for reproducibility.

### Snapshot the assets and register tasks

`prepare assets` copies and hashes already downloaded files; it does not fetch
datasets or obtain the key. Keep the key in the private asset mount and reserve
space for the snapshot copy. Run from the repository root:

```bash
python -m benchmarks.behavior_1k.prepare assets \
  --source "$BEHAVIOR_DATA" \
  --output-dir "$BEHAVIOR_DELIVERY/registry/behavior_1k/assets"
BEHAVIOR_SNAPSHOT=$(python -c \
  'import json, sys; print(json.load(open(sys.argv[1]))["path"])' \
  "$BEHAVIOR_DELIVERY/registry/behavior_1k/assets/asset.json")
```

Run the following commands in Bash from the repository root. They read all
100 tasks from `official_tasks.json`, then perform native reset for each task
inside the pulled image to generate the complete catalog. Each task uses seed 0,
public_test instance 301. The catalog and work directories must be new. This
loads 100 native tasks and takes substantially longer than a single-task check.

```bash
python - "$BEHAVIOR_DELIVERY/catalog-task-args.txt" <<'PY'
import json
import sys
from pathlib import Path

tasks = json.loads(Path("benchmarks/behavior_1k/official_tasks.json").read_text())["tasks"]
assert len(tasks) == len(set(tasks)) == 100, "Expected 100 unique official tasks"
Path(sys.argv[1]).write_text("".join(f"--task\n{task}\n" for task in tasks))
PY
mapfile -t BEHAVIOR_TASK_ARGS < "$BEHAVIOR_DELIVERY/catalog-task-args.txt"

docker run --rm --gpus all \
  --mount type=bind,src="$BEHAVIOR_SNAPSHOT",dst=/roborsi/assets/behavior_1k,readonly \
  --mount type=bind,src="$BEHAVIOR_DELIVERY",dst=/delivery \
  "$BEHAVIOR_IMAGE" python -m benchmarks.behavior_1k.prepare catalog \
  "${BEHAVIOR_TASK_ARGS[@]}" \
  --output-dir /delivery/registry/behavior_1k/catalog \
  --work-dir /delivery/catalog-work
```

After generation, verify that the catalog contains exactly the 100 official
tasks before registering it. Resolve generation or validation failures before
proceeding; do not register a partial catalog as the full task set:

```bash
python - "$BEHAVIOR_DELIVERY/registry/behavior_1k/catalog/catalog.json" <<'PY'
import json
import sys
from pathlib import Path

expected = json.loads(Path("benchmarks/behavior_1k/official_tasks.json").read_text())["tasks"]
rows = json.loads(Path(sys.argv[1]).read_text())["tasks"]
assert len(rows) == len(expected) == 100
assert {row["task_id"] for row in rows} == set(expected), "Catalog task mismatch"
assert all(row["seed"] == 0 and row["native"]["instance_id"] == 301 for row in rows)
print("Verified catalog: 100 tasks, seed 0, public_test instance 301")
PY
```

Read the runtime identity from the same image, then create its manifest:

```bash
docker run --rm \
  --mount type=bind,src="$BEHAVIOR_DELIVERY",dst=/delivery \
  "$BEHAVIOR_IMAGE" python -m benchmarks.behavior_1k.prepare runtime-info \
  --output /delivery/runtime-info.json

python -m benchmarks.behavior_1k.prepare manifest \
  --image "$BEHAVIOR_IMAGE" \
  --assets "$BEHAVIOR_SNAPSHOT" \
  --catalog "$BEHAVIOR_DELIVERY/registry/behavior_1k/catalog" \
  --runtime-info "$BEHAVIOR_DELIVERY/runtime-info.json" \
  --output "$BEHAVIOR_DELIVERY/registry/behavior_1k/manifest.json"
```

The resulting layout is:

```text
registry/behavior_1k/
  assets/
    asset.json
    snapshots/<sha256>/
  catalog/catalog.json
  manifest.json
```

Assets and catalog must be inside the manifest directory without parent
traversal or symlinks. The manifest records their relative paths and hashes,
the image digest and inspected tasks. Registry launch mounts the snapshot at
`/roborsi/assets/behavior_1k` and the catalog at
`/roborsi/assets/behavior_1k-catalog`.

## Evaluate with an Agent

Set `OPENAI_API_KEY` in the Agent environment and select a vision model with
tool calling. The shared preset receives the task, 21D action schema and RGB-D
observations. After registering the complete catalog above, the following command
evaluates all 100 tasks at seed 0, producing 100 cases. Omitting `--task` selects
all tasks in this Benchmark catalog; `--max-concurrent 1` runs them sequentially
to avoid multiple simulators competing for one GPU. Use a new `--run-id` for each
new evaluation:

```bash
bench eval --registry-root "$BEHAVIOR_DELIVERY/registry" \
  --results-root "$BEHAVIOR_DELIVERY/results" --run-id behavior-100-seed0 \
  --benchmark behavior_1k --seed 0 --max-concurrent 1 \
  --agent-config configs/agents/embodied/default.yaml \
  --agent-set 'runtime.llm.model="YOUR_VISION_TOOL_MODEL"' \
  --agent-set 'agent.action_timeout_s=900' \
  --timeout-s 900 --case-timeout-s 7200
```

Append `--dry-run` to the same evaluation command to inspect the planned
100 cases without starting case services. These are full-suite reproduction
instructions, not a claim that 100 real task evaluations have been completed.

The native task limit comes from the catalog; Agent turn limits and Runner
timeouts are separate controls. See [Agent configuration](../../docs/agents.md)
and [evaluation](../../docs/evaluation.md) for budgets, artifacts and resume.
Offline checks are described in the [test guide](../../tests/README.md).

The native simulator, robot assets and datasets retain their upstream licenses
and access terms.
