# Tests

The release tests cover the shared Agent/runtime/evaluation contracts and the
LIBERO-family, Bench2Dex, DexJoCo, BEHAVIOR-1K and RoboCasa365 adapters. Tests mirror their
production modules under `unit_tests/roborsi/` and `unit_tests/robots/<benchmark>/`. Shared
local services and fake native providers remain beside their tests or in
`fixtures/`.

## Offline checks

Install from the repository root, then run:

```bash
python -m pip install -e '.[runtime,test]'
pytest tests/unit_tests -q
```

Required CI uses Linux and Python 3.11/3.12. These tests do not contact model
providers, download assets, start Docker or execute a native simulator. They
exercise real local HTTP and child Python processes where appropriate. Package
installation tests build local wheels without network dependency resolution.

Focused checks:

```bash
pytest tests/unit_tests/roborsi/evaluation tests/unit_tests/roborsi/agents -q
pytest tests/unit_tests/roborsi/benchmarks tests/unit_tests/robots/libero -q
pytest tests/unit_tests/robots/bench2dex -q
pytest tests/unit_tests/robots/dexjoco -q
pytest tests/unit_tests/roborsi/runtime tests/unit_tests/roborsi/planner -q
pytest tests/unit_tests/robots/behavior_1k -q
pytest tests/unit_tests/roborsi/components/icl tests/unit_tests/roborsi/cli/test_icl.py -q
pytest tests/unit_tests/robots/robocasa tests/unit_tests/roborsi/benchmarks/test_robocasa365_registration.py -q
```

Benchmark bundle checks verify that exported server imports and CLI help work
without Agent code or model SDKs. Adapter tests use deterministic native
facades to check actions, stopping, score preservation and artifact identity.
These checks establish contracts, not native task success.

## LIBERO CPU evaluation check

Prepare Docker, scene assets and a LIBERO registry using the
[runtime recipe](../benchmarks/libero/README.md). For the short transport and
failure tests, generate a manifest with task 0, seed 0 and `--max-steps 3`.

```bash
ROBORSI_RUN_LIBERO_BENCHMARK=1 \
  ROBORSI_LIBERO_REGISTRY=/absolute/path/to/registry \
  pytest tests/integration_tests/robots/libero/test_roborsi.py \
    tests/integration_tests/robots/libero/test_roborsi_http.py \
    tests/integration_tests/robots/libero/test_roborsi_http_replay.py \
    tests/integration_tests/robots/libero/test_roborsi_action_contract.py -v -s
```

These opt-in checks execute the real native runtime, compare fixed actions
against direct LIBERO calls, verify camera/score artifacts and check cleanup.
The fixed-action Agent is expected to fail the manipulation task. A passing
test does not establish model performance.

Other opt-in files cover failures, queryable case status, Agent isolation,
dependency verification and batch resume. Batch resume needs tasks 0–3 at seed
0, each with a three-step budget. Dependency/environment checks need a complete
wheel directory selected with `ROBORSI_AGENT_WHEELS`. Read each test's module
docstring for its prerequisites. An unset opt-in flag causes a skip; collection
or a skip is not native validation.

PRO, Long++ and Ultra-Long have offline adapter and runtime-export coverage.
Their native runs require the assets and image prerequisites in their benchmark
recipes.

## BEHAVIOR-1K checks

Offline tests cover the 21D action contract, RGB-D conversion, task catalogs,
native metric handling, isolated worker cleanup and exported build inputs.
ICL checks cover deterministic package construction, pinned retrieval, writable
copies and snapshots, read-only context tools, CLI receipts, learned-only export,
task binding and learning cleanup. BEHAVIOR tests additionally check actual
training instance selection, offline FK layout, RGB time bounds and video hash
rejection. The [ICL guide](../docs/icl.md) includes a synthetic offline example.
Follow the [benchmark guide](../benchmarks/behavior_1k/README.md) to build an
image and prepare licensed assets. Catalog generation performs a native reset
for each selected task and records its instruction and step limit. Native
evaluation needs the image, RTX GPU, asset snapshot and generated registry.
Record the actual image digest and native score alongside the run artifacts.

## Bench2Dex

Restored upstream ICL tests cover package mutation, suite receipts, exported
Agent inputs and the original action-file protocol. The external-session tests
also exercise a real offline CLI subprocess on Linux, including cancellation
and process cleanup. See the [source audit](../docs/icl-coverage.md).

The focused unit command above checks this adapter offline. Native execution
requires the separate runtime and asset preparation described in
[Bench2Dex](../benchmarks/bench2dex/README.md). Passing unit tests, building a source
bundle or collecting tests does not establish a successful Docker build, GPU
run or native task result.

## DexJoCo

Restored upstream checks cover frozen policy configuration, two-session
lifecycle and persistent CLI process ownership across request threads.

The focused unit command above checks this adapter offline. Native execution
requires the separate runtime and asset preparation described in
[DexJoCo](../benchmarks/dexjoco/README.md). Passing unit tests, building a source
bundle or collecting tests does not establish a successful Docker build, GPU
run or native task result.

## RoboCasa365 checks

Offline checks cover the Target50 task catalog, native action contracts, scoring,
episode RPC cleanup, shared RLDX service ownership and benchmark exports. The
[benchmark guide](../benchmarks/robocasa365/README.md) and
[build guide](../benchmarks/robocasa365/BUILD.md) describe image access, kitchen
assets and RLDX checkpoint requirements. Native evaluation needs the pinned
environment and an NVIDIA GPU. Offline tests do not establish native task success.

## Adding tests

Use existing fixtures and test user-visible behavior. Keep required tests
offline and independent of a simulator or GPU. Never weaken a failing test to
make a change pass. Report the exact checks run and distinguish fakes, native
execution and real model calls.
