# ICL source coverage

[简体中文](icl-coverage.zh-CN.md) · [Package tools](icl.md)

This audit compares the old `SenseTime-FVG/RoboRSI` repository at
[`8d9a0af66422e1faad7696281a76819cd165eb44`](https://github.com/SenseTime-FVG/RoboRSI/tree/8d9a0af66422e1faad7696281a76819cd165eb44)
with this release. The saved local `NewRPent/origin/main` and authenticated
upstream `main` matched that commit during the audit. Entries below refer to
observed source files, including files inside the two preserved source ZIPs.
[The machine-readable manifest](icl-port-sources.json) records source hashes and
integration changes. Generated demonstration/ICL datasets are downloaded separately.

## Supported benchmarks

| Benchmark | Observed original source | Result |
| --- | --- | --- |
| BEHAVIOR-1K | `benchmarks/behavior_1k/materials.py`, `official_demonstrations.py`, `r1pro_fk.py`; `robots/behavior_1k/agents/baseline/icl.py`, `icl_prompt.md`, tools and presets | Restored material conversion, learning, frozen download and task handoff. [Guide](../benchmarks/behavior_1k/README.md#icl-and-evaluation). |
| Bench2Dex | `agents/bench2dex_compat/reference-code.zip`; its `src/gpt6_eval/{icl,icl_workflow,demo_reader}.py`, `prompts/preparation*.md`; `benchmarks/bench2dex/{agent_bundle,run_suite}.py`; original Agent and catalog | Restored original source archive, task package staging, Agent, prompts and suite. [Frozen packages](../benchmarks/bench2dex/ICL.md); [original preparation guide](bench2dex-icl.md). |
| DexJoCo | `robots/dexjoco/agents/baseline/resources/reference-code.zip`; its `src/gpt6_eval/dexjoco_icl.py`, `dexjoco_reader.py`, `scripts/download_dexjoco_demos.py`, `prompts/dexjoco_preparation.md`; `benchmarks/dexjoco/prepare_agent_assets.py` | Restored original source archive, asset preparation, prompts and two-session Agent. [Guide](../benchmarks/dexjoco/ICL.md). |
| LIBERO | Old exploration/evaluation prompts and `robots/libero/flash/{generate,relations}.py` | Upstream has no released standard-LIBERO ICL corpus or sealed-package learning pipeline to restore. The separate original Flash source-plan generator is restored below. |
| LIBERO-PRO | `scripts/libero_three/prepare_context.py`; original PRO context map | Existing frozen loading and image extraction remain. Restored local family-layout preparation for 40 packages mapped to 80 cases. |
| LIBERO Long++ | Same preparer; original Long++ context map | Existing loader remains; restored local `longpp/task-XX/sealed` preparation for 12 packages. |
| LIBERO Ultra-Long | Same preparer; original Ultra-Long context map | Existing loader remains; restored local `ultra/task-XX/sealed` preparation for 9 packages. |
| RoboCasa365 | `robots/robocasa/agents/baseline/{factory.py,agent.yaml}` and original benchmark guide | Current `model_eval_agent.py`, `configs/agents/robocasa365/icl.yaml` and evaluation bootstrap/download helpers already cover same-task seed-0 frozen experience. No original package-generation script was found. |
| RoboLab | `robots/robolab/agents/{baseline,historical}/` | Current Agent code, presets and prompts match the old source. Its historical route explicitly excludes exploration; no missing ICL construction implementation was found. |
| LHPR-VLN | `robots/vln_shared/{exploration,campaign,storage,execution,prepare,cli}.py`, baseline prompt/factory and `scripts/run_vln_shared.sh` | Learning, freeze, import and conditional evaluation code already match the old source. [Guide](vln_shared_evaluation.md). |
| VLN-CE: R2R / RxR | Same shared sources plus `robots/vlnce/agents/baseline/agent.yaml` | Both dataset conditions already use the original shared learning/frozen-snapshot workflow. |

## LIBERO local material preparation

The original preparer consumes **already-created sealed families**, not raw
demonstrations. Use a new destination outside the source tree:

```bash
python scripts/libero_three/prepare_context.py \
  --benchmark libero_pro --source /path/to/frozen-families \
  --destination /path/to/run/icl
```

Other supported values are `libero_long_plus_plus` and `libero_ultra_long`.
The output contains `packages/` and `package-map.json`. PRO's task/swap aliases
share the same verified family package. Existing published-image extraction
remains available through the current LIBERO evaluation preparation flow.

The separate Flash generator takes one successful audit and its primitive recipe:

```bash
python -m robots.libero.flash.generate \
  --audit /path/to/goal_swap_t3_s7.json \
  --recipe /path/to/goal_swap_t3_s7_recipe.jsonl \
  --destination /path/to/new-plan
```

Its original `--segments` option accepts measured segment readings. It produces
the original plan/anchor materials; this port does not restore the retired Flash
replay runner or present this output as a sealed ICL package. Only the original
self-contained generator and relations code are included.

## Other old-repository entries

The old repository also contains RoboDojo, RoboTwin and SIMPLE. They are outside
this release's registered benchmark set. RoboDojo has EEF prior/recipe consumption
and training-export code; RoboTwin has exploration prompts/tooling without a
sealed ICL publisher; SIMPLE explicitly excludes historical frozen ICL. Their
native runtimes are not added by this construction-source port.

The old `icl_vla_factory` / reviewed VLA policy examples consume existing sealed
ICL and require old VLA components. They are synthetic composition examples,
not missing benchmark ICL constructors, and are not claimed as code parity.

## Compatibility changes

Original package hashes, source ZIP bytes, model-facing prompts and training
entrypoints are retained. Bench2Dex's ICL-only contract functions and inventory
function are moved into Agent-only modules so the current native Benchmark
contracts and RGB encoder remain unchanged. Presets declare those moved inputs.
Its full original model-bearing protocol is preserved as `icl_protocol.json`;
camera paths are checked against the current case-scoped `MediaRef` before
reconstructing the original file layout and validating the unchanged PNG bytes.
The original `roborsi/embodied_agent.py` external CLI session and cancellation
support is also restored, retaining the current invalid-action retry setting.
The first BEHAVIOR port additionally closes the Agent before sealing, rejects
nested copy paths and handles launcher permissions/global flags. These changes
are listed as adaptations rather than attributed to the original implementation.

The guides distinguish historical upstream validation from checks performed on
this port. No new learning algorithm, historical score or native success result
is inferred from the existence of these files.
