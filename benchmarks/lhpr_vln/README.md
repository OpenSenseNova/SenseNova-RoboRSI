# LHPR-VLN delivery recipe

Runtime: Python 3.9.23 / Habitat-Sim 0.3.0. The independent native environment
comes from AgentCanvas commit `6c0a17d3830be5fa9c64e8dec35be29808581f2d`,
directory `workspace/nodesets/env/env_lhvln`. It is loaded as an isolated native
package; the AgentCanvas agent harness is not imported.

Preserved protocol: annotated pose; effective seed `42 + original_row_index`;
strict 1 m fixed-object success with same-region-name matching; three RGB views;
512 rendering, contrast 1.5 and resize 366; forward 0.25 m, turn 30 degrees;
force escape after three blocked actions, counting all recovery actions toward
the physical budget. STOP submits a subtask and may advance to the next one.

The test binding uses the original native runtime. The training bundle uses
the same implementation with the explicit scene-path patch in
`scenes-paths.patch`. Apply it from the `env_lhvln` directory with `patch -p1`.
The patch adds independent scene/config paths; it does not change scoring or
simulation. Source and resulting hashes are recorded in `provenance.json`.
`native-environment.json` is the validated target package inventory.

## Agent entry

Use `bench eval --agent-config robots/lhpr_vln/agent.yaml` with the Registry and
task selection arguments. This entry inherits
`robots/lhpr_vln/agents/baseline/agent.yaml` and the common
[navigation baseline](../../robots/vln_shared/agents/baseline/README.md).
The factory, prompts and tool policy are shared with VLNCE. Dependencies stay
in `agents/vln_shared/requirements.txt`.

The standard entry starts without frozen training memory. For the shared
exploration and conditional 500/5000 experiment, follow the
[run guide](../../docs/vln_shared_evaluation.md) and use the generated dataset
preset or its exported wheel. The one-click launcher selects this standard
entry before attaching the dataset's exploration snapshot.

## Image and asset delivery

The Dockerfile has the same two-interpreter base-image contract as
[VLNCE](../vlnce/README.md). Use a digest-pinned LHPR base with the recorded
native packages at `/opt/native/bin/python` and Python 3.11 at
`/opt/roborsi-service/bin/python`. Public runtime-source registration includes
`lhpr_vln`, its service factory/native adapter and the four shared modules listed
in the VLNCE guide. Agent, exploration, campaign code and the VLNCE adapter are
excluded from the service export. The exported service and manifest use the same
registered source digest as `adapter_version`.

```bash
python -m benchmarks.package --benchmark lhpr_vln --output /tmp/lhpr-context
docker build --build-arg NATIVE_RUNTIME_IMAGE=REGISTRY/IMAGE@sha256:ACTUAL_DIGEST \
  -t YOUR_LHPR_IMAGE /tmp/lhpr-context
```

Use the released native base by its immutable digest. A build recipe alone is
not acceptance evidence: verify the exported source, native environment, GPU
reset/step/finish and Registry deployment before publishing a benchmark digest.

Prepare `campaign.json` using the [run guide](../../docs/vln_shared_evaluation.zh-CN.md).
For test assets, a path-map JSON maps the three complete source directories to
separate relative directories:

```json
{
  "/path/to/AgentCanvas/workspace/nodesets/env/env_lhvln": "native/lhpr",
  "/path/to/EASI-private/datasets/oscarqjh_LHPR-VLN_easi": "data",
  "/path/to/SenseNova-Nav/vln_data_gen/outputs/scenes": "regions"
}
```

```bash
python -m robots.vln_shared.assets stage --config /path/to/campaign.json \
  --dataset lhpr_vln --mapping /path/to/path-map.json --output /delivery/lhpr/assets
python -m robots.vln_shared.assets manifest --assets /delivery/lhpr/assets \
  --image YOUR_LHPR_IMAGE@sha256:ACTUAL_DIGEST --output /delivery/lhpr/manifest.json
```

The asset copier may require substantial storage. It follows source links into
regular copies, checks pinned source/annotation hashes, and rebases absolute
paths in the copied Habitat scene configuration. The sealed output uses the
standard full-directory hash and mounts at `/roborsi/assets/lhpr_vln`. Do not
publish native code or HM3D data without the corresponding distribution rights.
Training export uses `--split train` with a map covering its patched runtime,
training annotations, shared HM3D and regions. Perform native parity checks in
the image after relocation; local path parity does not replace that gate.

The submitted code uses the common Provider/Adapter, HTTP, media and scoring
contracts. Local native tests cover both train and test paths, including a
nonzero original index. Full new model results remain distinct from historical
post-hoc 500/5000 filtering.

## Native dependency constraints

The delivery pins NumPy to `1.23.5` to satisfy Habitat-Sim `0.3.0`'s
`numpy<1.24` requirement. `native-fixes.txt` records the wheel hash, and
`Dockerfile.native` applies it to the pinned exported source environment.
Both the native base and benchmark build must pass `pip check`.
