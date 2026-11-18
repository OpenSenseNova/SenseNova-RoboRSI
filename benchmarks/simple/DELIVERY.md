# SIMPLE candidate delivery

> Migration note (2026-11-18): this delivery record was imported from the local
> NewRPent checkout. Its image/source digests identify the original build. They
> do not identify an image built from this repository's shared framework; see
> [VALIDATION.md](VALIDATION.md) for migration checks and rebuild before deployment.

This is a Benchmark HTTP service candidate, not a reproduction of the paper's
historical Astra policy. Source registration PR #86 merged as
`2d06bcf1417f779f68611ab3671cd0e13ee63056` and is included in this branch.
PR #85 must remain Draft until the framework/image review gates are satisfied.

## Published repair candidate

Registration #86 and the facade fix #88 are included from main `e357c6a5`.
The adapter now retries failed cleanup. The fixed candidate was published as
`ghcr.io/opensensenova/sensenova-roborsi-simple:v0.1.1-cu128` from adapter
commit `a1cfbfc802c6df8f1e5db8c5e2c96ec559275255`. Use its immutable reference:

```bash
SIMPLE_IMAGE=ghcr.io/opensensenova/sensenova-roborsi-simple@sha256:4b6f9eb28fa87aebba7a3d46fcf0498753d81a47719dadf30a78c22d1f208213
SIMPLE_SOURCE_SHA256=3833e3e3dc272828e9a7bc5070b65e1735127a7faee6dd07eba68e5ea16a848c
```

Authenticated push, registry manifest digest verification, digest-pinned pull and
standalone runtime source verification passed on the build host. GHCR credentials
were removed after logout. Independent recipient-host acceptance remains pending.
Set both variables from the same build when using the commands below; do not pair
the historical v0.1.0 digest with this source hash. The native 400-row failure-path
acceptance in VALIDATION.md exercised this exact image before publication.

## Historical image identity

**The v0.1.0 image below is historical and lacks the adapter/facade cleanup retry
fixes. Do not treat it as the fixed delivery.** The v0.1.1 export above includes
main `e357c6a5` and the SIMPLE retry fix. Current validation is in VALIDATION.md.
The old commands and identities below are retained for provenance and must use
matching image/source digests when checking that historical image.

Published candidate tag: `ghcr.io/opensensenova/sensenova-roborsi-simple:v0.1.0-cu128`.
The registry manifest hash was verified and this exact reference was pulled
successfully on the build host:

```text
ghcr.io/opensensenova/sensenova-roborsi-simple@sha256:c96dfdea288b31f8b56662f106a48c81c1457c3d815506019d45ef9f89cdefa3
```

Access is through the team's private GHCR package. The local unpacked image is
46.24 GB. An independent recipient-host pull/run has not been verified. The
publication process logged out and removed its temporary credentials; no model
requests were made.

The image was exported in an isolated checkout combining adapter commit
`0573b5a3a0c4c39e1ed7e425357997eb46db8c80` and registration proposal
`a793c8fc00f85503ee748ab28b0036fa4ff1d2f8`, based on main `519e15ac`.
The combined build commit is `1f0efcfad8602cd364854658b2623083c377de43`.
Its registered server-source SHA256 is
`bd14e521269bc2c9df01eefdad913700173737ab4ed374710ac2a80aee029867`.
No shared framework changes are included in adapter PR #85. Registration #86
and facade retry fix #88 are now inherited from main. The native base recipe
is under `native/`.

The image includes Ubuntu 22.04 / Python 3.10 / CUDA 12.8 / Torch 2.7 / Isaac 4.5 /
MuJoCo 3.3.6, native AMO and decoupled WBC dependencies, and the standard HTTP
service. It includes the hash-pinned restored XMoveBendPick task and license.
It excludes Agent/model SDKs, API keys, historical dialogs, private evaluation
outputs, and frozen ICL packages. Scene assets and reset catalogs are mounted
separately. One active case per service/GPU is the supported allocation.

## Prepare the client and materialized assets

Use the normal RoboRSI development installation from `CONTRIBUTING.md`; run the
following commands from the repository root. No model credentials are required.
Follow [ASSETS.md](ASSETS.md) to populate `/data/simple-registry/simple/assets`.
Download the six evaluation archives listed in `robots/simple/sources.json` from
`USC-PSI-Lab/psi-data` at its declared revision. For example:

```bash
hf download USC-PSI-Lab/psi-data --repo-type dataset \
  --revision e78fb93cc28912a3031a10b8656d32d7f0a2b867 \
  --include 'simple-eval/*.zip' --local-dir /data/psi-data
python -m benchmarks.simple.prepare --archives /data/psi-data \
  --tasks xmove bendpick handover tabletop mobile_pick xmove_bendpick \
  --budget-multiplier 1 --output /data/simple-registry/simple/catalog
```

The exporter checks exact bytes for each selected archive and emits all 180
cases. For 10× use `--budget-multiplier 10` and a separate new catalog directory;
no episode or context continuation is implied. Use fresh catalogs with this
version: the informational `state_dim` capability from older exports is omitted
to match the standard Registry schema. The state observation remains 32D.

## Start the pinned service and check one case

Accept the NVIDIA EULA before using `--accept-nvidia-eula`. Set `SIMPLE_IMAGE` to
the published `repository@sha256:digest`; it is intentionally not defaulted to a
mutable tag. Authenticate to GHCR in your terminal if the package is private.
Use the published v0.1.1 image and corresponding source variables above, or the
matching digest and source identity from your own build. The historical v0.1.0
image does not contain this repair.

```bash
python -m benchmarks.simple.serve --image "$SIMPLE_IMAGE" \
  --catalog /data/simple-registry/simple/catalog \
  --assets /data/simple-registry/simple/assets \
  --output /data/new-simple-service-output --gpu 0 --port 8000 \
  --dds-domain 21 --accept-nvidia-eula
```

The command keeps the image and assets read-only, runs as the current UID, binds
only localhost, and uses the new output directory for caches and native outputs.
A separate 4 GB `/tmp` tmpfs supports native OptiX/DDS temporary files.
Use an unoccupied GPU and a distinct port/output/DDS domain per service. Docker
must provide NVIDIA `compute,utility,graphics`; headless still needs graphics
libraries. Host driver 570.195.03 on H200 is the checked reference. H200 images
remain grainy and are not certified equivalent to RTX rendering.

In a second terminal, use the source SHA256 from the image's exported source
manifest (not the image digest):

```bash
python -m benchmarks.simple.smoke_http --endpoint http://127.0.0.1:8000 \
  --task-id xmove_bendpick/L0/E00 --output /data/new-simple-smoke \
  --source-sha256 "$SIMPLE_SOURCE_SHA256"
```

This checks reset, state, RGB, at most three hold rows, downloaded artifacts and
case close/delete. It makes zero model requests, keeps the original task budget,
and records neither an early cancellation nor a hold example as task success.
Ctrl-C the service terminal to stop only its own container.

## Check native termination and scoring without a model

The separate terminal diagnostic holds the reset state's joint/body targets and
zero locomotion commands, in chunks of at most 20 rows. It preserves the selected
task's original 1x budget and runs until native termination or that limit. For
BendPick this is at most 400 rows; no model request, budget override or synthetic
success is used. Keep the service running from the previous section, then run:

```bash
python -m benchmarks.simple.check_terminal --endpoint http://127.0.0.1:8000 \
  --task-id bendpick/L0/E00 --output /data/new-simple-terminal-check \
  --source-sha256 "$SIMPLE_SOURCE_SHA256"
```

This client requires a 1x catalog. It retains progress, observations, actions and
the terminal response; calls `finish`; downloads the native result and action
trace; checks the result hash, physical-row count, first success and binary score;
then closes and deletes its case. A failure score of zero is a valid diagnostic
outcome if it agrees with the native trace. `report.json` distinguishes diagnostic
pass/fail from task success. Successful `close` preserves an already completed
HTTP result state. Use `VALIDATION.md` for the actual native branch exercised.
This client runs outside the image and leaves the published service unchanged.

## Registry and Runner

The existing HTTP Registry discovers and verifies the service's 180 identities:

```python
from roborsi.evaluation.http_benchmark import HttpBenchmarkRegistry
registry = HttpBenchmarkRegistry({"simple": "http://127.0.0.1:8000"})
locked = registry.lock("simple")
print(len(locked.list_tasks()))
```

For actual Agent evaluation, use the standard `bench eval --benchmark-url
simple=http://127.0.0.1:8000 --agent-config robots/simple/agent.yaml` path after
configuring the external Agent. This command makes model requests; it is not
part of the model-free delivery checks. The preset has not been validated as a
complete task policy and has no historical ICL, FK/IK, or custom pro extension.

Generate a filesystem Registry manifest from the same runtime JSON saved by the
smoke (the asset/catalog folders must be inside the manifest's parent directory):

```bash
python -m benchmarks.simple.manifest --image "$SIMPLE_IMAGE" \
  --catalog /data/simple-registry/simple/catalog \
  --assets /data/simple-registry/simple/assets \
  --runtime-info /data/new-simple-smoke/runtime.json \
  --output /data/simple-registry/simple/manifest.json --accept-nvidia-eula
```

```python
from roborsi.benchmarks.registry import BenchmarkRegistry
registry = BenchmarkRegistry(["/data/simple-registry"])
locked = registry.lock("simple")
print(locked.image_reference, len(locked.list_tasks()))
```

The generator hashes the full materialized asset/reset trees, retains task and
seed identity, and rejects a tag without a digest. Changed assets fail verification.
For this published candidate, filesystem discovery/hash verification passed
with all 180 task messages exactly equal to the exported catalog. The digest-pinned
image also passed HTTP discovery of all 180 tasks after pull. These checks used
the build host, with zero model calls.

This filesystem check is separate from Docker-executor native acceptance:
the shared Docker executor currently fixes `/tmp` at 1 GB, while observed Isaac
shader caches alone can exceed 1 GB. Use the independently deployed HTTP service
above. This PR does not change shared executor limits or claim that path passed.

## Validation boundary

- Native checks are bounded interface checks, with no paid model calls.
- The restored task is the original upstream implementation; its hardcoded
  reward >= 0.8 success predicate is preserved, and its source hash is recorded.
- Native failure termination/finish/scoring passed for BendPick L0/E00 at its
  original 400-row limit (343.12 s, 46 downloaded artifacts, zero model calls).
  Successful native task scoring, complete Agent tasks, an independent recipient's
  host, and reproduction of historical paper scores remain unverified.
- The published candidate passed digest-pinned pull, HTTP runtime identity and
  180-task filesystem Registry/asset checks on the build host. Native execution
  coverage and source comparisons are documented in `VALIDATION.md`.
