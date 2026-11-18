# SIMPLE build inputs and external prerequisites

This is a service integration candidate migrated from NewRPent on 2026-11-18.
References to registration #86, facade retry fix #88 and adapter #85 describe
the upstream SenseTime-FVG/RoboRSI history. This repository registers the SIMPLE
bundle in its own shared framework. Imported publication identities and verified
pull scope in [DELIVERY.md](DELIVERY.md) apply to the original builds; rebuild and
verify this checkout independently. A local image ID is not a GHCR repository digest.

## External native base

The [native base recipe](native/README.md) builds the pinned Python 3.10,
Isaac 4.5, CUDA 12.8 and controller stack from public sources. It uses a six-file
build-context allowlist and performs dependency and credential checks. Build it
separately before the shared service export below; it is not the final HTTP image.

Prepare a dedicated Linux native base containing:

- Python 3.10 and NVIDIA Isaac Sim 4.5.0.0, with its accepted EULA;
- MuJoCo 3.3.6 and the pinned SIMPLE installation, including its non-Agent
  simulator/controller dependencies and submodules;
- materialized original AMO checkpoints, or the decoupled WBC dependencies and
  G1 meshes/controller weights required by the selected task;
- graphics userspace compatible with the host driver; headless is not compute-only;
- loadable `libGLU.so.1`, `libvulkan.so.1` and NVIDIA graphics libraries;
- NumPy and Pillow compatible with that native stack.

Use SIMPLE `6d10628794d9c7de4596b4f2afb2c054a637c2bc` from
https://github.com/physical-superintelligence-lab/SIMPLE and initialize the submodule
revisions recorded by it. Follow its installation and asset instructions. Preserve
upstream notices and comply with NVIDIA and controller/asset licenses. Do not copy
third-party binaries, datasets or weights into this source repository.
Run `git lfs install --local` before materializing the checkpoints so Git can
recognize the checkpoint bytes as matching their committed LFS pointers.

The base must **exclude Agent/model SDKs and their credentials**. In particular,
installing this repository's full `roborsi` package or SIMPLE's optional Agent SDK
extras into a Benchmark base does not meet the server boundary. Its native Python
must resolve `git` and the installed native dependencies at runtime. CUDA/Vulkan
mapping, original weights, GPU memory, cleanup and source provenance need real
acceptance evidence; no universal H200 driver recipe is implied.

Record the base digest, operating system, Python/pip freeze, Torch/CUDA,
MuJoCo/Isaac versions, native source/submodule/LFS identities and assets hashes.
`requirements-http.txt` describes wrapper constraints. `requirements-http.lock`
pins the eleven Python 3.10 HTTP dependency packages and distribution hashes,
matching the validated native base. The Dockerfile installs it with
`--require-hashes --no-deps`, checks dependencies, and saves the resolved service
environment at `/opt/simple-build/service-pip-freeze.txt`. This HTTP lock does not
replace the native source, binary, system-library and GPU-driver records.

## Catalog and asset mounts

Prepare catalogs with `python -m benchmarks.simple.prepare` using the source-pinned
archives. Keep the output read-only inside the native Benchmark service and mount
assets separately. This command exports task instructions and reset metadata, not
training demonstrations or a ready simulator installation. It never evaluates a
Policy. Catalog files include per-reset hashes and source revisions.

Use the same selected asset/controller/graphics stack for compared settings and
record it. Only compare 1× and 10× outcomes with the declared episode identities and
budget rules; a failure rerun is a fresh physical trajectory, not simulator resume.
No historical skill or outcome is inferred from filenames.

## Shared export after framework registration

Registration [PR #86](https://github.com/SenseTime-FVG/RoboRSI/pull/86) is merged
and present in this branch. Use the existing exporter; no bespoke copy/package
implementation is introduced here:

```bash
python -m benchmarks.package --benchmark simple --output /data/fresh-simple-context
# Choose an actual native base, already built and pinned by digest.
docker build /data/fresh-simple-context \
  --build-arg SIMPLE_RUNTIME_IMAGE=YOUR_NATIVE_REPOSITORY@sha256:YOUR_BASE_DIGEST \
  --tag YOUR_SIMPLE_IMAGE_TAG
```

Those placeholders are rebuild inputs, not image identities. Review `source.json`
and `build.json`, run native checks, publish, verify authenticated digest-pinned
pull, and generate the Registry manifest from the actual service runtime. Record
build-host verification separately from an independent recipient-host check.
The historical v0.1.0 candidate was built in an isolated checkout before #86
merged. It lacks the current adapter/facade cleanup fixes. Rebuild from this
branch and publish a new version; do not reuse the historical source digest.

The Dockerfile copies only the shared exported `benchmark-runtime/` and declared
HTTP dependency lock. `verify_benchmark_runtime` rejects inherited Agent SDKs and
undeclared files. Agent prompts/config are never part of that server source bundle.

For a non-root service, the worker scopes Git trust to the explicitly configured
native checkout and uses its own episode directory for LFS scratch files. It does
not require forwarding arbitrary `GIT_CONFIG_*` values or trusting every directory.
Pinned revision and tracked-source checks still run before simulator startup.

The model-free `benchmarks.simple.smoke_http` client verifies the exported source
identity, reset, at most three action rows, downloaded media, cancellation and
case deletion. See the README command. It never reduces the official task budget
or forces a terminal score. Native `finish` is exercised only if the environment
actually terminates within those rows; CPU HTTP tests separately cover both
terminal and nonterminal lifecycles. A short check is not full task acceptance.

## GHCR publication

The team-assigned package is
`ghcr.io/opensensenova/sensenova-roborsi-simple`. It is private by default. The
existing placeholder does not constitute a runnable SIMPLE release. Refer to
[DELIVERY.md](DELIVERY.md) for the candidate version and verified publication
identity. Pending framework and native-scoring gates are stated there.

Before publishing, accept the OpenSenseNova organization invitation, join the
`roborsi-maintainers` team, and confirm Package Write permission. Authenticate
with your own PAT (classic) with `write:packages`, and authorize it for SSO if
the organization requires it. Enter the PAT at the Docker password prompt;
never put it in a Dockerfile, build argument, source file or committed log.
GitHub CLI login does not by itself establish Docker/GHCR authentication.

On a machine with a working Docker daemon and the validated local image:

```bash
# Use your GitHub username; enter your PAT at the password prompt.
docker login ghcr.io -u YOUR_GITHUB_USERNAME

# Example only: confirm the final image uses CUDA 12.8 before selecting cu128.
# Verify this release tag has not already been published; never overwrite it.
SIMPLE_RELEASE_IMAGE=ghcr.io/opensensenova/sensenova-roborsi-simple:v1.0.0-cu128
docker tag YOUR_VALIDATED_LOCAL_IMAGE:TAG "$SIMPLE_RELEASE_IMAGE"
docker push "$SIMPLE_RELEASE_IMAGE"
docker pull "$SIMPLE_RELEASE_IMAGE"
docker image inspect "$SIMPLE_RELEASE_IMAGE" --format '{{json .RepoDigests}}'
```

Release tags follow `v<major>.<minor>.<patch>-<environment>` and are immutable by
team convention. Use the environment of the final built image, not the CUDA
version reported as supported by the host driver. The candidate native stack
uses Torch `2.7.0+cu128`; preserve the version and digest in build provenance.

Save the tag, registry digest from the successful push, build-source commit,
base-image digest, dependency lock and native acceptance evidence. Have a regular
maintainer verify authenticated upload/pull access, and verify a recipient can
pull and run the image by `ghcr.io/opensensenova/sensenova-roborsi-simple@sha256:…`.
Put that actual digest in the delivery table, README and PR. Do not use a local
image ID or placeholder digest. A successful push/pull alone does not establish
native rendering, task/controller support or reproduction of historical scores.

If access is denied, check the Docker login account, team membership, Package
Write permission, PAT scope and SSO authorization. Keep catalogs/assets mounted
as documented above and Agent credentials outside the Benchmark image.

## Agent package

The Agent is installed separately with the existing wheel exporter:

```bash
python -m roborsi.agents.package --config robots/simple/agent.yaml \
  --name simple_baseline --output /data/fresh-simple-agent
python -m build --wheel /data/fresh-simple-agent
```

The destination must be outside the source checkout and new. Supply credentials
at runtime on the Agent side. The model placeholder must be replaced explicitly.
Loading/exporting this preset does not make an API request or reproduce historical
max/pro, context, ICL or numerical tool settings.

## Restored task and candidate builds

The final wrapper additionally downloads `XMoveBendPickTeleop` from its last
upstream task revision using `install_compat.py` and `sources.json`. It preserves
the original bytes and MIT license outside `/opt/simple`; native tracked-source
verification remains enabled. For a host installation run the same installer
with `--sources robots/simple/sources.json --output /data/simple-compat`, then
set `ROBORSI_SIMPLE_COMPAT_ROOT=/data/simple-compat` in the Benchmark environment.

Export directly from this checkout and record the exact source commit and
exported hash. The migrated shared framework requires a new image; never
overwrite the historical version. A release must be rebuilt/rechecked if
service source hashes change.

The candidate includes a standard Registry manifest generator
`python -m benchmarks.simple.manifest`. It takes the actual pullable image digest,
service runtime JSON, and materialized assets under the manifest parent. It
rejects schema-incompatible old catalogs; re-export them with `prepare`.
`state_dim` remains an observation property, not a manifest capability.
The score protocol remains in each task's native metadata, separate from the
four standard evaluation fields in the service/manifest.

Use `serve` and the existing HTTP Registry/Runner path for native execution.
The framework's default Docker executor provides only 1 GB of `/tmp`; observed
Isaac shader caches exceed that size. It also allocates GPUs by count. No shared
executor is patched in this PR; filesystem Registry validation and a passing
external HTTP check must not be represented as default Docker-executor acceptance.
