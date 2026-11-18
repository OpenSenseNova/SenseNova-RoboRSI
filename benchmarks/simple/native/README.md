# Native SIMPLE base

This recipe builds the simulator and controller dependencies independently of the
shared RoboRSI server source exporter. It is a native base, not the final Benchmark
HTTP image. Final service export still requires `../FRAMEWORK_REQUEST.md`.

Build from this directory, with Docker GPU support already installed on the host:

```bash
docker build --progress=plain -t simple-native:isaac45-cu128 .
```

The explicit Docker ignore file admits six recipe files. Sources are fetched over
public HTTPS at pinned commits; no home directory, SSH agent, model credential,
experiment log or filled environment file is supplied to the build. Scene assets,
evaluation reset catalogs and frozen ICL references are separate artifacts. ICL
references belong on the Agent side, not in this simulator image.

The base uses Ubuntu 22.04, Python 3.10, CUDA 12.8, Torch 2.7, Isaac Sim 4.5,
MuJoCo 3.3.6 and the original AMO/decoupled WBC sources. cuRobo is compiled for
compute capabilities 8.9 and 9.0 with PTX for the latter. The source revisions
match the declared SIMPLE commit and its selected submodules. It does not build
the separate C++ SONIC deployment binary or install real-robot GUI/teleop extras.

Build completion runs `pip check`, rejects installed model SDKs and known
credential locations, and records resolved Python/system versions and source
revisions under `/opt/simple-build/`. These records identify the build; the
top-level requirements are not a hash lock for every external artifact.

At runtime, supply the host NVIDIA runtime with `compute,utility,graphics`
capabilities, mount scene assets read-only, and set `OMNI_KIT_ACCEPT_EULA=Y` only
after accepting the NVIDIA license. Keep inference outside the simulator. A
successful image build does not establish renderer compatibility, task success,
historical Astra score reproduction or completed Registry acceptance. Record
native reset/RGB/action evidence for the final chosen GPU and mounted asset set.

For a non-root runtime user, provide a writable `HOME`, output directory and Git
LFS cache. The source check runs Git's LFS clean filter, which needs temporary
storage even when the source is read-only. Set these container environment values
with `/output` owned by the runtime UID:

```text
GIT_CONFIG_COUNT=2
GIT_CONFIG_KEY_0=safe.directory
GIT_CONFIG_VALUE_0=/opt/simple
GIT_CONFIG_KEY_1=lfs.storage
GIT_CONFIG_VALUE_1=/output/lfs-cache
```

Keep the source integrity check enabled. The Docker daemon must also be able to
resolve each host mount path. With root-squashed network storage, stage the needed
assets into a dedicated local directory if necessary; do not expose private home
directories or credentials to solve an asset-mount permission problem.
