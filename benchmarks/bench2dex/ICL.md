# Download the frozen Bench2Dex ICL release

> Source: the same path at upstream `8d9a0af66422e1faad7696281a76819cd165eb44`. Original tools, versions and workflow are preserved below. Historical release/validation statements are not results of this port. See the [source audit](../../docs/icl-coverage.md).

[中文](ICL.zh-CN.md). [Pinned Hugging Face release](https://huggingface.co/datasets/sensenova/SenseNova-RoboRSI-Packages/tree/861d045d47466673826d0e9c6374fa0dba944096/bench2dex/v1.0.0).

Package: `bench2dex/v1.0.0`, regime `task_icl_v3`, 26 tasks, 1.194 GB of ZIPs.
HF commit: `861d045d47466673826d0e9c6374fa0dba944096`.
All 33 original release files are preserved byte for byte, including the 26 ZIPs
and their 3,610 payload files. No ICL generation is required. This is separate
from the official `Bench2Dex/Assets` runtime dataset and the GHCR native image.

Install `huggingface_hub` and authenticate with `hf auth login` using an account
with access; the dataset is currently private. From the RoboRSI checkout, run:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="sensenova/SenseNova-RoboRSI-Packages",
    repo_type="dataset",
    revision="861d045d47466673826d0e9c6374fa0dba944096",
    allow_patterns=["bench2dex/v1.0.0/*", "scripts/verify_package.py", "packages.json"],
    local_dir="work/roborsi-packages",
    token=True,
)
```

Verify the full release and stage a task Agent:

```bash
python work/roborsi-packages/scripts/verify_package.py \
  work/roborsi-packages/bench2dex/v1.0.0 \
  --manifest-sha256 e537864c8744b1918ba1fd3571915bafa39fd2af5d45f66155973eac4c507150
python -m benchmarks.bench2dex.agent_bundle --task 26 \
  --source work/roborsi-packages/bench2dex/v1.0.0 --output work/task26-agent
```

Pass the **version root** above to `run_suite --icl-source` too. The repository
root does not directly contain `packages/`, so the standalone `--repo-id`
download mode is unsuitable for this aggregate layout. The Agent independently
verifies the selected ZIP and every payload against the checked-in catalog.
Use a fresh output directory. For one task only, `allow_patterns` can select
`bench2dex/v1.0.0/packages/task_26.zip`; Agent verification still applies, but the
full-release verifier requires every release file.

Record this full HF SHA, version path, evaluation code commit and image digest
with the run. Preserve the ZIP bytes. The package's archived README describes
the standalone evaluator; its `RELEASE.md` describes the shared dataset layout.
The original `LICENSE_NOTICE.md` records licensing as pending; this private
upload does not resolve owner or third-party redistribution permissions.

Publication validation: fresh download by the SHA above matched all 35 uploaded
files, 26 task manifests and 3,610 payload hashes. Task-26 Agent staging from that
download also passed. No new simulator or Astra run was performed for upload;
see [validation limits](https://github.com/SenseTime-FVG/RoboRSI/blob/8d9a0af66422e1faad7696281a76819cd165eb44/benchmarks/bench2dex/VALIDATION.md) and [framework prerequisites](https://github.com/SenseTime-FVG/RoboRSI/blob/8d9a0af66422e1faad7696281a76819cd165eb44/benchmarks/bench2dex/FRAMEWORK_REGISTRATION.md).
