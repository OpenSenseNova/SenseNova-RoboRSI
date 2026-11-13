# DexJoCo ICL sources and task Agent

[简体中文](ICL.zh-CN.md) · [Benchmark setup](README.md) · [Source audit](../../docs/icl-coverage.md)

These entries come from upstream commit
`8d9a0af66422e1faad7696281a76819cd165eb44`. The original source ZIP is preserved
byte for byte at `robots/dexjoco/agents/baseline/resources/reference-code.zip`.
Its SHA-256 is `e883be2bf8f22674b24e56758721f754cce6353a45044f32025e26be7e870674`.
No new learning algorithm or prompt is introduced by this port.

## Use the frozen task packages

The original benchmark README pins the following aggregate HF snapshot. Use
your existing HF authentication if access requires it:

```python
from pathlib import Path
from huggingface_hub import snapshot_download

download = snapshot_download(
    repo_id="sensenova/SenseNova-RoboRSI-Packages", repo_type="dataset",
    revision="8b81a9d779230705d4ebc427a3422073b157b02d",
    allow_patterns=["dexjoco/v1.0.0/*"], local_dir="../roborsi-packages",
)
print(Path(download) / "dexjoco/v1.0.0")
```

Set `KNOWLEDGE_ROOT` to the printed directory containing `rand_full/`, and use
the pinned native checkout from the benchmark guide:

```bash
python -m benchmarks.dexjoco.prepare_agent_assets \
  --upstream "$ROBORSI_DEXJOCO_UPSTREAM" \
  --knowledge "$KNOWLEDGE_ROOT" --output ../dexjoco-agent-assets
export ROBORSI_DEXJOCO_AGENT_ASSETS="$(realpath ../dexjoco-agent-assets)"
```

This original preparation helper verifies all task packages against the archived
catalog and copies the public robot references. The original compatibility Agent
is selected with `--agent-config robots/dexjoco/agent.yaml`; the alias
`configs/agents/dexjoco/compat.yaml` selects the same preset.

The Agent keeps the original Astra/max, Codex CLI 0.154.0, external file tools
and fresh-session 1x→10x continuation. Active prompts are ordinary declared files
under `robots/dexjoco/agents/baseline/prompts/`; their baseline bytes are checked
against the source ZIP by the original tests. The Agent runtime and numerical
policy environment are separate. The original locks and setup notes are in
[`agents/dexjoco_compat/`](../../agents/dexjoco_compat/README.md). Runtime inputs are:

```bash
export ROBORSI_DEXJOCO_POLICY_PYTHON=/absolute/path/to/separate-policy-venv
export ROBORSI_DEXJOCO_CODEX=/absolute/path/to/native/codex
export ROBORSI_DEXJOCO_AUTH=/absolute/path/to/your/auth.json
```

Use the current Benchmark service/registry instructions in [README.md](README.md),
then select the task Agent in the same `bench eval` command. These commands make
model calls only when an evaluation is started without `--dry-run`.

## Original demonstration-to-ICL implementation

The source ZIP also contains the construction code, rather than only a package
loader:

- `dexjoco-eval/scripts/download_dexjoco_demos.py`: pinned demonstration download
  and task inventory verification.
- `dexjoco-eval/src/gpt6_eval/dexjoco_reader.py`: verified on-demand demo reader.
- `dexjoco-eval/src/gpt6_eval/dexjoco_icl.py`: isolated preparation and sealing.
- `dexjoco-eval/prompts/dexjoco_preparation.md`: original learning instructions.

Extract the preserved archive to a new directory. The archived project declares
Python `>=3.10,<3.12` and its own pinned dependencies; use that environment for
construction, following its README and `pyproject.toml`:

```bash
python -m zipfile -e robots/dexjoco/agents/baseline/resources/reference-code.zip ../dexjoco-study
cd ../dexjoco-study/dexjoco-eval
python -m pip install -e .
python scripts/download_dexjoco_demos.py --help
python -m gpt6_eval.dexjoco_icl --help
```

The original parser supports the following one-task route after installing the
recorded Codex CLI, configuring authentication and preparing the pinned upstream:

```bash
python scripts/download_dexjoco_demos.py \
  --task water_plant --regime rand_full --output "$DEXJOCO_DEMOS"
python -m gpt6_eval.dexjoco_icl \
  --config configs/dexjoco_icl_5ep_rand_full_robot_v2_two_session.json \
  --upstream "$ROBORSI_DEXJOCO_UPSTREAM" --task water_plant \
  --dataset "$DEXJOCO_DEMOS" --output "$DEXJOCO_PREPARATION" \
  --auth "$ROBORSI_DEXJOCO_AUTH"
```

Preparation requires a new output directory. Original code validates the training
inventory, serves the selected demonstrations, requires readiness plus `TASK.md`
and `SOURCES.md`, and seals `frozen/` with provenance and model evidence. This
starts a real model preparation. Its output is a new experiment: the shared
compatibility Agent still accepts the original published manifest catalog.
Evaluate newly generated packages through the archived evaluator's declared
`--package` and `--package-sha256` inputs; do not substitute them for frozen
release hashes.

The source and locks retain their original notices. Offline tests of this port
do not establish successful learning or a native task score.
