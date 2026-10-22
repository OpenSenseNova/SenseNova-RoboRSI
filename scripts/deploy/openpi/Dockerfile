# Same CUDA runtime as the pinned upstream OpenPI serving Dockerfile.
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
ARG OPENPI_REVISION=215abfb217dbac7d5f1273282331b9b1866c0479
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs ca-certificates build-essential clang curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/
WORKDIR /opt/openpi
ENV GIT_LFS_SKIP_SMUDGE=1
RUN git init . && git remote add origin https://github.com/Physical-Intelligence/openpi.git \
    && git fetch --depth 1 origin ${OPENPI_REVISION} && git checkout --detach FETCH_HEAD \
    && git submodule update --init --recursive
ENV UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 OPENPI_DATA_HOME=/cache \
    XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
RUN uv venv --python 3.11.9 .venv && uv sync --frozen --no-dev
# Upstream requires these patches for its optional PyTorch checkpoint loader.
RUN .venv/bin/python -c "import pathlib,shutil,transformers; shutil.copytree('src/openpi/models_pytorch/transformers_replace',pathlib.Path(transformers.__file__).parent,dirs_exist_ok=True)"
COPY robots/libero/vla_server.py /app/vla_server.py
COPY scripts/evaluation/downloads.py /app/scripts/evaluation/downloads.py
ENTRYPOINT ["/opt/openpi/.venv/bin/python", "/app/vla_server.py"]
