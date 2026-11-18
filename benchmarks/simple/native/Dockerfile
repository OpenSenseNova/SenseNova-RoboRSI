# Native simulator only. Build from this directory, never from a user's home.
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04@sha256:54f18e2a8e1b3d03f77b9a6dc905533da46ac93a5513f10e8ba8e560db9fa5ab
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/simple-venv PATH=/opt/simple-venv/bin:${PATH}
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    git git-lfs ca-certificates curl build-essential cmake ninja-build pkg-config \
    libssl-dev uuid-dev libgl1 libegl1 libglx0 libglu1-mesa libvulkan1 vulkan-tools \
    libx11-6 libxext6 libxrandr2 libxi6 libxt6 libsm6 libice6 libxkbcommon0 \
    libglib2.0-0 libnss3 libasound2 libgomp1 ffmpeg patchelf \
    && rm -rf /var/lib/apt/lists/* \
    && python3.10 -m venv /opt/simple-venv
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install pip==25.1.1 setuptools==80.9.0 wheel==0.45.1
COPY constraints.txt requirements.txt /opt/simple-build/
ENV PIP_CONSTRAINT=/opt/simple-build/constraints.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install 'isaacsim[all,extscache]==4.5.0.0' \
    --extra-index-url https://pypi.nvidia.com
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r /opt/simple-build/requirements.txt
COPY fetch_sources.sh /opt/simple-build/
RUN bash /opt/simple-build/fetch_sources.sh
ENV CYCLONEDDS_HOME=/opt/cyclonedds LD_LIBRARY_PATH=/opt/cyclonedds/lib:${LD_LIBRARY_PATH} \
    TORCH_CUDA_ARCH_LIST="8.9;9.0+PTX" MAX_JOBS=4
RUN cmake -S /opt/cyclonedds-src -B /tmp/cyclonedds-build \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/cyclonedds \
    -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF \
    && cmake --build /tmp/cyclonedds-build --parallel 4 \
    && cmake --install /tmp/cyclonedds-build \
    && rm -rf /tmp/cyclonedds-build
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install cyclonedds==0.10.2 hatchling==1.27.0 editables==0.5 \
    && python -m pip install --no-build-isolation \
       -e /opt/simple/third_party/openpi-client \
       -e /opt/simple/third_party/gear_sonic \
       -e /opt/simple/third_party/unitree_sdk2_python \
       -e /opt/simple/third_party/decoupled_wbc
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-build-isolation -e '/opt/simple/third_party/curobo[isaacsim]' \
    && python -m pip install --no-deps --no-build-isolation -e /opt/simple
# SIMPLE imports the asset downloader even with all scene assets mounted locally.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install huggingface-hub==0.34.4
COPY audit_runtime.py /opt/simple-build/
RUN python -m pip check \
    && python /opt/simple-build/audit_runtime.py \
    && python -m pip freeze --all > /opt/simple-build/pip-freeze.txt \
    && dpkg-query -W > /opt/simple-build/dpkg-versions.txt
ENV ROBORSI_SIMPLE_ROOT=/opt/simple ROBORSI_SIMPLE_PYTHON=/opt/simple-venv/bin/python \
    MUJOCO_GL=egl PYNPUT_BACKEND=dummy OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
WORKDIR /opt/simple
ENTRYPOINT []
CMD ["python", "-c", "print('SIMPLE native base: mount assets and use the validated benchmark entry point.')"]
