ARG PYTHON_IMAGE=python:3.11-slim-bookworm@sha256:a36c24f9cbdf4fd0f52d67f0823eeac19c2028c637cecc392d97f980d4fec56b
FROM ${PYTHON_IMAGE}

ENV MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_TYPE=pro PYTHONUNBUFFERED=1 \
    ROBORSI_LIBERO_PRO_COMMIT=d1e11fb181b8544487d27742c0caa3a4d46452ad \
    ROBORSI_LIBERO_VERSION=libero-pro-d1e11fb181b8544487d27742c0caa3a4d46452ad
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libc6-dev linux-libc-dev libgl1 libegl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --no-deps \
    'https://download-r2.pytorch.org/whl/cpu/torch-2.7.1%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl#sha256=a1684793e352f03fa14f78857e55d65de4ada8405ded1da2bf4f452179c4b779'
RUN pip install --no-cache-dir --timeout 120 \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    pydantic==2.11.7 jsonschema==4.23.0 pillow==11.3.0 imageio==2.37.0 \
    imageio-ffmpeg==0.6.0 numpy==1.26.4 scipy==1.15.3 \
    opencv-python==4.11.0.86 robosuite==1.5.2 mujoco==3.3.0 \
    omegaconf==2.3.0 gymnasium==0.29.1
RUN pip install --no-cache-dir --timeout 120 \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    filelock==3.20.0 jinja2==3.1.6 networkx==3.5 sympy==1.14.0
RUN pip install --no-cache-dir --timeout 120 \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    rpent-libero==0.2.0 bddl==1.0.1 gym==0.25.2 h5py==3.14.0 \
    matplotlib==3.9.2 easydict==1.13 future==1.0.0 cloudpickle==3.1.2 \
    huggingface-hub==0.34.4
RUN pip install --no-cache-dir --no-deps \
    https://files.pythonhosted.org/packages/cb/db/0d7849be4c5406e45d6126cca03a978e8a6cf146470148201c7064a0d411/rpent_rlinf-0.3.0-py3-none-any.whl
ARG LIBERO_PRO_ARCHIVE_URL=https://codeload.github.com/RLinf/LIBERO-PRO/tar.gz/d1e11fb181b8544487d27742c0caa3a4d46452ad
ARG LIBERO_PRO_ARCHIVE_SHA256=88c73fe35cc056e704bc7097aa17fb21055f6cc00d19e97f789f47d865c6ad37
ADD ${LIBERO_PRO_ARCHIVE_URL} /tmp/libero-pro.tar.gz
RUN echo "${LIBERO_PRO_ARCHIVE_SHA256}  /tmp/libero-pro.tar.gz" | sha256sum -c - \
    && mkdir /opt/libero-pro && tar -xzf /tmp/libero-pro.tar.gz -C /opt/libero-pro --strip-components=1 \
    && pip install --no-cache-dir --no-deps /opt/libero-pro \
    && cp -a /opt/libero-pro/liberopro/liberopro/assets/. \
        /usr/local/lib/python3.11/site-packages/liberopro/liberopro/assets/

WORKDIR /opt/roborsi
COPY benchmark-runtime/ /opt/roborsi/
COPY evaluation-assets/ /opt/libero-pro-evaluation/
RUN python -B -c "from pathlib import Path; from roborsi.benchmarks.registry import directory_sha256; root = Path('/opt/libero-pro-evaluation'); expected = 'd4a0a1ddbe8137cbcb410a41d9ec0296f27cb709e03f0ae6f9bd3fb11da7b59d'; actual = directory_sha256(root); assert actual == expected, (actual, expected)"
ENV PYTHONPATH=/opt/roborsi PYTHONDONTWRITEBYTECODE=1 \
    LIBERO_CONFIG_PATH=/tmp/libero-config MPLCONFIGDIR=/tmp/matplotlib \
    ROBORSI_LIBERO_BENCHMARK=libero_pro \
    ROBORSI_LIBERO_ASSETS_ROOT=/usr/local/lib/python3.11/site-packages/liberopro/liberopro/assets \
    ROBORSI_LIBERO_BENCHMARK_ROOT=/usr/local/lib/python3.11/site-packages/liberopro/liberopro \
    ROBORSI_LIBERO_DATASETS_ROOT=/opt/libero-pro/liberopro/datasets \
    ROBORSI_LIBERO_BDDL_ROOT=/opt/libero-pro-evaluation/bddl_files \
    ROBORSI_LIBERO_INIT_ROOT=/opt/libero-pro-evaluation/init_files \
    ROBORSI_LIBERO_EVALUATION_ASSET_SHA256=d4a0a1ddbe8137cbcb410a41d9ec0296f27cb709e03f0ae6f9bd3fb11da7b59d
RUN python -B -c "from pathlib import Path; from roborsi.benchmarks.runtime_source import verify_benchmark_runtime; verify_benchmark_runtime(Path('/opt/roborsi'), 'libero_pro')"
RUN LIBERO_CONFIG_PATH=/tmp/libero-import-check python -B -c "import bddl, gym, h5py, liberopro, matplotlib; from liberopro.liberopro.envs import OffScreenRenderEnv; print('LIBERO-PRO native imports verified')" \
    && rm -rf /tmp/libero-import-check
CMD ["python", "-m", "robots.libero.three_benchmark_serve", "--benchmark", "libero_pro", "--profile", "official"]
