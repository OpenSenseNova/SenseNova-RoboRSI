# A deployment-only image for schedulers that accept one image per job. The
# canonical Benchmark and Agent images remain independently buildable.
ARG AGENT_IMAGE
ARG BENCHMARK_IMAGE
FROM ${AGENT_IMAGE} AS agent
FROM ${BENCHMARK_IMAGE}

ARG AGENT_IMAGE
ARG BENCHMARK_IMAGE
ARG ROBORSI_REVISION
LABEL org.opencontainers.image.revision=${ROBORSI_REVISION} \
      org.opencontainers.image.source="https://github.com/SenseTime-FVG/RoboRSI" \
      org.opencontainers.image.description="RoboRSI three-LIBERO single-image runtime"

# The locked inputs use the same Python 3.11 base. Keep Agent packages separate
# from the native environment instead of merging incompatible site-packages.
COPY --from=agent /usr/local/lib/python3.11/site-packages/ \
    /opt/roborsi-agent-python/
COPY --from=agent /opt/roborsi-agent-assets/ /opt/roborsi-agent-assets/

# Overlay the source under review. The Benchmark base supplies native packages
# and its pinned assets; the Agent stage supplies planner dependencies/bundles.
COPY roborsi/ /opt/roborsi/roborsi/
COPY robots/__init__.py /opt/roborsi/robots/__init__.py
COPY robots/libero/ /opt/roborsi/robots/libero/
COPY configs/agents/libero/ /opt/roborsi/configs/agents/libero/
COPY configs/agents/embodied/default.yaml /opt/roborsi/configs/agents/embodied/default.yaml
COPY benchmarks/libero_three_combined/run_combined.py \
    /opt/roborsi/benchmarks/libero_three_combined/run_combined.py
COPY benchmarks/libero_three_combined/run_combined_subset.py \
    /opt/roborsi/benchmarks/libero_three_combined/run_combined_subset.py

ENV ROBORSI_LIBERO_LONGPP_ICL_ROOT=/opt/roborsi-agent-assets/longpp/packages \
    ROBORSI_LIBERO_ULTRA_ICL_ROOT=/opt/roborsi-agent-assets/ultra/packages \
    ROBORSI_LIBERO_PRO_ICL_ROOT=/opt/roborsi-agent-assets/pro/packages \
    ROBORSI_CONTEXT_TOKENIZER_PATH=/opt/roborsi-agent-assets/tokenizer.json \
    ROBORSI_AGENT_ASSET_MANIFEST=/opt/roborsi-agent-assets/asset.json \
    ROBORSI_IMAGE_REVISION=${ROBORSI_REVISION} \
    ROBORSI_AGENT_IMAGE=${AGENT_IMAGE} \
    ROBORSI_BENCHMARK_IMAGE=${BENCHMARK_IMAGE} \
    PYTHONPATH=/opt/roborsi:/opt/roborsi-agent-python:${PYTHONPATH}

RUN python -B -c "from pathlib import Path; import roborsi_agent_bundle_libero_long_plus_plus_v13, roborsi_agent_bundle_libero_ultra_long_v13, roborsi_agent_bundle_libero_pro, pydantic_ai, tokenizers; from scipy.spatial.transform import Rotation; Rotation.identity(); assert Path('/opt/roborsi-agent-assets/tokenizer.json').is_file()"

WORKDIR /opt/roborsi
ENTRYPOINT ["python", "-B", "/opt/roborsi/benchmarks/libero_three_combined/run_combined.py"]
