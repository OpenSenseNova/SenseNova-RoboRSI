# Overlay the source bundle onto an operator-prepared, digest-pinned native image.
# The base contains RoboDojo/Isaac/cuRobo and no Agent SDKs or credentials.
ARG ROBODOJO_BASE_IMAGE
FROM ${ROBODOJO_BASE_IMAGE}
ARG ROBODOJO_NATIVE_COMMIT=726e9aabfaa642203722eb126f5eaf0f37f3e1ad
ENV ROBORSI_ROBODOJO_NATIVE_COMMIT=${ROBODOJO_NATIVE_COMMIT}
USER root
COPY native-requirements.txt /opt/robodojo-build/native-requirements.txt
RUN python -m pip install -r /opt/robodojo-build/native-requirements.txt
RUN python -c "from pathlib import Path; p=Path('/workspace/RoboDojo/Assets'); p.symlink_to('/roborsi/assets/robodojo') if not (p.exists() or p.is_symlink()) else None; assert p.is_symlink() and str(p.readlink()) == '/roborsi/assets/robodojo', 'Prepare the official Assets link in the native base'"
ENV PYTHONPATH=/opt/roborsi:/workspace/RoboDojo \
    PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
    ROBORSI_ROBODOJO_NATIVE_ROOT=/workspace/RoboDojo \
    ROBORSI_ROBODOJO_ASSETS=/roborsi/assets/robodojo \
    ROBORSI_ROBODOJO_CATALOG=/roborsi/assets/robodojo-catalog
WORKDIR /opt/roborsi
COPY benchmark-runtime/ /opt/roborsi/
RUN python -P -c "from pathlib import Path; from roborsi.benchmarks.runtime_source import verify_benchmark_runtime; from robots.robodojo.benchmark_service import verify_native_source; verify_benchmark_runtime(Path('/opt/roborsi'), 'robodojo'); verify_native_source()"
ENTRYPOINT []
HEALTHCHECK NONE
CMD ["python", "-P", "-m", "roborsi.evaluation.service_host", "--kind", "benchmark", "--factory", "robots.robodojo.benchmark_service:make_robodojo_benchmark", "--host", "0.0.0.0", "--port", "8000"]
