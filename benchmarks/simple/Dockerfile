# Requires a clean, already installed SIMPLE / Isaac 4.5 native base by digest.
# Export with the shared packager after framework source registration is merged.
ARG SIMPLE_RUNTIME_IMAGE
FROM ${SIMPLE_RUNTIME_IMAGE}
WORKDIR /opt/roborsi
COPY benchmark-runtime/ /opt/roborsi/
COPY requirements-http.lock /tmp/simple-http.lock
COPY install_compat.py /opt/simple-build/install_compat.py
RUN python /opt/simple-build/install_compat.py \
    --sources /opt/roborsi/robots/simple/sources.json --output /opt/simple-compat
RUN python -m pip install --no-cache-dir --require-hashes --no-deps -r /tmp/simple-http.lock \
    && python -m pip check \
    && python -m pip freeze --all > /opt/simple-build/service-pip-freeze.txt
ENV PYTHONPATH=/opt/roborsi PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    ROBORSI_SIMPLE_ASSETS=/roborsi/assets/simple-assets \
    HOME=/tmp/simple-home XDG_CACHE_HOME=/tmp/simple-home/.cache
RUN python -B -c "from pathlib import Path; from roborsi.benchmarks.runtime_source import verify_benchmark_runtime; verify_benchmark_runtime(Path('/opt/roborsi'), 'simple')"
ENTRYPOINT []
CMD ["python", "-m", "roborsi.evaluation.service_host", "--kind", "benchmark", "--factory", "robots.simple.benchmark_service:make_simple_benchmark", "--host", "0.0.0.0", "--port", "8000"]
