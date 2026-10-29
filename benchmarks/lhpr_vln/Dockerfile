# Context: python -m benchmarks.package, after the framework source registration.
# The digest-pinned base supplies /opt/native/bin/python and Python 3.11 at
# /opt/roborsi-service/bin/python. See README.md for the native runtime contract.
ARG NATIVE_RUNTIME_IMAGE
FROM ${NATIVE_RUNTIME_IMAGE}
USER root
ENV PATH=/opt/roborsi-service/bin:${PATH}
WORKDIR /opt/roborsi
COPY benchmark-runtime/ /opt/roborsi/
COPY service-requirements.txt native-environment.json /opt/build-inputs/
RUN python -c "import sys; assert sys.version_info[:2] == (3, 11)" \
    && python -m pip install --no-cache-dir -r /opt/build-inputs/service-requirements.txt
RUN /opt/native/bin/python -c "import json,platform,importlib.metadata as m; d=json.load(open('/opt/build-inputs/native-environment.json')); assert platform.python_version()==d['python']; assert all(m.version(k)==v for k,v in d['packages'].items())" \
    && /opt/native/bin/python -m pip check
ENV PYTHONPATH=/opt/roborsi PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    NVIDIA_DRIVER_CAPABILITIES=all
RUN python -c "from pathlib import Path; from roborsi.benchmarks.runtime_source import verify_benchmark_runtime; verify_benchmark_runtime(Path('/opt/roborsi'), 'lhpr_vln')"
ENTRYPOINT []
CMD ["python", "-m", "roborsi.evaluation.service_host", "--kind", "benchmark", "--factory", "robots.lhpr_vln.benchmark_service:make_benchmark", "--host", "0.0.0.0", "--port", "8000"]
