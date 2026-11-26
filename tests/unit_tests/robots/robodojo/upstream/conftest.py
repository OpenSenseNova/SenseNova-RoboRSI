from pathlib import Path
import os

import pytest
from .test_hybrid_backend import backend, experiment, reference, robot, vla

@pytest.fixture(autouse=True)
def offline_provider_environment(monkeypatch):
    # Original regression programs import this source package from temporary
    # working directories; scope their path to these tests, not the whole suite.
    paths = [str(Path(__file__).resolve().parents[5]), os.environ.get("PYTHONPATH", "")]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, paths)))
    monkeypatch.setenv("ROBORSI_CODEX_BASE_URL", "https://provider.example.invalid/v1")
    monkeypatch.setenv("ROBORSI_CODEX_MODEL", "fixture-model")
