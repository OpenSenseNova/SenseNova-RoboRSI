# Copyright 2026 The SenseNova-RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Real offline wheel installs, dependency checks and per-case file ownership."""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path
from threading import Event, Timer

import pytest

from roborsi.evaluation.agent_artifact import AgentArtifactStore
from roborsi.evaluation.agent_environment import AgentEnvironmentStore, _command
from roborsi.evaluation.deadline import CaseCancelled, CaseDeadline
from tests.unit_tests.roborsi.evaluation.test_agent_artifact import (
    _COMPONENTS,
    _FACTORY,
    _source,
)


def _wheel(directory: Path, version: str = "1.0", requires: str = "") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"roborsi_example_dependency-{version}-py3-none-any.whl"
    metadata = f"roborsi_example_dependency-{version}.dist-info"
    files = {
        "roborsi_example_dependency.py": f"VALUE = {version!r}\n",
        f"{metadata}/METADATA": f"Metadata-Version: 2.1\nName: roborsi-example-dependency\nVersion: {version}\n{requires}\n",
        f"{metadata}/WHEEL": "Wheel-Version: 1.0\nGenerator: roborsi-offline-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    files[f"{metadata}/RECORD"] = "".join(
        f"{name},,\n" for name in [*files, f"{metadata}/RECORD"]
    )
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return path


def test_offline_bundle_rebuilds_two_independent_environments(
    tmp_path: Path, monkeypatch
):
    wheels = tmp_path / "wheels"
    _wheel(wheels)
    bundle = AgentEnvironmentStore(tmp_path / "bundles").create(wheels)
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / "roborsi_example_dependency.py").write_text("VALUE = 'wrong'\n")
    monkeypatch.setenv("PYTHONPATH", str(shadow))
    monkeypatch.setenv("PIP_INDEX_URL", "http://127.0.0.1:1")
    installed = [bundle.prepare(tmp_path / name) for name in ("one", "two")]
    for environment in installed:
        output = subprocess.check_output(
            [
                environment.python_executable,
                "-I",
                "-B",
                "-c",
                "import json, sys, roborsi_example_dependency as d; print(json.dumps([d.VALUE, sys.prefix, sys.base_prefix]))",
            ],
            text=True,
        )
        value, prefix, base = json.loads(output)
        assert value == "1.0" and prefix != base
        assert Path(prefix) == environment.root
        environment.verify()
    with pytest.raises(FileExistsError):
        bundle.prepare(tmp_path / "one")
    next(installed[0].root.rglob("roborsi_example_dependency.py")).write_text(
        "VALUE = 'changed'\n"
    )
    with pytest.raises(ValueError, match="environment changed"):
        installed[0].verify()
    installed[1].verify()


def test_bundle_rejects_missing_transitive_dependencies(tmp_path: Path):
    wheels = tmp_path / "wheels"
    _wheel(wheels, requires="Requires-Dist: absent-roborsi-package==1.0")
    bundle = AgentEnvironmentStore(tmp_path / "bundles").create(wheels)
    with pytest.raises(ValueError, match="absent-roborsi-package"):
        bundle.prepare(tmp_path / "case")


def test_dependency_change_changes_agent_version_and_parent_inherits_environment(
    tmp_path: Path,
):
    store = AgentArtifactStore(tmp_path / "agents")
    source = _source(tmp_path / "source")
    first, second = tmp_path / "wheels1", tmp_path / "wheels2"
    _wheel(first)
    _wheel(second, version="2.0")
    env1, env2 = (store.environments.create(path) for path in (first, second))
    baseline = store.create(
        source, _FACTORY, components=_COMPONENTS, environment_version=env1.version_id
    )
    candidate = store.create(
        source,
        _FACTORY,
        components=_COMPONENTS,
        parent_version=baseline.version_id,
        environment_version=env2.version_id,
    )
    assert baseline.version_id != candidate.version_id
    assert candidate.parent_version == baseline.version_id
    assert candidate.changes == ()  # The dependency bundle changed, not Agent files.
    assert candidate.document["environment_version"] == env2.version_id
    inherited = store.create(
        source, _FACTORY, components=_COMPONENTS, parent_version=candidate.version_id
    )
    assert inherited.document["environment_version"] == env2.version_id
    wheel = next(env1.wheel_dir.iterdir())
    wheel.chmod(0o600)
    wheel.write_bytes(b"changed")
    with pytest.raises(ValueError, match="wheels changed"):
        store.load(baseline.version_id)


def test_setup_process_stops_on_case_cancellation():
    event = Event()
    timer = Timer(0.1, event.set)
    timer.start()
    try:
        with pytest.raises(CaseCancelled):
            _command(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                CaseDeadline(10, cancel_event=event),
                timeout_s=60,
            )
    finally:
        timer.cancel()


def test_setup_does_not_inherit_credentials_or_python_settings(monkeypatch):
    monkeypatch.setenv("ROBORSI_TEST_SECRET", "unused-test-value")
    monkeypatch.setenv("PYTHONPATH", "/untrusted")
    output = _command(
        [
            sys.executable,
            "-c",
            "import os; print('ROBORSI_TEST_SECRET' in os.environ or 'PYTHONPATH' in os.environ)",
        ],
        None,
        timeout_s=3,
    )
    assert output.strip() == "False"


def test_setup_ignores_pip_configuration(tmp_path):
    environment = tmp_path / "python"
    subprocess.run(
        [sys.executable, "-I", "-B", "-S", "-m", "venv", "--copies", str(environment)],
        check=True,
    )
    configured_index = "https://must-not-be-used.example/simple"
    (environment / "pip.conf").write_text(f"[global]\nindex-url = {configured_index}\n")
    command = [
        str(environment / "bin" / "python"),
        "-I",
        "-B",
        "-m",
        "pip",
        "--isolated",
        "config",
        "list",
    ]
    assert configured_index in subprocess.check_output(command, text=True, env={})
    assert configured_index not in _command(command, None, timeout_s=10)


def test_python_fingerprint_follows_linked_standard_library(tmp_path, monkeypatch):
    from roborsi.evaluation import python_runtime

    library = tmp_path / "stdlib"
    library.mkdir()
    linked = tmp_path / "linked"
    linked.mkdir()
    module = linked / "module.py"
    module.write_text("VALUE = 1\n")
    (library / "package").symlink_to(linked, target_is_directory=True)
    monkeypatch.setattr(python_runtime.sysconfig, "get_path", lambda _: str(library))
    first = python_runtime.describe_python()
    module.write_text("VALUE = 2\n")
    assert python_runtime.describe_python()["stdlib_sha256"] != first["stdlib_sha256"]
    (linked / "cycle").symlink_to(library, target_is_directory=True)
    with pytest.raises(ValueError, match="cyclic directory link"):
        python_runtime.describe_python()


def test_cli_selects_saved_source_and_environment(tmp_path: Path, monkeypatch, capsys):
    from roborsi.evaluation import cli
    from tests.unit_tests.roborsi.evaluation.test_batch import (
        _PublishedCases,
        _registry,
    )

    root = tmp_path / "registry"
    _registry(root)
    source = _source(tmp_path / "source")
    store = AgentArtifactStore(tmp_path / "agents")
    wheels = tmp_path / "wheels"
    _wheel(wheels)
    environment = store.environments.create(wheels)
    artifact = store.create(
        source,
        _FACTORY,
        components=_COMPONENTS,
        environment_version=environment.version_id,
    )
    launchers = []

    def orchestrator(registry, results, launcher, executor):
        launchers.append(launcher)
        return _PublishedCases(registry, results)

    monkeypatch.setattr(cli, "CaseOrchestrator", orchestrator)
    assert (
        cli.main(
            [
                "eval",
                "--registry-root",
                str(root),
                "--results-root",
                str(tmp_path / "results"),
                "--run-id",
                "saved",
                "--agent-artifact-store",
                str(store.root),
                "--agent-version",
                artifact.version_id,
                "--agent-credential-env",
                "MODEL_API_KEY",
                "--benchmark",
                "demo",
                "--seed",
                "0",
            ]
        )
        == 0
    )
    assert launchers[0].agent_version == artifact.version_id
    assert (
        json.loads(capsys.readouterr().out)["cases"][0]["result"]["provenance"][
            "agent_version"
        ]
        == artifact.version_id
    )
