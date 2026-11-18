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

"""Standalone Benchmark contexts exclude Agent sources and imports."""

import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.package import prepare_context
from roborsi.benchmarks.runtime_source import (
    benchmark_source_digest,
    benchmark_source_files,
)

ROOT = Path(__file__).resolve().parents[4]

_IMPORT_GUARD = r"""
import importlib.abc, importlib, sys
sys.path.insert(0, sys.argv[1])
blocked = ('roborsi.embodied_agent', 'roborsi.planner', 'roborsi.llm', 'roborsi.memory', 'roborsi.dashboard', 'roborsi.tools', 'roborsi.evaluation.services', 'openai', 'anthropic', 'pydantic_ai', 'mcp')
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise AssertionError('Agent import in Benchmark: ' + fullname)
sys.meta_path.insert(0, Guard())
"""


@pytest.mark.parametrize(
    "benchmark", ["libero", "bench2dex", "dexjoco", "behavior_1k", "robolab", "simple"]
)
def test_bundle_imports_and_service_help_without_agent_code(tmp_path, benchmark):
    context = prepare_context(benchmark, tmp_path / "context", source_root=ROOT)
    runtime = context / "benchmark-runtime"
    assert {
        p.relative_to(runtime).as_posix() for p in runtime.rglob("*") if p.is_file()
    } == set(benchmark_source_files(benchmark))
    assert benchmark_source_digest(ROOT, benchmark) == benchmark_source_digest(
        runtime, benchmark
    )
    code = (
        _IMPORT_GUARD
        + r"""
module = importlib.import_module('robots.' + sys.argv[2] + '.benchmark_service')
from roborsi.evaluation.benchmark_http import BenchmarkApplication, BenchmarkHttpServer
from roborsi.evaluation.service_host import main
assert not any(name in sys.modules for name in blocked)
from pathlib import Path
for name, loaded in tuple(sys.modules.items()):
    if name.split('.')[0] in ('roborsi', 'robots') and getattr(loaded, '__file__', None):
        assert Path(loaded.__file__).is_relative_to(sys.argv[1]), (name, loaded.__file__)
main(['--kind', 'benchmark', '--help'])
"""
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(runtime), benchmark],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--kind" in result.stdout
    dockerfile = (context / "Dockerfile").read_text()
    assert "COPY benchmark-runtime/ /opt/roborsi/" in dockerfile
    assert "COPY roborsi " not in dockerfile and "COPY robots " not in dockerfile
    assert "verify_benchmark_runtime" in dockerfile


def test_agent_edits_do_not_change_bundle_or_adapter_version(tmp_path):
    context = prepare_context("libero", tmp_path / "first", source_root=ROOT)
    checkout = context / "benchmark-runtime"
    before = benchmark_source_digest(checkout, "libero")
    agent = checkout / "roborsi/embodied_agent.py"
    agent.write_text("# An external Agent implementation changed.\n")
    prompts = checkout / "robots/libero/prompts"
    prompts.mkdir()
    (prompts / "system.py").write_text("PROMPT = 'different agent prompt'\n")
    assert benchmark_source_digest(checkout, "libero") == before
    adapter = checkout / "robots/libero/roborsi_adapter.py"
    adapter.write_text(adapter.read_text() + "\n# A wrapper implementation changed.\n")
    assert benchmark_source_digest(checkout, "libero") != before


@pytest.mark.parametrize("benchmark", ["libero_pro", "libero_lilo"])
def test_extended_libero_bundles_import_without_agent_sources(tmp_path, benchmark):
    context = prepare_context(benchmark, tmp_path / benchmark, source_root=ROOT)
    dockerfile = (context / "Dockerfile").read_text()
    runtime = context / "benchmark-runtime"
    assert {
        path.relative_to(runtime).as_posix()
        for path in runtime.rglob("*")
        if path.is_file()
    } == set(benchmark_source_files(benchmark))
    code = (
        _IMPORT_GUARD
        + r"""
importlib.import_module('robots.libero.three_benchmark_service')
from roborsi.evaluation.service_host import main
assert not any(name in sys.modules for name in blocked)
main(['--kind', 'benchmark', '--help'])
"""
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(runtime)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--kind" in result.stdout
    if benchmark == "libero_lilo":
        assert "mkdir -p /opt/lilo-vla/configs/libero" in dockerfile
        assert "NUMBA_CACHE_DIR=/tmp/numba" in dockerfile


@pytest.mark.parametrize(
    "benchmark",
    ["libero", "libero_pro", "libero_lilo", "dexjoco", "behavior_1k", "robolab"],
)
def test_exported_manifest_help_does_not_import_native_dependencies(
    tmp_path, benchmark
):
    context = prepare_context(benchmark, tmp_path / "context", source_root=ROOT)
    runtime = context / "benchmark-runtime"
    code = (
        _IMPORT_GUARD
        + r"""
from pathlib import Path
native = ('rlinf', 'libero', 'liberopro', 'lilo_vla', 'robosuite', 'mujoco', 'torch', 'omnigibson', 'bddl', 'omni', 'isaacsim')
class NativeGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in native:
            raise AssertionError('Native import during manifest help: ' + fullname)
sys.meta_path.insert(0, NativeGuard())
behavior = sys.argv[2] == 'behavior_1k'
module = importlib.import_module('benchmarks.' + sys.argv[2] + ('.prepare' if behavior else '.manifest'))
assert Path(module.__file__).is_relative_to(sys.argv[1]), module.__file__
sys.argv = ['prepare', 'manifest', '--help'] if behavior else ['manifest', '--help']
module.main()
"""
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(runtime), benchmark],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--image" in result.stdout
    assert "--output" in result.stdout


@pytest.mark.parametrize("package", ["roborsi", "heart_runtime", "rpent"])
def test_image_verifier_rejects_stale_agent_files_and_model_sdks(
    tmp_path, monkeypatch, package
):
    from types import SimpleNamespace

    from roborsi.benchmarks import runtime_source

    runtime = (
        prepare_context("libero", tmp_path / "context", source_root=ROOT)
        / "benchmark-runtime"
    )
    monkeypatch.setattr(runtime_source, "distributions", lambda: [])
    runtime_source.verify_benchmark_runtime(runtime, "libero")
    extra = runtime / package / "embodied_agent.py"
    extra.parent.mkdir(exist_ok=True)
    extra.write_text("# stale Agent from the base image")
    with pytest.raises(ValueError, match="undeclared"):
        runtime_source.verify_benchmark_runtime(runtime, "libero")
    extra.unlink()
    monkeypatch.setattr(
        runtime_source,
        "distributions",
        lambda: [SimpleNamespace(metadata={"Name": "openai"})],
    )
    with pytest.raises(ValueError, match="Agent packages: openai"):
        runtime_source.verify_benchmark_runtime(runtime, "libero")


def test_exported_service_executes_http_and_downloads_artifacts_without_agent(tmp_path):
    import time

    from roborsi.evaluation.benchmark_client import BenchmarkClient
    from roborsi.evaluation.media import CaseMediaChannel
    from roborsi.evaluation.messages import Action
    from roborsi.utils.daemon import pick_free_port
    from roborsi.utils.rpc import RpcError
    from tests.unit_tests.roborsi.evaluation.test_benchmark_http import task

    runtime = (
        prepare_context("libero", tmp_path / "context", source_root=ROOT)
        / "benchmark-runtime"
    )
    # Copy a source-only fake native environment without exposing the repository
    # to the service's interpreter.
    fixture = ROOT / "tests/fixtures/isolated_benchmark_provider.py"
    (tmp_path / "provider.py").write_text(fixture.read_text())
    port = pick_free_port()
    code = (
        _IMPORT_GUARD
        + """
sys.path.insert(0, sys.argv[2])
from roborsi.evaluation.service_host import main
main(['--kind', 'benchmark', '--factory', 'provider:Provider', '--port', sys.argv[3]])
"""
    )
    with (tmp_path / "service.log").open("w+") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                code,
                str(runtime),
                str(tmp_path),
                str(port),
            ],
            cwd=tmp_path,
            stdout=log,
            stderr=log,
        )
        try:
            client = BenchmarkClient(f"http://127.0.0.1:{port}")
            deadline = time.monotonic() + 15
            while True:
                try:
                    assert client.tasks() == [task()]
                    break
                except RpcError:
                    if process.poll() is not None or time.monotonic() >= deadline:
                        log.seek(0)
                        pytest.fail(log.read())
                    time.sleep(0.05)
            start = client.start(task(), "isolated", timeout_s=3)
            channel = CaseMediaChannel(tmp_path / "runner/media")
            client.collect_observation(start.observation, channel, timeout_s=3)
            step = client.step(
                Action(benchmark="demo", episode_id="isolated", step_index=0, data={}),
                timeout_s=3,
            )
            client.collect_observation(step.observation, channel, timeout_s=3)
            result = client.finish(timeout_s=3)
            assert step.terminated and result.score == 0.75
            client.collect_native(result, channel.media_dir.parent, timeout_s=3)
            client.collect_artifacts(channel.media_dir.parent, timeout_s=3)
            assert {item.kind for item in client.artifacts()} == {
                "image",
                "native_result",
                "log",
            }
            client.close()
            client.delete("isolated")
            with pytest.raises(RpcError, match="episode not found"):
                client.status("isolated")
            channel.verify_observation(step.observation)
            assert (
                tmp_path / "runner/native/score.json"
            ).read_bytes() == b'{"score":0.75}'
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


@pytest.mark.parametrize("distribution", ["roborsi", "roborsi", "roborsi", "rpent"])
def test_benchmark_rejects_installed_agent_distribution(
    tmp_path, monkeypatch, distribution
):
    from types import SimpleNamespace

    from roborsi.benchmarks import runtime_source

    context = prepare_context("libero", tmp_path / "context", source_root=ROOT)
    monkeypatch.setattr(
        runtime_source,
        "distributions",
        lambda: [SimpleNamespace(metadata={"Name": distribution})],
    )
    with pytest.raises(ValueError, match="Agent packages"):
        runtime_source.verify_benchmark_runtime(context / "benchmark-runtime", "libero")
