# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Automatic setup must use locked inputs and clean up its own services."""

import json
import subprocess
from pathlib import Path

import pytest

from scripts.evaluation import bootstrap


def test_plan_is_side_effect_free_and_identifies_downloads(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("a plan must not run commands")

    monkeypatch.setattr(bootstrap, "_run", forbidden)
    destination = tmp_path / "not-created"
    plan = bootstrap.preparation_plan(
        ["libero", "libero_pro", "robocasa365"], destination
    )
    assert not destination.exists()
    assert plan["benchmarks"] == ["libero", "libero_pro", "robocasa365"]
    assert "@sha256:" in plan["images"]["libero_pro"]
    assert plan["icl"]["libero_pro"] == "embedded sealed packages (migration_subset)"
    assert bootstrap.ROBOCASA_REVISION in json.dumps(plan)


def test_docker_unavailable_prevents_downloads(tmp_path, monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        raise FileNotFoundError("docker")

    monkeypatch.setattr(bootstrap, "_run", run)
    with pytest.raises(RuntimeError, match="Docker"):
        bootstrap.prepare_benchmarks(["libero_pro"], tmp_path / "work")
    assert calls == [["docker", "info"]]
    assert not (tmp_path / "work").exists()


def test_preparation_uses_icl_preset_and_environment(tmp_path, monkeypatch):
    assets = tmp_path / "icl"
    assets.mkdir()
    (assets / "tokenizer.json").write_text("{}")
    calls = []
    monkeypatch.setattr(
        bootstrap, "_run", lambda command, **kwargs: calls.append(command)
    )
    monkeypatch.setattr(bootstrap, "_ensure_image", lambda image: image)
    monkeypatch.setattr(
        bootstrap, "_build_three_image", lambda name, work_dir, image: image
    )
    monkeypatch.setattr(bootstrap, "_ensure_tokenizer", lambda: None)
    monkeypatch.setattr(bootstrap, "_prepare_three_assets", lambda *_args: assets)
    prepared = bootstrap.prepare_benchmarks(["libero_pro"], tmp_path / "work")
    assert prepared.env["ROBORSI_LIBERO_PRO_ICL_ROOT"] == str(assets / "pro/packages")
    assert prepared.images["libero_pro"] == bootstrap.IMAGES["libero_pro"]
    preset = json.loads(Path(prepared.presets["libero_pro"]).read_text())
    assert preset["extends"].endswith("robots/libero/agents/three_benchmark/pro.yaml")
    assert preset["runtime"]["context"]["tokenizer_file"] == str(
        assets / "tokenizer.json"
    )
    from roborsi.agents.config import AgentConfig

    config = AgentConfig.from_file(prepared.presets["libero_pro"])
    assert config.resources["context_map"].name == "libero_pro.json"
    assert config.runtime().llm.openai_format == "chat"
    assert config.runtime().llm.prompt_cache_mode is None
    assert all("run" not in command for command in calls)


def test_cached_standard_assets_are_verified_without_downloading(tmp_path, monkeypatch):
    from roborsi.benchmarks.registry import directory_sha256

    root = tmp_path / "assets/libero"
    snapshot = root / "snapshots/data"
    snapshot.mkdir(parents=True)
    payload = snapshot / "asset.txt"
    payload.write_text("asset")
    (root / "asset.json").write_text(
        json.dumps(
            {
                "name": "libero",
                "kind": "directory",
                "version": bootstrap.LIBERO_ASSET_REVISION,
                "path": "snapshots/data",
                "sha256": directory_sha256(snapshot),
            }
        )
    )
    monkeypatch.setattr(
        bootstrap, "_run", lambda *_args, **_kwargs: pytest.fail("download")
    )
    assert bootstrap._prepare_standard_assets(tmp_path) == snapshot
    payload.write_text("corrupt")
    with pytest.raises(ValueError, match="checksum"):
        bootstrap._prepare_standard_assets(tmp_path)


def test_cached_icl_is_verified_instead_of_trusted(tmp_path, monkeypatch):
    cached = tmp_path / "assets/libero-three"
    cached.mkdir(parents=True)
    monkeypatch.setattr(
        bootstrap, "_run", lambda *_args, **_kwargs: pytest.fail("download")
    )
    monkeypatch.setattr(
        bootstrap,
        "_verify_three_assets",
        lambda path: (_ for _ in ()).throw(ValueError("changed ICL")),
    )
    with pytest.raises(ValueError, match="changed ICL"):
        bootstrap._prepare_three_assets(tmp_path, bootstrap.IMAGES["libero_pro"])


def test_service_overrides_combined_entrypoint_and_cleans_up_on_error(
    tmp_path, monkeypatch
):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        output = "127.0.0.1:12345\n" if command[1] == "port" else "container-id\n"
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    monkeypatch.setattr(bootstrap, "_run", run)
    monkeypatch.setattr(bootstrap, "_wait_service", lambda *args: None)
    prepared = bootstrap.Prepared(
        benchmarks=("libero_pro",),
        work_dir=tmp_path,
        presets={},
        env={},
        images={"libero_pro": bootstrap.IMAGES["libero_pro"]},
    )
    with pytest.raises(RuntimeError, match="evaluation failed"):
        with bootstrap.benchmark_services(prepared) as urls:
            assert urls == {"libero_pro": "http://127.0.0.1:12345"}
            raise RuntimeError("evaluation failed")
    start = calls[0]
    assert start[start.index("--entrypoint") + 1] == "python"
    assert "robots.libero.three_benchmark_serve" in start
    assert start[start.index("--profile") + 1] == "migration_subset"
    assert "127.0.0.1::8000" in start
    own_name = start[start.index("--name") + 1]
    assert calls[-1] == ["docker", "rm", "--force", own_name]


def test_partial_service_start_failure_removes_all_owned_containers(
    tmp_path, monkeypatch
):
    calls = []
    count = 0

    def run(command, **kwargs):
        nonlocal count
        calls.append(command)
        if command[1] == "run":
            count += 1
            if count == 2:
                raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(
            command, 0, stdout="127.0.0.1:12345\n", stderr=""
        )

    monkeypatch.setattr(bootstrap, "_run", run)
    monkeypatch.setattr(bootstrap, "_wait_service", lambda *args: None)
    names = ("libero_pro", "libero_ultra_long")
    prepared = bootstrap.Prepared(
        benchmarks=names,
        work_dir=tmp_path,
        presets={},
        env={},
        images={name: bootstrap.IMAGES[name] for name in names},
    )
    with pytest.raises(subprocess.CalledProcessError):
        with bootstrap.benchmark_services(prepared):
            pytest.fail("should not yield")
    started = [c[c.index("--name") + 1] for c in calls if c[1] == "run"]
    removed = [c[-1] for c in calls if c[1] == "rm"]
    assert removed == started[::-1]


def test_robocasa_preparation_binds_downloaded_package_to_real_factory(
    tmp_path, monkeypatch
):
    from roborsi.agents.config import AgentConfig

    package = tmp_path / "icl"
    package.mkdir()
    (package / "fixture.txt").write_text("same-task memory")
    calls = []
    monkeypatch.setattr(bootstrap, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bootstrap, "_build_image", lambda *args: "fixture-image")

    def download(**kwargs):
        calls.append(kwargs)
        return package

    monkeypatch.setattr(bootstrap, "download_snapshot", download)
    prepared = bootstrap.prepare_benchmarks(["robocasa365"], tmp_path / "work")
    assert calls[0]["revision"] == bootstrap.ROBOCASA_REVISION
    assert calls[0]["manifest_sha256"] == bootstrap.ROBOCASA_MANIFEST
    config = AgentConfig.from_file(prepared.presets["robocasa365"])
    assert config.factory == "robots.robocasa.model_eval_agent:make_agent"
    assert config.resources == {"icl_package": package.resolve()}


def test_service_catalog_settings_come_from_arguments(tmp_path, monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command, 0, stdout="127.0.0.1:12345", stderr=""
        )

    monkeypatch.setattr(bootstrap, "_run", run)
    monkeypatch.setattr(bootstrap, "_wait_service", lambda *args: None)
    prepared = bootstrap.Prepared(
        ("libero", "robocasa365"),
        tmp_path,
        {},
        {},
        {"libero": "fixture", "robocasa365": "fixture"},
        standard_asset=tmp_path / "assets",
    )
    with bootstrap.benchmark_services(
        prepared, libero_suite="libero_90", robocasa_cells="seed1"
    ):
        pass
    starts = [c for c in calls if c[1] == "run"]
    assert "ROBORSI_LIBERO_SUITE=libero_90" in starts[0]
    assert "--gpus" not in starts[0]
    assert "ROBORSI_ROBOCASA_CELLS=seed1" in starts[1]


def test_enhanced_image_overlays_current_service_on_pinned_native_stack(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        bootstrap.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 1),
    )
    builds = []

    def run(command, **kwargs):
        assert command[:2] == ["docker", "build"]
        context = Path(command[-1])
        dockerfile = (context / "Dockerfile").read_text()
        assert dockerfile.startswith(f"FROM {bootstrap.IMAGES['libero_pro']}\n")
        assert "verify_benchmark_runtime" in dockerfile
        assert "ENTRYPOINT []" in dockerfile
        source = "robots/libero/three_benchmark_serve.py"
        assert (context / "benchmark-runtime" / source).read_bytes() == (
            bootstrap.ROOT / source
        ).read_bytes()
        assert not (
            context / "benchmark-runtime/robots/libero/three_benchmark_agent.py"
        ).exists()
        builds.append(command)

    monkeypatch.setattr(bootstrap, "_run", run)
    image = bootstrap._build_three_image(
        "libero_pro", tmp_path, bootstrap.IMAGES["libero_pro"]
    )
    assert image.startswith("roborsi-onecmd-libero_pro:")
    assert len(builds) == 1
