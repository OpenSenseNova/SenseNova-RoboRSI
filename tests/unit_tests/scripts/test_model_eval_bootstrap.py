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
from scripts.evaluation.configuration import load_benchmark

IMAGES = {
    name: load_benchmark(name).environment["image"]
    for name in ("libero_pro", "libero_long_plus_plus", "libero_ultra_long")
}


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
    assert load_benchmark("robocasa365").environment["icl"]["revision"] in json.dumps(
        plan
    )


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
        bootstrap, "_build_three_image", lambda name, work_dir, image, *args: image
    )
    monkeypatch.setattr(bootstrap, "_ensure_tokenizer", lambda *args: None)
    monkeypatch.setattr(bootstrap, "_prepare_three_assets", lambda *_args: assets)
    prepared = bootstrap.prepare_benchmarks(["libero_pro"], tmp_path / "work")
    assert prepared.env["ROBORSI_LIBERO_PRO_ICL_ROOT"] == str(assets / "pro/packages")
    assert prepared.images["libero_pro"] == IMAGES["libero_pro"]
    preset = json.loads(Path(prepared.presets["libero_pro"]).read_text())
    assert preset["extends"].endswith("configs/agents/libero/pro.yaml")
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
                "version": load_benchmark("libero").environment["assets"]["revision"],
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
        lambda path, *args: (_ for _ in ()).throw(ValueError("changed ICL")),
    )
    with pytest.raises(ValueError, match="changed ICL"):
        bootstrap._prepare_three_assets(tmp_path, IMAGES["libero_pro"])


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
        images={"libero_pro": IMAGES["libero_pro"]},
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
    assert "--legacy-image-env" not in start
    own_name = start[start.index("--name") + 1]
    assert calls[-1] == ["docker", "rm", "--force", own_name]
    assert calls[-2] == ["docker", "logs", "--timestamps", own_name]
    assert (
        tmp_path / "logs/services" / f"{own_name}.log"
    ).read_text() == "container-id\n"


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
        images={name: IMAGES[name] for name in names},
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
    assert (
        calls[0]["revision"]
        == load_benchmark("robocasa365").environment["icl"]["revision"]
    )
    assert (
        calls[0]["manifest_sha256"]
        == load_benchmark("robocasa365").environment["icl"]["manifest_sha256"]
    )
    config = AgentConfig.from_file(prepared.presets["robocasa365"])
    assert config.factory == "robots.robocasa.model_eval_agent:make_agent"
    assert config.resources == {"icl_package": package.resolve()}
    assert config.parameters["icl_manifest_sha256"] == calls[0]["manifest_sha256"]
    assert config.parameters["icl_revision"] == calls[0]["revision"]


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
        assert dockerfile.startswith(f"FROM {IMAGES['libero_pro']}\n")
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
    image = bootstrap._build_three_image("libero_pro", tmp_path, IMAGES["libero_pro"])
    assert image.startswith("roborsi-onecmd-libero_pro:")
    assert len(builds) == 1


def test_build_arguments_from_yaml_are_used_and_invalidate_image_cache(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    original = load_benchmark("libero")
    builds = []
    monkeypatch.setattr(
        bootstrap.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 1),
    )
    monkeypatch.setattr(
        bootstrap, "_run", lambda command, **kwargs: builds.append(command)
    )
    first = bootstrap._build_image("libero", tmp_path, original)
    changed = replace(
        original,
        environment={
            **original.environment,
            "build_args": {
                "LIBERO_RUNTIME_IMAGE": "example/runtime@sha256:" + "a" * 64
            },
        },
    )
    second = bootstrap._build_image("libero", tmp_path, changed)
    assert first != second
    assert "--build-arg" in builds[0]
    assert (
        "LIBERO_RUNTIME_IMAGE="
        + original.environment["build_args"]["LIBERO_RUNTIME_IMAGE"]
        in builds[0]
    )
    assert "LIBERO_RUNTIME_IMAGE=example/runtime@sha256:" + "a" * 64 in builds[1]


def test_benchmark_agent_overrides_are_composed_with_local_bindings(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    from roborsi.agents.config import AgentConfig

    original = load_benchmark("libero_pro")
    overrides = json.loads(json.dumps(original.agent_overrides))
    overrides["agent"].update(reasoning_effort="high", max_tokens=4096)
    config = replace(original, agent_overrides=overrides)
    monkeypatch.setattr(bootstrap, "load_benchmark", lambda name: config)
    monkeypatch.setattr(bootstrap, "_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(bootstrap, "_ensure_image", lambda image: image)
    monkeypatch.setattr(bootstrap, "_ensure_tokenizer", lambda *args: None)
    monkeypatch.setattr(bootstrap, "_build_three_image", lambda *args: "fixture")
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "tokenizer.json").write_text("{}")
    monkeypatch.setattr(bootstrap, "_prepare_three_assets", lambda *args: assets)
    prepared = bootstrap.prepare_benchmarks(["libero_pro"], tmp_path / "work")
    preset = AgentConfig.from_file(prepared.presets["libero_pro"])
    assert preset.document["agent"]["reasoning_effort"] == "high"
    assert preset.document["agent"]["max_tokens"] == 4096
    assert preset.document["runtime"]["context"]["tokenizer_file"] == str(
        assets / "tokenizer.json"
    )


def test_pro_services_forward_protocol_and_tool_endpoints(tmp_path, monkeypatch):
    from types import SimpleNamespace

    protocol = tmp_path / "protocol.yaml"
    protocol.write_text("benchmark: libero_pro")
    calls = []
    monkeypatch.setenv("ROBORSI_VLA_URL", "http://vla.example:8000")
    monkeypatch.setenv("ROBORSI_SAM3_URL", "http://sam.example:8000")

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="127.0.0.1:12345")

    monkeypatch.setattr(bootstrap, "_run", run)
    monkeypatch.setattr(bootstrap, "_wait_service", lambda *_: None)
    prepared = SimpleNamespace(
        benchmarks=["libero_pro"],
        images={"libero_pro": "fixture"},
        configurations={},
        log_dir=None,
        work_dir=tmp_path,
    )
    with bootstrap.benchmark_services(
        prepared, pro_eval_config=protocol, framework="vla_sam3"
    ):
        pass
    command = next(c for c in calls if c[:2] == ["docker", "run"])
    assert "ROBORSI_VLA_URL=http://vla.example:8000" in command
    assert command[command.index("--framework") + 1] == "vla_sam3"
    assert command[command.index("--eval-config") + 1] == "/roborsi/pro-eval.yaml"
    assert "--legacy-image-env" in command
    assert any("readonly" in v and str(protocol) in v for v in command)
    assert calls[-1][:3] == ["docker", "rm", "--force"]


def test_tool_urls_from_config_reach_owned_benchmark_container(tmp_path, monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        output = "127.0.0.1:12345\n" if command[1] == "port" else "container\n"
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    monkeypatch.setattr(bootstrap, "_run", run)
    monkeypatch.setattr(bootstrap, "_wait_service", lambda *args: None)
    for name in ("ROBORSI_VLA_URL", "ROBORSI_SAM3_URL"):
        monkeypatch.delenv(name, raising=False)
    prepared = bootstrap.Prepared(
        benchmarks=("libero_pro",),
        work_dir=tmp_path,
        presets={},
        env={},
        images={"libero_pro": "fixture"},
    )
    urls = {"vla": "http://vla.example:8000", "sam3": "http://sam.example:8001"}
    with bootstrap.benchmark_services(
        prepared, framework="vla_sam3", service_tool_urls=urls
    ):
        pass
    assert "ROBORSI_VLA_URL=http://vla.example:8000" in calls[0]
    assert "ROBORSI_SAM3_URL=http://sam.example:8001" in calls[0]


@pytest.mark.parametrize("extended", [False, True])
def test_source_bundle_selection_controls_build_files(tmp_path, monkeypatch, extended):
    config = load_benchmark(
        "libero_pro",
        **(
            {"path": bootstrap.ROOT / "configs/benchmarks/libero_pro_80.yaml"}
            if extended
            else {}
        ),
    )
    bundle = config.environment["native_bundle"]
    calls = []
    monkeypatch.setattr(
        bootstrap.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 1),
    )

    def run(command, **kwargs):
        context = Path(command[-1])
        dockerfile = (context / "Dockerfile").read_text()
        for name in (
            "pro_protocol.py",
            "image_tools.py",
            "image_tool_adapter.py",
            "image_vla_bridge.py",
        ):
            assert (
                context / "benchmark-runtime/robots/libero" / name
            ).exists() is extended
        assert (
            "ENV ROBORSI_LIBERO_SOURCE_BUNDLE=libero_pro_image_tools" in dockerfile
        ) is extended
        assert (
            "RUN cp -a /opt/roborsi /opt/roborsi-image-tools" in dockerfile
        ) is extended
        assert (
            f"verify_benchmark_runtime(Path('/opt/roborsi'), '{bundle}')" in dockerfile
        )
        calls.append(command)

    monkeypatch.setattr(bootstrap, "_run", run)
    image = bootstrap._build_three_image(
        "libero_pro", tmp_path, IMAGES["libero_pro"], config
    )
    assert image.startswith(f"roborsi-onecmd-{bundle}:")
    assert len(calls) == 1
