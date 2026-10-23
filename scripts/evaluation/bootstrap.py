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

"""Prepare pinned environments and task-scoped ICL for model evaluation."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from scripts.evaluation.configuration import (
    BenchmarkConfig,
    catalog_setting,
    load_benchmark,
    resolve_model,
)
from scripts.evaluation.downloads import download_snapshot

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Prepared:
    benchmarks: tuple[str, ...]
    work_dir: Path
    presets: dict[str, str]
    env: dict[str, str]
    images: dict[str, str]
    standard_asset: Path | None = None
    configurations: dict[str, BenchmarkConfig] = field(default_factory=dict)
    vla: dict = field(default_factory=dict)
    log_dir: Path | None = None


def _run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    kwargs.setdefault("check", True)
    return subprocess.run(command, text=True, **kwargs)


def preparation_plan(benchmarks, work_dir: Path, *, configurations=None) -> dict:
    """Read configured downloads without creating files or contacting services."""
    configs = configurations or {name: load_benchmark(name) for name in benchmarks}
    from scripts.evaluation import vla

    return {
        "benchmarks": list(benchmarks),
        "vla": vla.plan(
            {name: str(config.agent) for name, config in configs.items()},
            models={
                name: resolve_model(config.model) for name, config in configs.items()
            },
            parameter_overrides={
                name: config.agent_overrides.get("parameters", {})
                for name, config in configs.items()
            },
        ),
        "evaluation": {name: config.evaluation for name, config in configs.items()},
        "work_dir": str(work_dir),
        "service_source": "current checkout; enhanced LIBERO overlays its service on the pinned native image",
        "images": {
            name: config.environment.get(
                "image", f"build benchmarks/{name}/Dockerfile from this checkout"
            )
            for name, config in configs.items()
        },
        "icl": {
            name: (
                "embedded sealed packages (migration_subset)"
                if config.environment["kind"] == "libero_three"
                else "{repo_id}@{revision}/{prefix}".format(**config.environment["icl"])
                if name == "robocasa365"
                else "no released standard-LIBERO ICL package"
            )
            for name, config in configs.items()
        },
        "configurations": {
            name: {
                "benchmark": str(c.path),
                "model": str(c.model),
                "agent": str(c.agent),
            }
            for name, c in configs.items()
        },
        "catalog_environment": {
            name: {
                key: catalog_setting(config, key)
                for key in config.environment["catalog"]
            }
            for name, config in configs.items()
        },
    }


def _ensure_image(image: str) -> str:
    inspected = subprocess.run(
        ["docker", "image", "inspect", image], capture_output=True, text=True
    )
    if inspected.returncode:
        print(f"Downloading environment: {image}", file=sys.stderr)
        _run(["docker", "pull", "--platform", "linux/amd64", image])
    return image


def _build_image(
    benchmark: str, work_dir: Path, config: BenchmarkConfig | None = None
) -> str:
    from benchmarks.package import prepare_context

    config = config or load_benchmark(benchmark)
    build_args = config.environment.get("build_args", {})
    with tempfile.TemporaryDirectory(
        prefix=f"build-{benchmark}-", dir=work_dir
    ) as temporary:
        context = prepare_context(
            benchmark, Path(temporary) / "context", source_root=ROOT
        )
        digest = json.loads((context / "build.json").read_text())["context_sha256"]
        identity = hashlib.sha256(
            json.dumps(
                {"source": digest, "build_args": build_args}, sort_keys=True
            ).encode()
        ).hexdigest()
        image = f"roborsi-onecmd-{benchmark}:{identity[:24]}"
        inspected = subprocess.run(
            ["docker", "image", "inspect", image], capture_output=True, text=True
        )
        if inspected.returncode:
            print(f"Building environment: {image}", file=sys.stderr)
            _run(
                [
                    "docker",
                    "build",
                    "--platform",
                    "linux/amd64",
                    "-t",
                    image,
                    *[
                        part
                        for key, value in sorted(build_args.items())
                        for part in ("--build-arg", f"{key}={value}")
                    ],
                    str(context),
                ]
            )
        return image


def _ensure_tokenizer(agent: Path) -> None:
    from roborsi.agents.config import AgentConfig

    dependencies = AgentConfig.from_file(agent).document.get("dependencies", [])
    pins = [
        value
        for value in dependencies
        if re.fullmatch(r"tokenizers==[0-9]+(?:\.[0-9]+)+", value)
    ]
    if len(pins) != 1:
        raise ValueError(
            "Sealed LIBERO Agent must declare one pinned tokenizers dependency"
        )
    package = pins[0]
    try:
        installed = importlib.metadata.version("tokenizers")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed == package.split("==", 1)[1]:
        return
    if sys.prefix == sys.base_prefix:
        raise RuntimeError(
            f"Activate a Python virtual environment for automatic {package} installation"
        )
    _run([sys.executable, "-m", "pip", "install", package])


def _build_three_image(
    benchmark: str, work_dir: Path, base: str, config: BenchmarkConfig | None = None
) -> str:
    """Keep the published native stack but serve this checkout's protocol code."""
    from roborsi.benchmarks.registry import directory_sha256
    from roborsi.benchmarks.runtime_source import (
        benchmark_source_digest,
        benchmark_source_files,
    )

    native = (config or load_benchmark(benchmark)).environment["native_bundle"]
    before = benchmark_source_digest(ROOT, native)
    with tempfile.TemporaryDirectory(
        prefix="build-libero-service-", dir=work_dir
    ) as temporary:
        context = Path(temporary)
        runtime = context / "benchmark-runtime"
        for name in benchmark_source_files(native):
            target = runtime / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        if (
            benchmark_source_digest(runtime, native) != before
            or benchmark_source_digest(ROOT, native) != before
        ):
            raise ValueError("Benchmark source changed while preparing the image")
        preserve_image_tools = False
        if config and native == "libero_pro_image_tools":
            from robots.libero.image_tools import configured

            preserve_image_tools = configured(config.agent)
        preserve = (
            "RUN cp -a /opt/roborsi /opt/roborsi-image-tools\n"
            if preserve_image_tools
            else ""
        )
        source_env = (
            "ENV ROBORSI_LIBERO_SOURCE_BUNDLE=libero_pro_image_tools\n"
            if native == "libero_pro_image_tools"
            else ""
        )
        (context / "Dockerfile").write_text(
            f"FROM {base}\n{preserve}COPY benchmark-runtime/ /opt/roborsi/\n{source_env}"
            'RUN python -B -c "from pathlib import Path; '
            "from roborsi.benchmarks.runtime_source import verify_benchmark_runtime; "
            f"verify_benchmark_runtime(Path('/opt/roborsi'), '{native}')\"\n"
            "ENTRYPOINT []\n",
            encoding="utf-8",
        )
        image = f"roborsi-onecmd-{native}:{directory_sha256(context)[:24]}"
        inspected = subprocess.run(
            ["docker", "image", "inspect", image], capture_output=True, text=True
        )
        if inspected.returncode:
            print(f"Building current LIBERO service: {image}", file=sys.stderr)
            _run(
                [
                    "docker",
                    "build",
                    "--platform",
                    "linux/amd64",
                    "-t",
                    image,
                    str(context),
                ]
            )
        return image


def _verify_three_assets(root: Path, config: BenchmarkConfig | None = None) -> None:
    from benchmarks.libero_three_combined.prepare_agent_assets import _verify_packages

    lock = json.loads(
        (config or load_benchmark("libero_pro"))
        .environment["icl"]["asset_lock"]
        .read_text()
    )
    for group in ("longpp", "ultra", "pro"):
        _verify_packages(group, root / group / "packages", lock, repository=ROOT)
    tokenizer = root / "tokenizer.json"
    if (
        tokenizer.is_symlink()
        or not tokenizer.is_file()
        or hashlib.sha256(tokenizer.read_bytes()).hexdigest()
        != lock["assets"]["context_tokenizer"]["sha256"]
    ):
        raise ValueError("ICL context tokenizer checksum mismatch")


def _prepare_three_assets(
    work_dir: Path, image: str, config: BenchmarkConfig | None = None
) -> Path:
    from benchmarks.libero_three_combined.prepare_agent_assets import prepare

    destination = work_dir / "assets/libero-three"
    if destination.exists():
        _verify_three_assets(destination, config)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="libero-icl-", dir=destination.parent
    ) as temporary:
        extracted = Path(temporary) / "extracted"
        extracted.mkdir()
        container = f"roborsi-icl-{uuid4().hex[:12]}"
        try:
            _run(["docker", "create", "--name", container, image], capture_output=True)
            _run(
                [
                    "docker",
                    "cp",
                    f"{container}:/opt/roborsi-agent-assets/.",
                    str(extracted),
                ]
            )
        finally:
            subprocess.run(
                ["docker", "rm", "--force", container], capture_output=True, check=False
            )
        staged = prepare(
            {
                group: extracted / group / "packages"
                for group in ("longpp", "ultra", "pro")
            },
            extracted / "tokenizer.json",
            Path(temporary) / "verified",
            repository=ROOT,
            asset_lock=(config or load_benchmark("libero_pro")).environment["icl"][
                "asset_lock"
            ],
        )
        if destination.exists():
            _verify_three_assets(destination, config)
        else:
            staged.rename(destination)
    return destination


def _prepare_standard_assets(
    work_dir: Path, config: BenchmarkConfig | None = None
) -> Path:
    from roborsi.benchmarks.registry import directory_sha256

    source = (config or load_benchmark("libero")).environment["assets"]
    root = work_dir / "assets/libero"
    if not (root / "asset.json").is_file():
        _run(
            [
                sys.executable,
                "-m",
                "benchmarks.libero.download_assets",
                "--output-dir",
                str(root),
                "--repo-id",
                source["repo_id"],
                "--revision",
                source["revision"],
            ],
            cwd=ROOT,
        )
    asset = json.loads((root / "asset.json").read_text())
    snapshot = root / asset["path"]
    if (
        asset.get("version") != source["revision"]
        or not snapshot.resolve().is_relative_to(root.resolve())
        or snapshot.is_symlink()
        or directory_sha256(snapshot) != asset.get("sha256")
    ):
        raise ValueError(
            "Standard LIBERO asset revision or checksum mismatch; choose a new work directory"
        )
    return snapshot


def _write_preset(
    work_dir: Path,
    benchmark: str,
    document: dict,
    config: BenchmarkConfig | None = None,
) -> str:
    def merge(base, patch):
        result = copy.deepcopy(base)
        for key, value in patch.items():
            result[key] = (
                merge(result[key], value)
                if isinstance(result.get(key), dict) and isinstance(value, dict)
                else copy.deepcopy(value)
            )
        return result

    config = config or load_benchmark(benchmark)
    document = merge(config.agent_overrides, document)
    # AgentConfig requires code, inherited presets and resources inside source_root.
    document["schema_version"] = 1
    from roborsi.agents.config import AgentConfig

    document["source_root"] = os.path.commonpath(
        [str(ROOT), str(work_dir), str(AgentConfig.from_file(config.agent).source_root)]
    )
    destination = work_dir / "presets" / f"{benchmark}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return str(destination)


def prepare_agents(benchmarks, work_dir: Path, *, configurations=None) -> Prepared:
    """Prepare Agent resources independently of simulator ownership/transport."""
    configs = configurations or {name: load_benchmark(name) for name in benchmarks}
    from roborsi.agents.config import AgentConfig

    for config in configs.values():
        AgentConfig.from_file(config.agent)
    from scripts.evaluation import vla

    # Check enabled profiles and factories before downloading native environments.
    models = {name: resolve_model(config.model) for name, config in configs.items()}
    vla.plan(
        {name: str(config.agent) for name, config in configs.items()},
        models=models,
        parameter_overrides={
            name: config.agent_overrides.get("parameters", {})
            for name, config in configs.items()
        },
    )
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    prepared = Prepared(tuple(benchmarks), work_dir, {}, {}, {}, configurations=configs)
    three_assets = None
    for name in benchmarks:
        config = configs[name]
        environment = config.environment
        agent = AgentConfig.from_file(config.agent)
        if agent.factory in {
            "robots.libero.three_benchmark_agent:make_three_benchmark_agent",
            "robots.libero.roborsi_27b_agent:make_agent",
        }:
            _ensure_tokenizer(config.agent)
            if three_assets is None:
                base_image = environment["image"]
                if not (work_dir / "assets/libero-three").exists():
                    base_image = _ensure_image(base_image)
                three_assets = _prepare_three_assets(work_dir, base_image, config)
            group, variable = (
                environment["icl"]["group"],
                environment["icl"]["root_env"],
            )
            lock = json.loads(environment["icl"]["asset_lock"].read_text())
            prepared.env[variable] = str(three_assets / group / "packages")
            prepared.presets[name] = _write_preset(
                work_dir,
                name,
                {
                    "extends": config.agent.as_posix(),
                    "runtime": {
                        "context": {
                            "tokenizer_file": str(three_assets / "tokenizer.json"),
                            "tokenizer_sha256": lock["assets"]["context_tokenizer"][
                                "sha256"
                            ],
                        },
                    },
                },
                config,
            )
        elif (
            agent.factory == "robots.robocasa.model_eval_agent:make_agent"
            and "icl_package" not in agent.resources
        ):
            package = download_snapshot(
                **environment["icl"],
                destination=work_dir / "assets/robocasa365-icl",
            )
            prepared.presets[name] = _write_preset(
                work_dir,
                name,
                {
                    "extends": config.agent.as_posix(),
                    "resources": {"icl_package": str(package)},
                    "parameters": {
                        "icl_manifest_sha256": environment["icl"]["manifest_sha256"],
                        "icl_revision": environment["icl"]["revision"],
                    },
                },
                config,
            )
        else:
            prepared.presets[name] = _write_preset(
                work_dir, name, {"extends": config.agent.as_posix()}, config
            )
    prepared.vla = vla.prepare_agents(prepared.presets, work_dir, models=models)
    return prepared


def prepare_benchmarks(benchmarks, work_dir: Path, *, configurations=None) -> Prepared:
    """Prepare Agent resources and locally owned native simulator images."""
    try:
        _run(["docker", "info"], capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        raise RuntimeError(
            "Docker is unavailable. Start Docker on Linux/WSL and retry."
        ) from None
    prepared = prepare_agents(benchmarks, work_dir, configurations=configurations)
    for name, config in prepared.configurations.items():
        environment = config.environment
        if environment["kind"] == "libero_three":
            base = _ensure_image(environment["image"])
            image = _build_three_image(name, prepared.work_dir, base, config)
        else:
            if name == "libero":
                prepared.standard_asset = _prepare_standard_assets(
                    prepared.work_dir, config
                )
            image = _build_image(name, prepared.work_dir, config)
        prepared.images[name] = image
    return prepared


def _wait_service(url: str, container: str, timeout_s: float = 900) -> None:
    started = time.monotonic()
    next_status = started + 15
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/v1/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        state = _run(
            ["docker", "inspect", "--format", "{{.State.Running}}", container],
            capture_output=True,
        )
        if state.stdout.strip() != "true":
            logs = _run(
                ["docker", "logs", "--tail", "40", container], capture_output=True
            )
            raise RuntimeError(
                f"Benchmark service exited: {logs.stdout}\n{logs.stderr}"
            )
        if time.monotonic() >= next_status:
            print(
                f"Waiting for Benchmark readiness: elapsed={time.monotonic() - started:.0f}s",
                file=sys.stderr,
                flush=True,
            )
            next_status = time.monotonic() + 15
        time.sleep(1)
    raise RuntimeError(
        f"Benchmark service did not become ready in {timeout_s:g}s: {container}"
    )


@contextmanager
def benchmark_services(
    prepared: Prepared,
    *,
    libero_suite=None,
    robocasa_cells=None,
    pro_eval_config=None,
    framework=None,
    service_tool_urls=None,
    image_observation_tools=False,
):
    """Own and stop only the containers created for this invocation."""
    owned = []
    endpoints = {}
    tool_urls = {}
    if framework == "vla_sam3":
        from robots.libero.image_tools import validate_tool_urls

        tool_urls = validate_tool_urls(
            service_tool_urls
            or {
                name: os.environ.get(f"ROBORSI_{name.upper()}_URL", "")
                for name in ("vla", "sam3")
            }
        )
    try:
        for name in prepared.benchmarks:
            config = prepared.configurations.get(name) or load_benchmark(name)
            environment = config.environment
            kind = environment["kind"]
            container = f"roborsi-eval-{name.replace('_', '-')}-{uuid4().hex[:12]}"
            owned.append(container)
            command = [
                "docker",
                "run",
                "-d",
                "--platform",
                "linux/amd64",
                "--name",
                container,
                "-p",
                "127.0.0.1::8000",
            ]
            if environment["gpus"]:
                command += ["--gpus", str(environment["gpus"])]
            if kind == "libero_three":
                command += [
                    "--entrypoint",
                    "python",
                    "-e",
                    "MUJOCO_GL=egl",
                    "-e",
                    "PYOPENGL_PLATFORM=egl",
                ]
            elif name == "libero":
                command += [
                    "-e",
                    f"ROBORSI_LIBERO_SUITE={libero_suite or catalog_setting(config, 'suite')}",
                    "--mount",
                    f"type=bind,src={prepared.standard_asset},dst=/roborsi/assets/libero,readonly",
                ]
            else:
                command += [
                    "-e",
                    f"ROBORSI_ROBOCASA_CELLS={robocasa_cells or catalog_setting(config, 'cells')}",
                ]
            if name == "libero_pro" and pro_eval_config is not None:
                command += [
                    "--mount",
                    f"type=bind,src={Path(pro_eval_config).resolve()},dst=/roborsi/pro-eval.yaml,readonly",
                ]
            if name == "libero_pro" and framework:
                if any("host.docker.internal" in url for url in tool_urls.values()):
                    command += ["--add-host", "host.docker.internal:host-gateway"]
                for tool, url in tool_urls.items():
                    command += ["-e", f"ROBORSI_{tool.upper()}_URL={url}"]
            command.append(prepared.images[name])
            if kind == "libero_three":
                command += [
                    "-B",
                    "-m",
                    "robots.libero.three_benchmark_serve",
                    "--benchmark",
                    name,
                    "--profile",
                    catalog_setting(config, "profile"),
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "8000",
                ]
            if (
                name == "libero_pro"
                and config.environment.get("native_bundle") == "libero_pro_image_tools"
            ):
                command += ["--source-bundle", "libero_pro_image_tools"]
            if name == "libero_pro" and pro_eval_config is not None:
                command += [
                    "--eval-config",
                    "/roborsi/pro-eval.yaml",
                    "--legacy-image-env",
                ]
            if name == "libero_pro" and framework:
                command += ["--framework", framework]
            if name == "libero_pro" and image_observation_tools:
                command += ["--image-observation-tools"]
            print(f"Starting Benchmark service: {name}", file=sys.stderr)
            _run(command, capture_output=True)
            port = _run(
                ["docker", "port", container, "8000/tcp"], capture_output=True
            ).stdout.strip()
            url = f"http://{port}"
            _wait_service(url, container, environment.get("startup_timeout_s", 900))
            endpoints[name] = url
        yield endpoints
    finally:
        for container in reversed(owned):
            from scripts.evaluation.reporting import save_service_log

            save_service_log(
                container, prepared.log_dir or prepared.work_dir / "logs/services", _run
            )
            try:
                _run(
                    ["docker", "rm", "--force", container],
                    capture_output=True,
                    check=False,
                )
            except OSError:
                print(f"Could not stop managed container: {container}", file=sys.stderr)
