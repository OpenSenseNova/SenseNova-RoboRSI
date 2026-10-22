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

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from scripts.evaluation.downloads import download_snapshot

ROOT = Path(__file__).resolve().parents[2]
LIBERO_ASSET_REVISION = "3ba78404b48e8c70fa4ac9782d5aac8e5b46d55f"
ROBOCASA_REVISION = "e977b5ad667b4ce753ee947cbb0b95d4fc0fe98c"
ROBOCASA_MANIFEST = "88dfecd78fada13a6fc7aa2b13d1a7c61cb005451b72d53c0d3f25cc2f6a90bf"
_LONG_IMAGE_DIGEST = "814139261e914b114cb89a359c6a39d4f409a556f79928e986b6ade3dfe7dc7a"
IMAGES = {
    "libero_pro": "ghcr.io/opensensenova/sensenova-roborsi-libero-pro@sha256:"
    "a83417f1459a4c555c457caa82ef00a5de50191e3d8deb943894fa9285cff172",
    "libero_long_plus_plus": "ghcr.io/opensensenova/sensenova-roborsi-libero-long-plus-plus@sha256:"
    + _LONG_IMAGE_DIGEST,
    "libero_ultra_long": "ghcr.io/opensensenova/sensenova-roborsi-libero-ultra-long@sha256:"
    + _LONG_IMAGE_DIGEST,
}
_THREE = {
    "libero_pro": ("pro.yaml", "pro", "ROBORSI_LIBERO_PRO_ICL_ROOT"),
    "libero_long_plus_plus": (
        "long_plus_plus_v13.yaml",
        "longpp",
        "ROBORSI_LIBERO_LONGPP_ICL_ROOT",
    ),
    "libero_ultra_long": (
        "ultra_long_v13.yaml",
        "ultra",
        "ROBORSI_LIBERO_ULTRA_ICL_ROOT",
    ),
}


@dataclass
class Prepared:
    benchmarks: tuple[str, ...]
    work_dir: Path
    presets: dict[str, str]
    env: dict[str, str]
    images: dict[str, str]
    standard_asset: Path | None = None


def _run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    kwargs.setdefault("check", True)
    return subprocess.run(command, text=True, **kwargs)


def preparation_plan(benchmarks, work_dir: Path) -> dict:
    """Describe selected downloads without touching Docker, disk or the network."""
    return {
        "benchmarks": list(benchmarks),
        "work_dir": str(work_dir),
        "service_source": "current checkout; enhanced LIBERO overlays its service on the pinned native image",
        "images": {
            name: IMAGES.get(
                name, f"build benchmarks/{name}/Dockerfile from this checkout"
            )
            for name in benchmarks
        },
        "icl": {
            name: (
                "embedded sealed packages (migration_subset)"
                if name in _THREE
                else f"sensenova/SenseNova-RoboRSI-Packages@{ROBOCASA_REVISION}/robocasa365/v1.0.0"
                if name == "robocasa365"
                else "no released standard-LIBERO ICL package"
            )
            for name in benchmarks
        },
        "catalog_environment": {
            "ROBORSI_LIBERO_SUITE": os.environ.get(
                "ROBORSI_LIBERO_SUITE", "libero_spatial"
            ),
            "ROBORSI_ROBOCASA_CELLS": os.environ.get("ROBORSI_ROBOCASA_CELLS", "all"),
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


def _build_image(benchmark: str, work_dir: Path) -> str:
    from benchmarks.package import prepare_context

    with tempfile.TemporaryDirectory(
        prefix=f"build-{benchmark}-", dir=work_dir
    ) as temporary:
        context = prepare_context(
            benchmark, Path(temporary) / "context", source_root=ROOT
        )
        digest = json.loads((context / "build.json").read_text())["context_sha256"]
        image = f"roborsi-onecmd-{benchmark}:{digest[:24]}"
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
                    str(context),
                ]
            )
        return image


def _ensure_tokenizer() -> None:
    try:
        installed = importlib.metadata.version("tokenizers")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed == "0.22.2":
        return
    if sys.prefix == sys.base_prefix:
        raise RuntimeError(
            "Activate a Python virtual environment for automatic tokenizers==0.22.2 installation"
        )
    _run([sys.executable, "-m", "pip", "install", "tokenizers==0.22.2"])


def _build_three_image(benchmark: str, work_dir: Path, base: str) -> str:
    """Keep the published native stack but serve this checkout's protocol code."""
    from roborsi.benchmarks.registry import directory_sha256
    from roborsi.benchmarks.runtime_source import (
        benchmark_source_digest,
        benchmark_source_files,
    )

    native = "libero_pro" if benchmark == "libero_pro" else "libero_lilo"
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
        (context / "Dockerfile").write_text(
            f"FROM {base}\nCOPY benchmark-runtime/ /opt/roborsi/\n"
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


def _verify_three_assets(root: Path) -> None:
    from benchmarks.libero_three_combined.prepare_agent_assets import _verify_packages

    lock = json.loads(
        (ROOT / "benchmarks/libero_three_combined/assets.lock.json").read_text()
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


def _prepare_three_assets(work_dir: Path, image: str) -> Path:
    from benchmarks.libero_three_combined.prepare_agent_assets import prepare

    destination = work_dir / "assets/libero-three"
    if destination.exists():
        _verify_three_assets(destination)
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
        )
        if destination.exists():
            _verify_three_assets(destination)
        else:
            staged.rename(destination)
    return destination


def _prepare_standard_assets(work_dir: Path) -> Path:
    from roborsi.benchmarks.registry import directory_sha256

    root = work_dir / "assets/libero"
    if not (root / "asset.json").is_file():
        _run(
            [
                sys.executable,
                "-m",
                "benchmarks.libero.download_assets",
                "--output-dir",
                str(root),
            ],
            cwd=ROOT,
        )
    asset = json.loads((root / "asset.json").read_text())
    snapshot = root / asset["path"]
    if (
        asset.get("version") != LIBERO_ASSET_REVISION
        or not snapshot.resolve().is_relative_to(root.resolve())
        or snapshot.is_symlink()
        or directory_sha256(snapshot) != asset.get("sha256")
    ):
        raise ValueError(
            "Standard LIBERO asset revision or checksum mismatch; choose a new work directory"
        )
    return snapshot


def _write_preset(work_dir: Path, benchmark: str, document: dict) -> str:
    # AgentConfig requires code, inherited presets and resources inside source_root.
    document["schema_version"] = 1
    document["source_root"] = os.path.commonpath([str(ROOT), str(work_dir)])
    destination = work_dir / "presets" / f"{benchmark}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return str(destination)


def prepare_benchmarks(benchmarks, work_dir: Path) -> Prepared:
    """Download selected assets and ICL, and prepare images without episodes."""
    try:
        _run(["docker", "info"], capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        raise RuntimeError(
            "Docker is unavailable. Start Docker on Linux/WSL and retry."
        ) from None
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    prepared = Prepared(tuple(benchmarks), work_dir, {}, {}, {})
    three_assets = None
    for name in benchmarks:
        if name in _THREE:
            base_image = _ensure_image(IMAGES[name])
            _ensure_tokenizer()
            if three_assets is None:
                three_assets = _prepare_three_assets(work_dir, base_image)
            image = _build_three_image(name, work_dir, base_image)
            preset, group, variable = _THREE[name]
            lock = json.loads(
                (ROOT / "benchmarks/libero_three_combined/assets.lock.json").read_text()
            )
            prepared.env[variable] = str(three_assets / group / "packages")
            prepared.presets[name] = _write_preset(
                work_dir,
                name,
                {
                    "extends": (
                        ROOT / "robots/libero/agents/three_benchmark" / preset
                    ).as_posix(),
                    "agent": {"reasoning_effort": "none"},
                    "runtime": {
                        "llm": {
                            "openai_format": "chat",
                            "prompt_cache_mode": None,
                            "prompt_cache_key": None,
                            "prompt_cache_static_prefix": False,
                        },
                        "context": {
                            "tokenizer_file": str(three_assets / "tokenizer.json"),
                            "tokenizer_sha256": lock["assets"]["context_tokenizer"][
                                "sha256"
                            ],
                            "cache_initial_task": False,
                            "cache_initial_context": False,
                            "cache_latest_tool_feedback": False,
                        },
                    },
                },
            )
        elif name == "libero":
            prepared.standard_asset = _prepare_standard_assets(work_dir)
            image = _build_image(name, work_dir)
        elif name == "robocasa365":
            package = download_snapshot(
                repo_id="sensenova/SenseNova-RoboRSI-Packages",
                revision=ROBOCASA_REVISION,
                prefix="robocasa365/v1.0.0",
                destination=work_dir / "assets/robocasa365-icl",
                manifest_sha256=ROBOCASA_MANIFEST,
            )
            prepared.presets[name] = _write_preset(
                work_dir,
                name,
                {
                    "extends": (
                        ROOT / "configs/agents/robocasa365/icl.yaml"
                    ).as_posix(),
                    "resources": {"icl_package": str(package)},
                },
            )
            image = _build_image(name, work_dir)
        else:
            raise ValueError(f"Unsupported automatic benchmark: {name}")
        prepared.images[name] = image
    return prepared


def _wait_service(url: str, container: str, timeout_s: float = 900) -> None:
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
        time.sleep(1)
    raise RuntimeError(
        f"Benchmark service did not become ready in {timeout_s:g}s: {container}"
    )


@contextmanager
def benchmark_services(
    prepared: Prepared, *, libero_suite="libero_spatial", robocasa_cells="all"
):
    """Own and stop only the containers created for this invocation."""
    owned = []
    endpoints = {}
    try:
        for name in prepared.benchmarks:
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
            if name != "libero":
                command += ["--gpus", "all"]
            if name in _THREE:
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
                    f"ROBORSI_LIBERO_SUITE={libero_suite}",
                    "--mount",
                    f"type=bind,src={prepared.standard_asset},dst=/roborsi/assets/libero,readonly",
                ]
            else:
                command += ["-e", f"ROBORSI_ROBOCASA_CELLS={robocasa_cells}"]
            command.append(prepared.images[name])
            if name in _THREE:
                command += [
                    "-B",
                    "-m",
                    "robots.libero.three_benchmark_serve",
                    "--benchmark",
                    name,
                    "--profile",
                    "migration_subset",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "8000",
                ]
            print(f"Starting Benchmark service: {name}", file=sys.stderr)
            _run(command, capture_output=True)
            port = _run(
                ["docker", "port", container, "8000/tcp"], capture_output=True
            ).stdout.strip()
            url = f"http://{port}"
            _wait_service(url, container)
            endpoints[name] = url
        yield endpoints
    finally:
        for container in reversed(owned):
            try:
                _run(
                    ["docker", "rm", "--force", container],
                    capture_output=True,
                    check=False,
                )
            except OSError:
                print(f"Could not stop managed container: {container}", file=sys.stderr)
