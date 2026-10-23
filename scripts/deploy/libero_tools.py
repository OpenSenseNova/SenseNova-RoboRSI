#!/usr/bin/env python3
# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
"""Prepare public Pi0.5/SAM3 weights and own their service lifecycle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from contextlib import ExitStack, contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
IMAGE_ROOTS = (Path("/opt/roborsi-image-tools"), Path("/opt/roborsi"))


def image_source_plan(work):
    """Use the same public image and source lock as the explicit PRO profile."""
    import yaml

    config = yaml.safe_load(
        (ROOT / "configs/benchmarks/libero_pro_80.yaml").read_text()
    )
    image = config["environment"]["image"]
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("PRO tool source image must be pinned by digest")
    lock = json.loads((ROOT / "robots/libero/image_tools.lock.json").read_text())
    hashes = {lock["source_file"]: lock["sha256"], **lock["service_sources"]}
    identity = hashlib.sha256(
        json.dumps([image, hashes], sort_keys=True).encode()
    ).hexdigest()
    return {
        "image": image,
        "destination": str(work / "image-sources" / identity[:24]),
        "sha256": hashes,
    }


def verify_image_sources(root, hashes):
    for name, expected in hashes.items():
        path = Path(root) / name
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(
                f"image service source is missing or differs from lock: {path}"
            )


def prepare_image_sources(work):
    """Reuse verified image code or export it without starting a Docker service."""
    plan = image_source_plan(work)
    explicit = os.environ.get("ROBORSI_IMAGE_TOOLS_ROOT")
    if explicit:
        root = Path(explicit).resolve()
        verify_image_sources(root, plan["sha256"])
        return root
    destination = Path(plan["destination"])
    for root in (*IMAGE_ROOTS, destination):
        if root.is_dir():
            try:
                verify_image_sources(root, plan["sha256"])
            except ValueError:
                if root == destination:
                    raise  # Never reuse or overwrite a corrupted source cache.
            else:
                print(f"Using verified image service sources: {root}", flush=True)
                return root
    if shutil.which("docker") is None:
        raise RuntimeError(
            "Docker is required to export the pinned PRO service sources; "
            "inside the published PRO image, /opt/roborsi is reused automatically"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = plan["image"]
    inspected = subprocess.run(
        ["docker", "image", "inspect", image], capture_output=True
    )
    if inspected.returncode:
        print(f"Downloading PRO service source image: {image}", flush=True)
        subprocess.run(
            ["docker", "pull", "--platform", "linux/amd64", image], check=True
        )
    # Export through an unstarted temporary container; remove only that container.
    container = subprocess.check_output(
        [
            "docker",
            "create",
            "--platform",
            "linux/amd64",
            "--entrypoint",
            "/bin/true",
            image,
        ],
        text=True,
    ).strip()
    try:
        with tempfile.TemporaryDirectory(
            prefix="export-", dir=destination.parent
        ) as tmp:
            subprocess.run(
                ["docker", "cp", f"{container}:/opt/roborsi/.", tmp], check=True
            )
            verify_image_sources(Path(tmp), plan["sha256"])
            shutil.move(tmp, destination)
    finally:
        subprocess.run(
            ["docker", "rm", container], check=True, stdout=subprocess.DEVNULL
        )
    (work / "image-source.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(f"Prepared image service sources: {destination}", flush=True)
    return destination


def settings(config, work):
    import yaml

    return model_settings(yaml.safe_load(Path(config).read_text()))["deployment"]


def model_settings(document):
    """Resolve RLinf/SAM3 model settings without starting or downloading anything."""
    from roborsi.utils.config_values import reject_secrets
    from robots.libero.vla_client import endpoint_url

    reject_secrets(document)
    deployment = dict(document.get("tool_runtime", {}))
    urls = {}
    for name in ("vla", "sam3"):
        item = dict(document.get(name) or {})
        if name == "vla" and item.pop("backend", None) != "rlinf":
            raise ValueError("RLinf/SAM3 tools require vla.backend: rlinf")
        url = os.environ.get(f"ROBORSI_{name.upper()}_URL") or item.pop(
            "endpoint", None
        )
        item.pop("endpoint", None)
        if url:
            urls[name] = endpoint_url(url)
        deployment[name] = item
    if urls and len(urls) != 2:
        raise ValueError(
            "provide both VLA and SAM3 endpoints, or leave both unset for deployment"
        )
    if set(deployment) != {
        "python",
        "torch_index_url",
        "bind_host",
        "advertise_host",
        "startup_timeout_s",
        "vla",
        "sam3",
    }:
        raise ValueError(
            "model tool deployment fields differ from the supported schema"
        )
    if (
        type(deployment["startup_timeout_s"]) is not int
        or deployment["startup_timeout_s"] <= 0
    ):
        raise ValueError("startup_timeout_s must be a positive integer")
    for name in ("vla", "sam3"):
        tool = deployment[name]
        if set(tool) != {"repo_id", "revision", "checkpoint", "cuda_device", "port"}:
            raise ValueError(f"invalid {name} deployment fields")
        if (
            not isinstance(tool["revision"], str)
            or len(tool["revision"]) != 40
            or any(c not in "0123456789abcdef" for c in tool["revision"])
        ):
            raise ValueError("weight revision must be a pinned commit")
        if type(tool["port"]) is not int or not 1 <= tool["port"] <= 65535:
            raise ValueError("service port must be within 1..65535")
        if type(tool["cuda_device"]) is not int or tool["cuda_device"] < 0:
            raise ValueError("cuda_device must be a nonnegative integer")
        path = Path(tool["checkpoint"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("checkpoint must stay inside --work-dir")
    if deployment["vla"]["port"] == deployment["sam3"]["port"]:
        raise ValueError("VLA and SAM3 need distinct ports")
    return {"deployment": deployment, "urls": urls}


def preparation_commands(d, work):
    python = str(work / "venv/bin/python")
    commands = [
        [d["python"], "-m", "venv", str(work / "venv")],
        [
            python,
            "-m",
            "pip",
            "install",
            "torch==2.7.1",
            "torchvision==0.22.1",
            "--index-url",
            d["torch_index_url"],
        ],
        [
            python,
            "-m",
            "pip",
            "install",
            "-r",
            str(ROOT / "scripts/deploy/libero-tools-requirements.txt"),
        ],
    ]
    download = "from huggingface_hub import snapshot_download; import json,sys; snapshot_download(**json.loads(sys.argv[1]))"
    for name in ("vla", "sam3"):
        item = d[name]
        destination = work / item["checkpoint"]
        kwargs = {
            "repo_id": item["repo_id"],
            "revision": item["revision"],
            "local_dir": str(destination if name == "vla" else destination.parent),
        }
        if name == "vla":
            kwargs["allow_patterns"] = [
                "metadata.pt",
                "model.safetensors",
                "physical-intelligence/libero/norm_stats.json",
            ]
        else:
            kwargs["allow_patterns"] = [destination.name]
        commands.append([python, "-c", download, json.dumps(kwargs)])
    return commands


def service_command(d, work, name):
    module = "pi05_vla_server" if name == "vla" else "sam3_server"
    item = d[name]
    command = [
        str(work / "venv/bin/python"),
        "-m",
        "robots.libero." + module,
        "--transport",
        "http",
        "--host",
        d["bind_host"],
        "--port",
        str(item["port"]),
        "--cuda-device",
        str(item["cuda_device"]),
    ]
    if name == "vla":
        command += [
            "--api",
            "openpi",
            "--embodiment",
            "libero",
            "--model-backend",
            "openpi_pytorch",
            "--model-path",
            str(work / item["checkpoint"]),
        ]
    return command


def prepare(d, work):
    work.mkdir(parents=True, exist_ok=True)
    prepare_image_sources(work)
    # A successful marker prevents reinstalling dependencies on every run.
    requirements = ROOT / "scripts/deploy/libero-tools-requirements.txt"
    identity = hashlib.sha256(
        (json.dumps(d, sort_keys=True) + requirements.read_text()).encode()
    ).hexdigest()
    marker = work / "prepared.json"
    expected = [
        work / d["vla"]["checkpoint"] / "model.safetensors",
        work / d["vla"]["checkpoint"] / "metadata.pt",
        work / d["vla"]["checkpoint"] / "physical-intelligence/libero/norm_stats.json",
        work / d["sam3"]["checkpoint"],
    ]
    if (
        marker.exists()
        and json.loads(marker.read_text()).get("identity") == identity
        and all(p.is_file() and p.stat().st_size for p in expected)
        and (work / "venv/bin/python").is_file()
    ):
        print(f"Reusing prepared VLA/SAM3 environment and weights: {work}", flush=True)
        return
    print(f"Preparing isolated VLA/SAM3 environment and weights: {work}", flush=True)
    for command in preparation_commands(d, work):
        subprocess.run(command, check=True, cwd=ROOT)
    if not all(p.is_file() and p.stat().st_size for p in expected):
        raise RuntimeError("downloaded checkpoint files are incomplete")
    freeze = subprocess.check_output(
        [str(work / "venv/bin/python"), "-m", "pip", "freeze"], text=True
    )
    (work / "requirements-resolved.txt").write_text(freeze)
    marker.write_text(
        json.dumps({"identity": identity, "deployment": d}, indent=2) + "\n"
    )


def healthy(port, host="127.0.0.1", *, name="sam3"):
    if name == "vla":
        try:
            from robots.libero.image_tools import OpenPiExecutionClient

            OpenPiExecutionClient(f"http://{host}:{port}").call("healthz", timeout_s=2)
            return True
        except (OSError, ValueError, RuntimeError):
            return False
    request = urllib.request.Request(
        f"http://{host}:{port}/call",
        data=json.dumps(
            {"method": "healthz", "args": [], "kwargs": {}, "session_id": None}
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=2) as response:
            payload = json.load(response)
        return payload.get("ok") is True
    except (OSError, ValueError):
        return False


def terminate(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


@contextmanager
def services(d, work, *, log_dir=None):
    import socket

    source_root = prepare_image_sources(work)
    host = d["advertise_host"]
    if not host or any(c in host for c in "<>/:"):
        raise ValueError(
            "set advertise_host to a host/IP reachable from the Benchmark container"
        )
    # Check before spawning; never reuse/terminate a process owned by another run.
    for name in ("vla", "sam3"):
        with socket.socket() as sock:
            sock.bind((d["bind_host"], d[name]["port"]))
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env["PYTHONPATH"] = str(ROOT)
    env["ROBORSI_IMAGE_TOOLS_ROOT"] = str(source_root)
    env["SAM3_CHECKPOINT_PATH"] = str(work / d["sam3"]["checkpoint"])
    run_dir = log_dir or work / "logs" / time.strftime("%Y%m%dT%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    with ExitStack() as stack:
        children = []
        for name in ("vla", "sam3"):
            log = stack.enter_context((run_dir / (name + ".log")).open("w"))
            child = subprocess.Popen(
                service_command(d, work, name),
                env=env,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            stack.callback(terminate, child)
            children.append((name, child))
        deadline = time.monotonic() + d["startup_timeout_s"]
        health_host = "127.0.0.1" if d["bind_host"] == "0.0.0.0" else d["bind_host"]
        while True:
            if any(p.poll() is not None for _, p in children):
                raise RuntimeError(f"service exited; inspect {run_dir}")
            if all(
                healthy(d[name]["port"], health_host, name=name) for name, _ in children
            ):
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"service startup timed out; inspect {run_dir}")
            time.sleep(1)
        urls = {name: f"http://{host}:{d[name]['port']}" for name in ("vla", "sam3")}
        (run_dir / "services.json").write_text(json.dumps(urls, indent=2) + "\n")
        print("Services ready:", json.dumps(urls), flush=True)
        yield urls


def run_services(d, work, config, *, evaluate, extra):
    with services(d, work) as urls:
        if evaluate:
            env = os.environ.copy()
            env.update(ROBORSI_VLA_URL=urls["vla"], ROBORSI_SAM3_URL=urls["sam3"])
            return subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/evaluation/evaluate_model.py"),
                    "--benchmark",
                    "libero_pro",
                    "--benchmark-config",
                    str(ROOT / "configs/benchmarks/libero_pro_80.yaml"),
                    "--model-config",
                    str(config),
                    "--agent-config",
                    str(
                        ROOT
                        / "configs/agents/libero/sensenova_roborsi_27b_vla_sam3.yaml"
                    ),
                    *extra,
                ],
                env=env,
                cwd=ROOT,
            ).returncode
        while True:
            time.sleep(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "serve", "run"])
    parser.add_argument(
        "--model-config",
        dest="config",
        type=Path,
        default=ROOT / "configs/models/roborsi_27b.yaml",
    )
    parser.add_argument("--work-dir", type=Path, default=ROOT / "work/libero_tools")
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args(argv)
    if extra[:1] == ["--"]:
        extra = extra[1:]
    if extra and args.action != "run":
        parser.error("evaluation arguments require run")
    args.config = args.config.resolve()
    work = args.work_dir.resolve()
    d = settings(args.config, work)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "image_sources": image_source_plan(work),
                    "prepare": preparation_commands(d, work),
                    "services": {
                        name: service_command(d, work, name) for name in ("vla", "sam3")
                    },
                    "evaluation_arguments": extra,
                },
                indent=2,
            )
        )
        return 0
    if sys.platform != "linux":
        parser.error("tool services require Linux and NVIDIA CUDA GPUs")

    def interrupted(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    if args.action == "run":
        host = d["advertise_host"]
        if not host or any(c in host for c in "<>/:"):
            parser.error(
                "set tool_runtime.advertise_host before downloading or running"
            )
    prepare(d, work)
    if args.action == "prepare":
        return 0
    return run_services(
        d, work, args.config, evaluate=args.action == "run", extra=extra
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
