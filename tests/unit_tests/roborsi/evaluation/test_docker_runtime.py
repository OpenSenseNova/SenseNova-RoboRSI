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

"""Docker launch commands honor the locked manifest and clean up on failure."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from threading import Event, Timer

import pytest

from roborsi.benchmarks.registry import BenchmarkRegistry, directory_sha256
from roborsi.evaluation.deadline import CaseCancelled, CaseDeadline, CaseTimeout
from roborsi.evaluation.docker_runtime import (
    AdapterContainer,
    DockerCommandError,
    DockerExecutor,
    _docker_command,
)
from roborsi.utils.rpc import RpcError
from tests.unit_tests.roborsi.benchmarks.test_registry_contracts import _manifest


class FakeDocker:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.present = False
        self.digest = "sha256:" + "b" * 64
        self.daemon_available = True
        self.inspection = None

    def __call__(self, args: list[str], timeout_s: float) -> str:
        self.calls.append(args)
        if args[1] == "version":
            if not self.daemon_available:
                raise DockerCommandError("Cannot connect to Docker daemon")
            return "28.0"
        if args[1:3] == ["image", "inspect"]:
            if not self.present:
                raise DockerCommandError("image not found")
            return json.dumps(
                {
                    "Id": "sha256:" + "e" * 64,
                    "RepoDigests": [f"example.org/demo@{self.digest}"],
                }
            )
        if args[1] == "pull":
            self.present = True
            return args[-1]
        if args[1] == "run":
            image = next(i for i, arg in enumerate(args) if "@sha256:" in arg)
            ip, port, service_port = args[args.index("--publish") + 1].split(":")
            mounts = []
            for i, arg in enumerate(args):
                if arg == "--mount":
                    parts = args[i + 1].split(",")
                    fields = dict(part.split("=", 1) for part in parts if "=" in part)
                    mounts.append(
                        {
                            "Type": "bind",
                            "Destination": fields["dst"],
                            "Source": fields["src"],
                            "RW": "readonly" not in parts,
                        }
                    )
            self.inspection = {
                "Image": "sha256:" + "e" * 64,
                "State": {"Running": True},
                "Config": {
                    "Cmd": args[image + 1 :],
                    "User": args[args.index("--user") + 1],
                    "Env": [
                        "TEST_CONFIG=fixture",
                        *Path(args[args.index("--env-file") + 1])
                        .read_text()
                        .splitlines(),
                    ],
                    "WorkingDir": "/app",
                    "Entrypoint": None,
                },
                "HostConfig": {
                    "ReadonlyRootfs": "--read-only" in args,
                    "NanoCpus": round(
                        float(args[args.index("--cpus") + 1]) * 1_000_000_000
                    ),
                    "Memory": int(args[args.index("--memory") + 1][:-1]) * 1024 * 1024,
                    "PortBindings": {
                        f"{service_port}/tcp": [{"HostIp": ip, "HostPort": port}]
                    },
                    "DeviceRequests": [],
                    "NetworkMode": "default",
                    "Tmpfs": {"/tmp": "rw,nosuid,nodev,size=1g"},
                },
                "Mounts": mounts,
            }
            return "c" * 64
        if args[1] == "inspect":
            return json.dumps(self.inspection)
        if args[1] == "rm":
            return "c" * 64
        raise AssertionError(f"unexpected Docker command: {args}")


class FakeRpc:
    def __init__(self, *, healthy: bool = True) -> None:
        self.healthy = healthy
        self.closed = False
        self.calls: list[str] = []

    def call(self, method: str, *args, **kwargs):
        self.calls.append(method)
        if method == "healthz" and not self.healthy:
            raise RpcError(method, "not ready")
        return {"status": "ok"}

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("container_user", [None, "1101:1202"])
def test_docker_executor_pins_image_and_mounts_assets(
    tmp_path: Path, container_user: str | None
) -> None:
    _, asset = _manifest(tmp_path)
    lock = BenchmarkRegistry([tmp_path]).lock("demo")
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    docker, rpc = FakeDocker(), FakeRpc()
    executor = DockerExecutor(
        command_runner=docker,
        client_factory=lambda endpoint: rpc,
        container_user=container_user,
    )

    with executor.start_adapter(lock, case_dir, host_port=41234) as container:
        assert container.container_id == "c" * 64
        assert container.client is not None

    assert [call[1] for call in docker.calls].count("pull") == 1
    command = next(call for call in docker.calls if call[1] == "run")
    assert command[command.index("--publish") + 1] == "127.0.0.1:41234:8765"
    assert command[command.index("--cpus") + 1] == "1"
    assert command[command.index("--memory") + 1] == "512m"
    assert "--pull=never" in command
    assert command[command.index("--user") + 1] == (
        container_user if container_user is not None else f"{os.getuid()}:{os.getgid()}"
    )
    mounts = [
        command[i + 1] for i, argument in enumerate(command) if argument == "--mount"
    ]
    assert all(str(case_dir) not in mount for mount in mounts)
    assert all("readonly" in mount for mount in mounts)
    assert f"type=bind,src={asset},dst=/roborsi/assets/tasks,readonly" in command
    assert command[-4:] == [
        lock.image_reference,
        "python",
        "-m",
        "benchmark.adapter",
    ]
    assert rpc.closed and "shutdown" in rpc.calls
    assert docker.calls[-1] == ["docker", "rm", "--force", "c" * 64]


def test_benchmark_tokens_are_private_and_do_not_change_comparison_settings(tmp_path):
    _manifest(tmp_path)
    lock = BenchmarkRegistry([tmp_path]).lock("demo")
    records, tokens = [], []
    for name in ("one", "two"):
        directory = tmp_path / name
        directory.mkdir()
        docker = FakeDocker()
        with DockerExecutor(
            command_runner=docker, client_factory=lambda _: FakeRpc()
        ).start_adapter(lock, directory) as service:
            tokens.append(
                next(
                    value.partition("=")[2]
                    for value in docker.inspection["Config"]["Env"]
                    if value.startswith("ROBORSI_BENCHMARK_TOKEN=")
                )
            )
            run = next(command for command in docker.calls if command[1] == "run")
            assert not Path(run[run.index("--env-file") + 1]).exists()
            assert tokens[-1] not in json.dumps(docker.calls)
        records.append(service.runtime_record)
        assert tokens[-1] not in json.dumps(records[-1].as_dict())
    assert tokens[0] != tokens[1]
    assert records[0].configuration == records[1].configuration


def test_docker_executor_mounts_verified_directory_asset_read_only(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _manifest(tmp_path)
    tree = manifest_path.parent / "assets" / "models"
    tree.mkdir()
    (tree / "policy.bin").write_bytes(b"weights")
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["assets"] = [
        {
            "name": "models",
            "version": "v1",
            "kind": "directory",
            "path": "assets/models",
            "sha256": directory_sha256(tree),
        }
    ]
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    lock = BenchmarkRegistry([tmp_path]).lock("demo")
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    docker = FakeDocker()
    executor = DockerExecutor(command_runner=docker, client_factory=lambda _: FakeRpc())

    with executor.start_adapter(lock, case_dir, host_port=41235):
        pass

    command = next(call for call in docker.calls if call[1] == "run")
    assert f"type=bind,src={tree},dst=/roborsi/assets/models,readonly" in command


def test_docker_executor_rejects_missing_asset_or_digest_before_run(
    tmp_path: Path,
) -> None:
    _, asset = _manifest(tmp_path)
    lock = BenchmarkRegistry([tmp_path]).lock("demo")
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    docker = FakeDocker()
    executor = DockerExecutor(command_runner=docker, client_factory=lambda _: FakeRpc())
    asset.unlink()
    with pytest.raises(FileNotFoundError, match="asset is missing"):
        executor.start_adapter(lock, case_dir)
    assert docker.calls == []

    asset.write_bytes(b"task fixture")
    docker.present = True
    docker.digest = "sha256:" + "d" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        executor.start_adapter(lock, case_dir)
    assert not any(call[1] == "run" for call in docker.calls)


def test_docker_executor_reports_daemon_failure_without_pull(tmp_path: Path) -> None:
    _manifest(tmp_path)
    lock = BenchmarkRegistry([tmp_path]).lock("demo")
    docker = FakeDocker()
    docker.daemon_available = False
    with pytest.raises(DockerCommandError, match="Cannot connect"):
        DockerExecutor(command_runner=docker).ensure_image(lock)
    assert len(docker.calls) == 1


def test_docker_executor_removes_container_if_health_check_fails(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _manifest(tmp_path)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["resources"]["startup_timeout_s"] = 0.01
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    lock = BenchmarkRegistry([tmp_path]).lock("demo")
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    docker, rpc = FakeDocker(), FakeRpc(healthy=False)
    with pytest.raises(TimeoutError, match="did not become ready"):
        DockerExecutor(
            command_runner=docker, client_factory=lambda _: rpc
        ).start_adapter(lock, case_dir, host_port=41234)
    assert rpc.closed
    assert docker.calls[-1] == ["docker", "rm", "--force", "c" * 64]


def test_docker_executor_removes_container_if_run_finishes_after_case_deadline(
    tmp_path: Path,
) -> None:
    _manifest(tmp_path)
    lock = BenchmarkRegistry([tmp_path]).lock("demo")
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    now = [10.0]

    class LateDocker(FakeDocker):
        def __call__(self, args: list[str], timeout_s: float) -> str:
            result = super().__call__(args, timeout_s)
            if args[1] == "run":
                assert timeout_s <= 5.0
                now[0] += 6.0
            return result

    docker = LateDocker()
    deadline = CaseDeadline(5.0, clock=lambda: now[0])
    with pytest.raises(CaseTimeout, match="deadline exceeded"):
        DockerExecutor(command_runner=docker).start_adapter(
            lock, case_dir, deadline=deadline
        )
    assert docker.calls[-1] == ["docker", "rm", "--force", "c" * 64]


def test_docker_executor_removes_named_container_after_ambiguous_run_failure(
    tmp_path: Path,
) -> None:
    _manifest(tmp_path)
    lock = BenchmarkRegistry([tmp_path]).lock("demo")
    case_dir = tmp_path / "case"
    case_dir.mkdir()

    class PartialDocker(FakeDocker):
        def __call__(self, args: list[str], timeout_s: float) -> str:
            if args[1] == "run":
                self.calls.append(args)
                raise DockerCommandError(
                    "Docker run timed out after creating container"
                )
            return super().__call__(args, timeout_s)

    docker = PartialDocker()
    with pytest.raises(DockerCommandError, match="timed out"):
        DockerExecutor(command_runner=docker).start_adapter(lock, case_dir)
    run_command = next(call for call in docker.calls if call[1] == "run")
    name = run_command[run_command.index("--name") + 1]
    assert docker.calls[-1] == ["docker", "rm", "--force", name]


def test_docker_command_interrupts_in_flight_process_on_cancellation() -> None:
    cancelled = Event()
    deadline = CaseDeadline(10.0, cancel_event=cancelled)
    timer = Timer(0.1, cancelled.set)
    started = time.monotonic()
    timer.start()
    try:
        with pytest.raises(CaseCancelled, match="case cancelled"):
            _docker_command(
                [sys.executable, "-c", "import time; time.sleep(20)"],
                10.0,
                abort_check=deadline.check,
            )
    finally:
        timer.cancel()
    assert time.monotonic() - started < 3.0


@pytest.mark.parametrize("remaining", ["", "c" * 64, None])
def test_container_cleanup_only_ignores_confirmed_absence(remaining) -> None:
    def command(args, timeout_s):
        if args[1] == "rm":
            raise DockerCommandError("remove failed")
        assert args[1:3] == ["container", "ls"]
        if remaining is None:
            raise DockerCommandError("daemon unavailable")
        return remaining

    rpc = FakeRpc()
    container = AdapterContainer("c" * 64, rpc, command)
    if remaining == "":
        container.close()
    else:
        with pytest.raises(DockerCommandError):
            container.close()
    assert rpc.closed


@pytest.mark.parametrize(
    "change", ["image", "root", "asset", "port", "memory", "gpu", "command"]
)
def test_container_inspection_rejects_changed_runtime_and_removes_it(tmp_path, change):
    _manifest(tmp_path)
    lock = BenchmarkRegistry([tmp_path]).lock("demo")
    directory = tmp_path / "case"
    directory.mkdir()

    class AlteredDocker(FakeDocker):
        def __call__(self, args, timeout_s):
            reply = super().__call__(args, timeout_s)
            if args[1] == "inspect":
                value = json.loads(reply)
                if change == "image":
                    value["Image"] = "sha256:" + "f" * 64
                elif change == "root":
                    value["HostConfig"]["ReadonlyRootfs"] = False
                elif change == "asset":
                    value["Mounts"][-1]["RW"] = True
                elif change == "port":
                    value["HostConfig"]["PortBindings"]["8765/tcp"][0]["HostIp"] = (
                        "0.0.0.0"
                    )
                elif change == "memory":
                    value["HostConfig"]["Memory"] = 1024
                elif change == "gpu":
                    value["HostConfig"]["DeviceRequests"] = [{"Count": 1}]
                else:
                    value["Config"]["Cmd"] = ["other-adapter"]
                return json.dumps(value)
            return reply

    docker, rpc = AlteredDocker(), FakeRpc()
    with pytest.raises(ValueError, match="differ"):
        DockerExecutor(
            command_runner=docker, client_factory=lambda _: rpc
        ).start_adapter(lock, directory)
    assert rpc.closed and docker.calls[-1] == ["docker", "rm", "--force", "c" * 64]


def test_assets_are_checked_after_the_episode_and_cleanup_survives_mismatch(tmp_path):
    _, asset = _manifest(tmp_path)
    lock = BenchmarkRegistry([tmp_path]).lock("demo")
    directory = tmp_path / "case"
    directory.mkdir()
    docker = FakeDocker()
    container = DockerExecutor(
        command_runner=docker, client_factory=lambda _: FakeRpc()
    ).start_adapter(lock, directory)
    asset.write_bytes(b"changed during evaluation")
    with pytest.raises(ValueError, match="asset"):
        container.close()
    assert not container.runtime_record.verified_after_close
    assert docker.calls[-1] == ["docker", "rm", "--force", "c" * 64]


@pytest.mark.parametrize("case_seconds", [None, 60.0, 3.0])
def test_final_asset_verification_can_exceed_five_seconds_within_case_deadline(
    tmp_path, case_seconds
):
    _manifest(tmp_path)
    lock = BenchmarkRegistry([tmp_path]).lock("demo")
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    now = [0.0]

    class HashingRpc(FakeRpc):
        def __init__(self):
            super().__init__()
            self.runtime_calls = 0
            self.final_budget = None

        def call(self, method, *args, **kwargs):
            if method == "healthz":
                return {"status": "ok", "runtime_info_versions": [1]}
            if method == "runtime_info":
                self.runtime_calls += 1
                if self.runtime_calls == 2:
                    # Model a complete asset hash requiring six seconds of I/O.
                    self.final_budget = kwargs["timeout_s"]
                    if self.final_budget < 6.0:
                        raise TimeoutError("asset verification needs more time")
                    now[0] += 6.0
                return "null"
            return super().call(method, *args, **kwargs)

    docker, rpc = FakeDocker(), HashingRpc()
    deadline = (
        CaseDeadline(case_seconds, clock=lambda: now[0])
        if case_seconds is not None
        else None
    )
    executor = DockerExecutor(command_runner=docker, client_factory=lambda _: rpc)
    container = executor.start_adapter(lock, case_dir, deadline=deadline)
    if case_seconds == 3.0:
        with pytest.raises(TimeoutError, match="asset verification"):
            container.close()
        assert rpc.final_budget <= case_seconds
        assert not container.runtime_record.verified_after_close
    else:
        container.close()
        assert container.runtime_record.verified_after_close
    assert rpc.closed
    assert docker.calls[-1] == ["docker", "rm", "--force", "c" * 64]
