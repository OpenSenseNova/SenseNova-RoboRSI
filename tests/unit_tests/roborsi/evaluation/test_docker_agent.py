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

"""Agent containers expose only the current workspace and read-only images."""

import json
import stat
from pathlib import Path

import pytest

from roborsi.evaluation.docker_agent import DockerAgentLauncher
from roborsi.evaluation.docker_common import DockerCommandError, remove_container
from roborsi.evaluation.messages import (
    ContainerAgentRuntimeRecord,
    RuntimeVerification,
)
from roborsi.evaluation.store import ComparisonPolicy, ResultStore
from tests.unit_tests.roborsi.evaluation.test_docker_runtime import (
    FakeDocker,
    FakeRpc,
)
from tests.unit_tests.roborsi.evaluation.test_store import (
    _provenance,
    _result,
    _runtime,
    _task,
)

IMAGE = "example.org/demo@sha256:" + "b" * 64


class AgentDocker(FakeDocker):
    def __init__(self):
        super().__init__()
        self.networks = set()
        self.container_present = False
        self.env_path = None
        self.changed = None

    def __call__(self, args, timeout_s):
        if args[1] == "network":
            self.calls.append(args)
            if args[2] == "create":
                self.networks.add(args[-1])
                return "d" * 64
            if args[2] == "rm":
                self.networks.discard(args[-1])
                return args[-1]
            if args[2] == "ls":
                return "d" * 64 if self.networks else ""
            if args[2] == "inspect":
                return json.dumps([{"Driver": "bridge", "Containers": {"c" * 64: {}}}])
        if args[1] == "run":
            self.env_path = Path(args[args.index("--env-file") + 1])
            assert stat.S_IMODE(self.env_path.stat().st_mode) == 0o600
            value = super().__call__(args, timeout_s)
            self.container_present = True
            self.inspection["Config"].update(
                Entrypoint=["python"], WorkingDir="/roborsi/work"
            )
            self.inspection["HostConfig"].update(
                Privileged=False,
                CapAdd=None,
                CapDrop=["ALL"],
                SecurityOpt=["no-new-privileges"],
                PidsLimit=int(args[args.index("--pids-limit") + 1]),
                NetworkMode=args[args.index("--network") + 1],
                IpcMode="private",
                PidMode="",
                Tmpfs={"/tmp": "rw,nosuid,nodev,size=512m"},
            )
            self.inspection["NetworkSettings"] = {
                "Networks": {args[args.index("--network") + 1]: {}}
            }
            return value
        if args[1] == "rm":
            self.container_present = False
        if args[1] == "inspect" and self.changed:
            self.changed(self.inspection)
        return super().__call__(args, timeout_s)


def _directories(tmp_path):
    workspace, media = tmp_path / "agent", tmp_path / "media"
    workspace.mkdir()
    media.mkdir()
    return workspace, media


def test_failed_startup_cleanup_checks_container_name():
    def docker(args, timeout_s):
        if args[1] == "rm":
            raise DockerCommandError("removal failed")
        return "c" * 64 if "name=^roborsi-agent-owned$" in args else ""

    with pytest.raises(DockerCommandError, match="removal failed"):
        remove_container(docker, "roborsi-agent-owned")


def test_only_case_workspace_images_and_selected_credentials_are_exposed(
    tmp_path, monkeypatch
):
    workspace, media = _directories(tmp_path)
    monkeypatch.setenv("TEST_MODEL_TOKEN", "private-test-token")
    monkeypatch.setenv("OTHER_SECRET", "never-pass-this")
    docker, rpc = AgentDocker(), FakeRpc()
    launcher = DockerAgentLauncher(
        IMAGE,
        "agent:make_agent",
        command_runner=docker,
        client_factory=lambda _: rpc,
        credential_env=["TEST_MODEL_TOKEN"],
    )
    launcher.verify_version(launcher.agent_version)
    with pytest.raises(ValueError, match="version differs"):
        launcher.verify_version("wrong")
    with launcher.start(workspace, media_dir=media) as container:
        assert docker.container_present and len(docker.networks) == 1
        assert not docker.env_path.exists()
        assert container.runtime_record is None
    record = container.runtime_record
    assert (
        record.verified_after_close and record.agent_version == launcher.agent_version
    )
    assert not docker.container_present and not docker.networks and rpc.closed
    mounts = docker.inspection["Mounts"]
    assert {(m["Source"], m["Destination"], m["RW"]) for m in mounts} == {
        (str(workspace), "/roborsi/work", True),
        (str(media), "/roborsi/media", False),
    }
    environment = docker.inspection["Config"]["Env"]
    assert "TEST_MODEL_TOKEN=private-test-token" in environment
    assert not any(
        "OTHER_SECRET" in value or "ROBORSI_BENCHMARK_TOKEN" in value
        for value in environment
    )
    assert "private-test-token" not in json.dumps(docker.calls)
    assert "private-test-token" not in json.dumps(record.as_dict())
    assert record.configuration["credential_env"] == ["TEST_MODEL_TOKEN"]


@pytest.mark.parametrize(
    "change", ["write_root", "extra_mount", "network", "capability", "resources"]
)
def test_bad_runtime_configuration_is_rejected_and_cleaned_up(tmp_path, change):
    workspace, media = _directories(tmp_path)
    docker, rpc = AgentDocker(), FakeRpc()

    def alter(value):
        if change == "write_root":
            value["HostConfig"]["ReadonlyRootfs"] = False
        elif change == "extra_mount":
            value["Mounts"].append(
                {"Source": "/", "Destination": "/host", "Type": "bind", "RW": True}
            )
        elif change == "network":
            value["NetworkSettings"]["Networks"]["shared"] = {}
        elif change == "capability":
            value["HostConfig"]["CapAdd"] = ["SYS_ADMIN"]
        else:
            value["HostConfig"]["PidsLimit"] = -1

    docker.changed = alter
    with pytest.raises(ValueError):
        DockerAgentLauncher(
            IMAGE,
            "agent:make_agent",
            command_runner=docker,
            client_factory=lambda _: rpc,
        ).start(workspace, media_dir=media)
    assert rpc.closed and not docker.container_present and not docker.networks


def test_unready_agent_and_failed_network_creation_leave_no_resources(tmp_path):
    workspace, media = _directories(tmp_path)
    docker, rpc = AgentDocker(), FakeRpc(healthy=False)
    with pytest.raises(TimeoutError):
        DockerAgentLauncher(
            IMAGE,
            "agent:make_agent",
            startup_timeout_s=0.01,
            command_runner=docker,
            client_factory=lambda _: rpc,
        ).start(workspace, media_dir=media)
    assert rpc.closed and not docker.container_present and not docker.networks

    class NetworkFailure(AgentDocker):
        def __call__(self, args, timeout_s):
            if args[1:3] == ["network", "create"]:
                super().__call__(args, timeout_s)
                raise DockerCommandError("interrupted after network creation")
            return super().__call__(args, timeout_s)

    docker = NetworkFailure()
    with pytest.raises(DockerCommandError, match="interrupted"):
        DockerAgentLauncher(IMAGE, "agent:make_agent", command_runner=docker).start(
            workspace, media_dir=media
        )
    assert not docker.networks and not docker.container_present


def test_writable_media_overlap_and_secret_newlines_are_rejected(tmp_path, monkeypatch):
    workspace, media = _directories(tmp_path)
    docker = AgentDocker()
    launcher = DockerAgentLauncher(
        IMAGE, "agent:make_agent", command_runner=docker, credential_env=["TOKEN"]
    )
    with pytest.raises(ValueError, match="separate"):
        launcher.start(workspace, media_dir=workspace)
    assert not docker.calls
    monkeypatch.setenv("TOKEN", "one\nOTHER=value")
    with pytest.raises(ValueError, match="single-line"):
        launcher.start(workspace, media_dir=media)
    assert not docker.networks and not docker.container_present


def test_client_close_failure_still_removes_container_and_network(tmp_path):
    workspace, media = _directories(tmp_path)
    docker = AgentDocker()

    class BrokenClient(FakeRpc):
        def close(self):
            self.closed = True
            raise RuntimeError("client cleanup failed")

    rpc = BrokenClient()
    service = DockerAgentLauncher(
        IMAGE, "agent:make_agent", command_runner=docker, client_factory=lambda _: rpc
    ).start(workspace, media_dir=media)
    with pytest.raises(RuntimeError, match="client cleanup"):
        service.close()
    assert not docker.networks and not docker.container_present and rpc.closed
    service.close()


def test_cli_derives_image_version_and_passes_resource_limits(
    tmp_path, monkeypatch, capsys
):
    from roborsi.evaluation import cli
    from tests.unit_tests.roborsi.evaluation.test_batch import (
        _PublishedCases,
        _registry,
    )

    _registry(tmp_path / "registry")
    launchers = []

    def runner(registry, store, launcher, executor):
        launchers.append(launcher)
        return _PublishedCases(registry, store)

    monkeypatch.setattr(cli, "CaseOrchestrator", runner)
    assert (
        cli.main(
            [
                "eval",
                "--registry-root",
                str(tmp_path / "registry"),
                "--results-root",
                str(tmp_path / "results"),
                "--run-id",
                "container",
                "--benchmark",
                "demo",
                "--agent-factory",
                "agent:make_agent",
                "--agent-image",
                IMAGE,
                "--agent-cpus",
                "2",
                "--agent-memory-mb",
                "4096",
                "--agent-credential-env",
                "MODEL_TOKEN",
            ]
        )
        == 0
    )
    assert isinstance(launchers[0], DockerAgentLauncher)
    assert launchers[0].cpus == 2 and launchers[0].memory_mb == 4096
    result = next(iter(ResultStore(tmp_path / "results").list_cases("container")))
    assert (
        ResultStore(tmp_path / "results")
        .load("container", result)
        .provenance.agent_version
        == launchers[0].agent_version
    )
    assert json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("change", ["same", "limits", "host"])
def test_comparison_checks_container_agent_settings(tmp_path, change):
    store = ResultStore(tmp_path)
    provenance = _provenance("1" * 64)
    document = _runtime(provenance).as_dict()
    document["agent"] = ContainerAgentRuntimeRecord(
        agent_version="1" * 64,
        docker_image_digest="sha256:" + "a" * 64,
        factory="agent:make",
        configuration={"cpus": 1},
        verified_after_close=True,
    ).as_dict()
    first = RuntimeVerification.from_dict(document)
    if change == "limits":
        document["agent"]["configuration"]["cpus"] = 2
    elif change == "host":
        document["agent"] = _runtime(provenance).agent.as_dict()
    second = RuntimeVerification.from_dict(document)
    for run, runtime in (("first", first), ("second", second)):
        store.claim(run, "case", _task(), provenance)
        store.publish(run, "case", _result(_task(), provenance, runtime=runtime))
    assert store.compare(
        "first", "case", "second", "case", policy=ComparisonPolicy.strict_runtime()
    ).comparable is (change == "same")


def test_preset_overrides_are_versioned_forwarded_and_inspected(tmp_path):
    workspace, media = tmp_path / "work", tmp_path / "media"
    workspace.mkdir()
    media.mkdir()
    docker = AgentDocker()
    factory = "roborsi_agent_bundle_demo:make_agent"
    original = DockerAgentLauncher(IMAGE, factory, agent_config="bundle:demo")
    changed = DockerAgentLauncher(
        IMAGE,
        factory,
        agent_config="bundle:demo",
        agent_overrides=["agent.max_turns=7"],
        command_runner=docker,
        client_factory=lambda _: FakeRpc(),
    )
    assert changed.agent_version != original.agent_version
    with pytest.raises(ValueError, match="version differs"):
        changed.verify_version(original.agent_version)
    with changed.start(workspace, media_dir=media) as process:
        command = docker.inspection["Config"]["Cmd"]
        assert command[command.index("--agent-config") + 1] == "bundle:demo"
        assert command[command.index("--agent-set") + 1] == "/agent/max_turns=7"
    assert process.runtime_record.configuration["agent_overrides"] == [
        "/agent/max_turns=7"
    ]
    assert process.runtime_record.verified_after_close
    with changed.start(workspace, media_dir=media) as process:
        docker.inspection["Config"]["Cmd"][-1] = "/agent/max_turns=8"
        with pytest.raises(ValueError, match="differs"):
            process.close()
    assert not docker.container_present and not docker.networks
