# Copyright 2026 The RoboRSI Contributors.
# Licensed under the Apache License, Version 2.0.

"""Compare the public transport with the unchanged historical evaluator."""

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from roborsi.evaluation.messages import EpisodeStop
from robots.bench2dex import contracts, icl
from robots.bench2dex.compat_agent import FrozenPolicyAgent
from robots.bench2dex.frozen_reference import load_reference

from .test_protocol_contracts import adapter, raw_observation

REFERENCE = (
    Path(__file__).resolve().parents[4] / "agents/bench2dex_compat/reference-code.zip"
)


@pytest.fixture
def reference():
    return load_reference(REFERENCE)


@pytest.fixture
def episode_type(reference):
    class ScriptedEpisode(reference.bridge.Episode):
        """Run the real mailbox validator with deterministic synthetic actions."""

        def start(self):
            self.owner_id = threading.get_ident()
            self.proc = SimpleNamespace(
                poll=lambda: None, wait=lambda timeout=None: 0, returncode=0
            )

        def model_evidence(self):
            return {"model": "gpt-6-astra", "fixture": True}

        def action(self, raw):
            count = min(3, self.budget - self.index)
            reference.core.write_json(
                self.action_transport / "request.json",
                {
                    "decision": self.decision,
                    "payload": {
                        "actions": np.tile(
                            raw["joint_action"]["vector"], (count, 1)
                        ).tolist()
                    },
                },
            )
            return super().action(raw)

    return ScriptedEpisode


def make_agent(tmp_path, package, episode_type):
    return FrozenPolicyAgent(
        tmp_path / "agent",
        reference=REFERENCE,
        package=package,
        media_dir=tmp_path / "benchmark/media",
        binary=Path("/unused/codex"),
        auth=Path("/unused/auth.json"),
        episode_factory=episode_type,
    )


def tree(root):
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file()
    }


def test_exact_model_inputs_prompt_and_chunk_boundaries(
    tmp_path, task, package, reference, episode_type
):
    boundary, native = adapter(tmp_path, task)
    policy = make_agent(tmp_path, package, episode_type)
    policy.reset(task)
    direct = episode_type(
        tmp_path / "direct",
        dict(icl.PROTOCOL),
        task.instruction,
        task.native["robot_key"],
        native.kinematic_card,
        task.max_steps,
        Path("/unused/codex"),
        Path("/unused/auth.json"),
        (reference.root / "prompts/policy_icl.md").read_text()
        + "\n"
        + (reference.root / "prompts/calibrated_deployment.md").read_text(),
    )
    reference.knowledge.attach_package(
        package,
        icl.package_entry(task.task_id)["manifest_sha256"],
        26,
        icl.PROTOCOL,
        direct.inputs,
        direct.private,
    )
    observation = boundary.start(task, "episode-1").observation
    try:
        for index in range(4):
            if index in (0, 3):
                expected_actions = direct.action(raw_observation(task, index))
            else:
                direct.observe(raw_observation(task, index))
            # The request thread exits after EVERY call, as with HTTP servers.
            with ThreadPoolExecutor(max_workers=1) as request:
                action = request.submit(policy.act, observation).result(timeout=10)
            if index in (0, 3):
                assert action.data == {"actions": expected_actions.tolist()}
            else:
                assert action.data == {"continue": True}
            assert tree(policy.episode.inputs) == tree(direct.inputs)
            assert tree(policy.episode.interface) == tree(direct.interface)
            assert policy.episode.prompt == direct.prompt
            assert policy.episode.owner_id != threading.get_ident()
            result = boundary.step(action)
            observation = result.observation
        assert policy.episode.decision == direct.decision == 2
        policy.observe_end(result)
        direct.finish()
        assert tree(policy.episode.inputs) == tree(direct.inputs)
        assert policy.episode.finished
        assert boundary.finish().score == 0
    finally:
        policy.close()
        direct.finish()
        boundary.close()


@pytest.mark.parametrize("corrupt", ["clock", "private", "robot", "png", "path"])
def test_rejects_changed_public_input_before_inference(
    tmp_path, task, package, episode_type, corrupt
):
    boundary, _ = adapter(tmp_path, task)
    policy = make_agent(tmp_path, package, episode_type)
    policy.reset(task)
    obs = boundary.start(task, "episode-1").observation
    if corrupt == "clock":
        obs.data["sample"]["physics_step"] = 99
    elif corrupt == "private":
        obs.data["success"] = True
    elif corrupt == "robot":
        obs.data["robot"]["hidden_state"] = 1
    elif corrupt == "path":
        obs.data["sample"]["cameras"][0]["path"] = "input/unrelated.png"
    else:
        obs.data["sample"]["cameras"][0]["rgb_sha256"] = "0" * 64
    try:
        with pytest.raises(ValueError):
            policy.act(obs)
        assert policy.episode is None
    finally:
        policy.close()
        boundary.close()


def test_policy_failure_is_scored_without_extra_native_steps(
    tmp_path, task, package, reference, episode_type
):
    class BadEpisode(episode_type):
        def validate_submission(self, *args):
            raise ValueError("synthetic malformed model action")

    boundary, native = adapter(tmp_path, task)

    def stop_policy():
        native.result = {"success": False, "terminated_reason": "error", "steps": 0}

    native.stop_policy = stop_policy
    policy = make_agent(tmp_path, package, BadEpisode)
    policy.reset(task)
    obs = boundary.start(task, "episode-1").observation
    try:
        stop = policy.act(obs)
        assert isinstance(stop, EpisodeStop)
        result = boundary.stop(stop)
        assert result.observation.data == {"episode_done": True}
        policy.observe_end(result)
        score = boundary.finish()
        assert not score.success and score.score == 0
        assert score.details["policy_failure"] and native.steps == 0
    finally:
        policy.close()
        boundary.close()


def test_close_wakes_infinite_action_wait_and_cleans_auth(
    tmp_path, task, package, reference, episode_type
):
    ready = threading.Event()

    class WaitingEpisode(episode_type):
        def action(self, raw):
            return reference.bridge.Episode.action(self, raw)

        def start(self):
            super().start()
            (self.home / "auth.json").write_text('{"synthetic": true}')
            ready.set()

    boundary, _ = adapter(tmp_path, task)
    policy = make_agent(tmp_path, package, WaitingEpisode)
    policy.reset(task)
    obs = boundary.start(task, "episode-1").observation
    with ThreadPoolExecutor(max_workers=1) as request:
        future = request.submit(policy.act, obs)
        assert ready.wait(10)
        policy.close()
        with pytest.raises(RuntimeError, match="Evaluator stopped"):
            future.result(timeout=5)
    assert policy.episode.finished and not (policy.episode.home / "auth.json").exists()
    boundary.close()


def test_reference_rejects_modified_archive(tmp_path):
    path = tmp_path / "reference.zip"
    path.write_bytes(REFERENCE.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="archive hash"):
        load_reference(path)


def test_cli_command_is_original_and_owner_survives_request_exit(
    tmp_path, task, package, reference, episode_type
):
    import shutil
    import subprocess

    from codex_cli_bin import bundled_codex_path

    if not shutil.which("bwrap"):
        pytest.skip("bubblewrap is not installed on this CPU test host")
    boundary, _ = adapter(tmp_path, task)
    policy = make_agent(tmp_path, package, episode_type)
    policy.binary = Path(bundled_codex_path())
    policy.reset(task)
    try:
        with ThreadPoolExecutor(max_workers=1) as request:
            request.submit(
                policy.act, boundary.start(task, "episode-1").observation
            ).result(10)
        episode = policy.episode
        expected = reference.bridge.isolation.command(
            episode.workspace,
            episode.home,
            episode.interface,
            policy.binary,
            icl.PROTOCOL,
        )
        assert episode.policy_command() == expected
        assert 'model_reasoning_effort="max"' in expected
        assert "--ignore-user-config" in expected and "--ignore-rules" in expected
        command = [
            "bwrap",
            "--die-with-parent",
            "--ro-bind",
            "/",
            "/",
            "--",
            "/bin/cat",
        ]
        try:
            process = policy.owner.submit(
                subprocess.Popen,
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).result(10)
            with ThreadPoolExecutor(max_workers=1) as request:
                request.submit(lambda: None).result()
            output, error = process.communicate(b"owner-alive\n", timeout=5)
            if process.returncode and b"Operation not permitted" in error:
                pytest.skip("host kernel disallows bubblewrap user namespaces")
            assert process.returncode == 0 and output == b"owner-alive\n", error
        finally:
            if "process" in locals() and process.poll() is None:
                process.kill()
                process.wait()
    finally:
        policy.close()
        boundary.close()


def test_frozen_package_survives_archive_installation(
    tmp_path, task, package, episode_type
):
    import zipfile

    archive = tmp_path / "icl.zip"
    files = sorted(p for p in package.rglob("*") if p.is_file())
    with zipfile.ZipFile(archive, "w") as output:
        for path in files:
            output.write(path, path.relative_to(package))
    icl.package_entry(task.task_id).update(
        archive_bytes=archive.stat().st_size,
        package_bytes=sum(p.stat().st_size for p in files),
        sha256=contracts.file_sha256(archive),
    )
    boundary, _ = adapter(tmp_path, task)
    policy = make_agent(tmp_path, archive, episode_type)
    try:
        policy.reset(task)
        policy.act(boundary.start(task, "episode-1").observation)
        assert tree(policy.episode.inputs / "task_knowledge") == tree(
            package / "payload"
        )
    finally:
        policy.close()
        boundary.close()


def test_native_policy_stop_uses_original_error_envelope_without_action(
    tmp_path, task, monkeypatch
):
    from robots.bench2dex.native import NativeRuntime

    runtime = NativeRuntime(
        tmp_path,
        upstream=tmp_path,
        assets=tmp_path,
        isaaclab=tmp_path,
        python=Path("/unused"),
    )
    runtime.task = task
    observed = []

    def finish():
        observed.append(runtime._commands.get_nowait())
        runtime.result = {"success": False, "terminated_reason": "error"}

    monkeypatch.setattr(runtime, "_next", finish)
    runtime.stop_policy()
    assert observed == [{"error": "Policy infrastructure failure"}]
    assert runtime.steps == 0 and runtime.policy_failed
    runtime.close()


def test_runtime_source_identity_does_not_require_unmerged_registration(monkeypatch):
    from roborsi.benchmarks import runtime_source
    from robots.bench2dex.benchmark_service import adapter_digest

    def unregistered(*args):
        raise ValueError("unsupported Benchmark source bundle")

    monkeypatch.setattr(runtime_source, "benchmark_source_digest", unregistered)
    assert len(adapter_digest()) == 64


def test_success_requires_native_stable_stop_not_transient_flag(tmp_path, task):
    boundary, native = adapter(tmp_path, task)
    boundary.start(task, "episode-1")
    boundary.ended = True
    native.result = {"success": True, "terminated_reason": "max_steps", "steps": 12}
    try:
        result = boundary.finish()
        assert result.score == 0 and not result.success
        assert result.details["official_success"] is True
    finally:
        boundary.close()
