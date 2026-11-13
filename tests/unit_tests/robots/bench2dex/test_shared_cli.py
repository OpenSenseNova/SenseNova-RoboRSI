# Copyright 2026 The RoboRSI Contributors.
# Licensed under the Apache License, Version 2.0.

"""Exercise configured text and the real shared subprocess/file transport."""

import hashlib
import json
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from roborsi.agents.config import AgentConfig
from roborsi.embodied_agent import EmbodiedAgent
from robots.bench2dex.agents.baseline.cli_episode import (
    episode_type,
    historical_cli_config,
)
from robots.bench2dex.agents.baseline.factory import (
    FrozenPolicyAgent,
    configured_policy,
    historical_prompt,
)
from robots.bench2dex.frozen_reference import load_reference

from .test_protocol_contracts import adapter

ROOT = Path(__file__).resolve().parents[4]
REFERENCE = ROOT / "agents/bench2dex_compat/reference-code.zip"


def test_baseline_and_alias_have_exact_historical_prompt():
    reference = load_reference(REFERENCE)
    for entry in (
        "robots/bench2dex/agent.yaml",
        "robots/bench2dex/agents/baseline/agent.yaml",
        "configs/agents/bench2dex/compat.yaml",
    ):
        config = AgentConfig.from_file(ROOT / entry)
        prompt, cli, identity = configured_policy(config, reference)
        assert prompt == historical_prompt(reference)
        assert not identity["skills"]
        assert cli.tools == "external"
        assert config.document["agent"]["planner"] == "codex_cli"


def configured_copy(tmp_path, *, customized=True):
    config = AgentConfig.from_file(ROOT / "robots/bench2dex/agent.yaml")
    # A composed preset is portable after all declared files are copied.
    for path in config.input_files:
        dest = tmp_path / path.relative_to(config.source_root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    selected = tmp_path / "robots/bench2dex/agent.yaml"
    if customized:
        (selected.parent / "custom.md").write_text("CUSTOM: observe before grasping.\n")
        (selected.parent / "skill.md").write_text("SKILL: inspect the wrist camera.\n")
        selected.write_text(
            selected.read_text() + "prompt_file: ./custom.md\nskills: [./skill.md]\n"
        )
    return AgentConfig.from_file(selected)


def test_prompt_skill_changes_are_loaded_and_fingerprinted(tmp_path):
    config = configured_copy(tmp_path)
    prompt, _, identity = configured_policy(config, load_reference(REFERENCE))
    assert (
        prompt
        == "CUSTOM: observe before grasping.\n\n\nSKILL: inspect the wrist camera.\n"
    )
    assert len(identity["skills"]) == 1
    assert prompt != historical_prompt(load_reference(REFERENCE))
    skill = config.path.parent / "skill.md"
    skill.write_text("DIFFERENT SKILL\n")
    changed = AgentConfig.from_file(config.path)
    assert config.fingerprint != changed.fingerprint
    assert "DIFFERENT SKILL" in configured_policy(changed, load_reference(REFERENCE))[0]
    with pytest.raises(ValueError):
        config.verify()


def test_declared_export_imports_without_source_checkout(tmp_path):
    config = configured_copy(tmp_path / "relocated", customized=False)
    root = config.source_root
    program = (
        "import pathlib,sys; "
        f"sys.path.insert(0, {str(root)!r}); "
        "from robots.bench2dex.agents.baseline import factory; "
        f"assert pathlib.Path(factory.__file__).is_relative_to({str(root)!r})"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", program],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "override",
    [
        'agent.model="different-model"',
        'agent.reasoning_effort="low"',
        'agent.codex_cli.config_overrides.web_search="live"',
        "agent.planner_timeout_s=60",
    ],
)
def test_unsupported_experiment_changes_fail_explicitly(override):
    config = AgentConfig.from_file(
        ROOT / "robots/bench2dex/agent.yaml", overrides=[override]
    )
    with pytest.raises(ValueError, match="pinned Astra/max"):
        configured_policy(config, load_reference(REFERENCE))


FAKE_CLI = """#!/usr/bin/python3
import json, pathlib, subprocess, sys
if "--version" in sys.argv:
    print("codex-cli 0.154.0")
    raise SystemExit
root = pathlib.Path("/workspace")
(root / "received-prompt.txt").write_text(sys.stdin.read())
(root / "received-argv.json").write_text(json.dumps(sys.argv[1:]))
print(json.dumps({"type":"thread.started", "thread_id":"offline"}), flush=True)
while not (root / "input/episode_done.json").exists():
    observation = json.loads((root / "input/observation.json").read_text())
    count = min(3, observation["remaining_policy_steps"])
    actions = root / "fixture-action.json"
    actions.write_text(json.dumps({"actions": [observation["qpos"]] * count}))
    subprocess.run([sys.executable, "/interface/step.py", str(actions)],
                   check=True, stdout=subprocess.DEVNULL)
print(json.dumps({"type":"turn.completed", "usage":{"input_tokens":1,"output_tokens":1}}), flush=True)
"""


@pytest.mark.parametrize("customized", [False, True])
@pytest.mark.parametrize("terminal_error", [False, True])
def test_real_shared_planner_receives_text_and_preserves_chunk_feedback(
    tmp_path,
    task,
    package,
    customized,
    terminal_error,
):
    if not shutil.which("bwrap"):
        pytest.skip("Linux bubblewrap required for the original mount boundary")
    reference = load_reference(REFERENCE)
    config = configured_copy(tmp_path / "relocated", customized=customized)
    prompt, cli, identity = configured_policy(config, reference)
    binary = tmp_path / "codex"
    script = FAKE_CLI
    if terminal_error:
        script = script.replace(
            'print(json.dumps({"type":"turn.completed"',
            'print(json.dumps({"type":"turn.failed", "error":{"message":"late planner failure"}}), flush=True)\nraise SystemExit(1)\nprint(json.dumps({"type":"turn.completed"',
        )
    binary.write_text(script)
    binary.chmod(0o755)
    binary.with_name("codex-code-mode-host").write_text("offline fixture")
    auth = tmp_path / "auth.json"
    auth.write_text("{}")
    cls = episode_type(reference, replace(cli, expected_sha256=None))

    class FixtureEpisode(cls):
        def model_evidence(self):
            return {"fixture": True}

    boundary, _ = adapter(tmp_path, task)
    policy = FrozenPolicyAgent(
        tmp_path / "agent",
        reference=REFERENCE,
        package=package,
        media_dir=tmp_path / "benchmark/media",
        binary=binary,
        auth=auth,
        episode_factory=FixtureEpisode,
        prompt=prompt,
        prompt_identity=identity,
    )
    policy.reset(task)
    obs = boundary.start(task, "shared-cli-test").observation
    try:
        actions = []
        for _ in range(4):
            with ThreadPoolExecutor(max_workers=1) as request:
                action = request.submit(policy.act, obs).result(timeout=20)
            actions.append(action.data)
            result = boundary.step(action)
            obs = result.observation
        if terminal_error:
            with pytest.raises(RuntimeError, match="late planner failure"):
                policy.observe_end(result)
        else:
            policy.observe_end(result)
        episode = policy.episode
        assert isinstance(episode.embodied_agent, EmbodiedAgent)
        assert episode.session.done
        assert policy.closed and episode.finished
        call = episode.private / "codex-cli/call-0001"
        assert (call / "input.txt").read_text() == prompt
        assert (episode.workspace / "received-prompt.txt").read_text() == prompt
        assert len(actions[0]["actions"]) == 3
        assert actions[1:3] == [{"continue": True}] * 2
        assert len(actions[3]["actions"]) == 1
        assert len(list((episode.private / "observations").iterdir())) == 4
        runtime = json.loads((call / "runtime.json").read_text())
        assert runtime["tools"] == "external"
        assert runtime["cleanup"]["status"] == "complete"
        planner_result = json.loads((call / "result.json").read_text())
        if terminal_error:
            assert "late planner failure" in planner_result["error"]
            assert planner_result["error_kind"] == "infrastructure"
        else:
            assert planner_result["error"] is None
        provenance = json.loads((policy.directory / "provenance.json").read_text())
        assert (
            provenance["effective_prompt_sha256"]
            == hashlib.sha256(prompt.encode()).hexdigest()
        )
        assert provenance["historical_prompt_equal"] is not customized
        assert not (episode.home / "auth.json").exists()
        # Argument ordering changes, but every historical option/value remains.
        original = episode.policy_command()
        original = original[original.index("--") + 2 :]
        actual = json.loads((episode.workspace / "received-argv.json").read_text())

        def options(argv):
            singles, pairs = [], []
            args = iter(argv)
            for arg in args:
                if arg in {"--model", "--sandbox", "--color", "-c", "--disable"}:
                    pairs.append((arg, next(args)))
                else:
                    singles.append(arg)
            return sorted(singles), sorted(pairs)

        assert options(actual) == options(original)
    finally:
        policy.close()
        boundary.close()


@pytest.mark.parametrize("mode", ["cancel", "version_failure"])
def test_shared_planner_cancellation_and_startup_failure_remove_auth(tmp_path, mode):
    if not shutil.which("bwrap"):
        pytest.skip("Linux bubblewrap required")
    reference = load_reference(REFERENCE)
    binary = tmp_path / "codex"
    version = "wrong-version" if mode == "version_failure" else "codex-cli 0.154.0"
    binary.write_text(
        "#!/usr/bin/python3\nimport sys,time,pathlib\n"
        f"if '--version' in sys.argv:\n    print({version!r})\n    sys.exit(0)\n"
        "sys.stdin.read()\npathlib.Path('/workspace/started').touch()\ntime.sleep(120)\n"
    )
    binary.chmod(0o755)
    binary.with_name("codex-code-mode-host").write_text("offline fixture")
    auth = tmp_path / "auth.json"
    auth.write_text("{}")
    cls = episode_type(reference, historical_cli_config(reference.bridge.isolation))
    episode = cls(
        tmp_path / "episode",
        {"model": "gpt-6-astra", "reasoning_effort": "max"},
        "fixture",
        "fixture",
        {},
        4,
        binary,
        auth,
        "offline prompt",
    )
    episode.start()
    call = episode.private / "codex-cli/call-0001"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if (episode.workspace / "started").exists() or episode.session.done:
            break
        threading.Event().wait(0.01)
    try:
        if mode == "version_failure":
            with pytest.raises(RuntimeError, match="embodied episode failed"):
                episode.finish()
        else:
            assert (episode.workspace / "started").exists()
            episode.finish()
        assert not (episode.home / "auth.json").exists()
        assert episode.session.done and episode.finished
        if mode == "version_failure":
            # EmbodiedAgent rejects invalid startup before planner.solve.
            assert not (episode.workspace / "started").exists()
            with pytest.raises(RuntimeError, match="embodied episode failed"):
                episode.session.wait(timeout=0)
        else:
            result = json.loads((call / "result.json").read_text())
            assert result["error_kind"] == "cancelled"
            assert (
                json.loads((call / "runtime.json").read_text())["cleanup"]["status"]
                == "complete"
            )
    finally:
        if not episode.finished:
            episode.finish()


@pytest.mark.parametrize("fault", ["terminal_write", "auth_unlink", "session_close"])
def test_close_retries_io_and_cleanup_failures_without_losing_session(
    tmp_path, task, package, monkeypatch, fault
):
    if not shutil.which("bwrap"):
        pytest.skip("Linux bubblewrap required")
    reference = load_reference(REFERENCE)
    binary = tmp_path / "codex"
    binary.write_text(FAKE_CLI)
    binary.chmod(0o755)
    binary.with_name("codex-code-mode-host").write_text("offline fixture")
    auth = tmp_path / "auth.json"
    auth.write_text("{}")
    cls = episode_type(reference, historical_cli_config(reference.bridge.isolation))

    class FixtureEpisode(cls):
        def model_evidence(self):
            return {"fixture": True}

    boundary, _ = adapter(tmp_path, task)
    policy = FrozenPolicyAgent(
        tmp_path / "agent",
        reference=REFERENCE,
        package=package,
        media_dir=tmp_path / "benchmark/media",
        binary=binary,
        auth=auth,
        episode_factory=FixtureEpisode,
    )
    try:
        policy.reset(task)
        obs = boundary.start(task, "cleanup-test").observation
        for _ in range(4):
            result = boundary.step(policy.act(obs))
            obs = result.observation
        episode = policy.episode
        session = episode.session
        with monkeypatch.context() as patch:
            if fault == "terminal_write":
                original = reference.core.write_json

                def fail_write(path, value):
                    if path == episode.inputs / "episode_done.json":
                        raise OSError("injected marker I/O failure")
                    return original(path, value)

                patch.setattr(reference.core, "write_json", fail_write)
            elif fault == "auth_unlink":
                original = Path.unlink

                def fail_unlink(path, *args, **kwargs):
                    if path == episode.home / "auth.json":
                        raise OSError("injected auth I/O failure")
                    return original(path, *args, **kwargs)

                patch.setattr(Path, "unlink", fail_unlink)
            else:

                def fail_close(timeout):
                    raise TimeoutError("injected cleanup timeout")

                patch.setattr(session, "close", fail_close)
            with pytest.raises(OSError, match="injected"):
                policy.observe_end(result)
            assert policy.closing and not policy.closed and not episode.finished
            assert episode.session is session
            # Both marker and auth I/O errors still stop the model worker.
            assert session.done
            if fault != "auth_unlink":
                assert not (episode.home / "auth.json").exists()
            if fault == "terminal_write":
                assert episode.cancel_requested
                record = episode.private / "codex-cli/call-0001/runtime.json"
                assert json.loads(record.read_text())["cleanup"]["status"] == "complete"
            with pytest.raises(ValueError, match="ended"):
                policy.act(obs)
        policy.close()
        assert policy.closed and episode.finished and session.done
        assert not (episode.home / "auth.json").exists()
        policy.close()  # Successful cleanup is idempotent.
    finally:
        policy.close()
        boundary.close()


def test_unrequested_cancellation_is_an_infrastructure_failure(tmp_path):
    from types import SimpleNamespace

    reference = load_reference(REFERENCE)
    cls = episode_type(reference, historical_cli_config(reference.bridge.isolation))
    episode = cls(
        tmp_path / "episode",
        {},
        "fixture",
        "fixture",
        {},
        4,
        Path("/unused"),
        Path("/unused"),
        "fixture prompt",
    )
    episode.cancel_requested = False
    result = SimpleNamespace(
        error="unexpected cancellation", error_kind="cancelled", stop_reason="closed"
    )
    episode.session = SimpleNamespace(
        done=True,
        close=lambda timeout: None,
        wait=lambda timeout: result,
    )
    with pytest.raises(RuntimeError, match="unexpected cancellation"):
        episode.finish()
    assert episode.finished


def test_incomplete_worker_cleanup_keeps_agent_retryable(tmp_path, task, package):
    from types import SimpleNamespace

    reference = load_reference(REFERENCE)
    cls = episode_type(reference, historical_cli_config(reference.bridge.isolation))
    policy = FrozenPolicyAgent(
        tmp_path / "agent",
        reference=REFERENCE,
        package=package,
        media_dir=tmp_path / "media",
        binary=Path("/unused"),
        auth=Path("/unused"),
        episode_factory=cls,
    )
    episode = cls(
        tmp_path / "episode",
        {},
        "fixture",
        "fixture",
        {},
        4,
        Path("/unused"),
        Path("/unused"),
        "fixture prompt",
    )
    policy.episode = episode
    result = SimpleNamespace(
        error="owner cancelled", error_kind="cancelled", stop_reason="closed"
    )
    cancelled = threading.Event()

    def wait(timeout):
        if not session.done:
            raise TimeoutError("worker still running")
        return result

    def close(timeout):
        if not session.done:
            raise TimeoutError("cleanup still running")

    session = SimpleNamespace(
        done=False, closed=False, wait=wait, close=close, cancel=cancelled.set
    )
    episode.session = session
    episode.cancel_requested = False
    try:
        with pytest.raises(TimeoutError, match="cleanup still running"):
            policy.close()
        assert cancelled.is_set()
        assert not policy.closed and not episode.finished
        assert episode.session is session and not session.done
        session.done = True
        policy.close()
        assert policy.closed and episode.finished
    finally:
        session.done = True
        policy.close()


@pytest.mark.parametrize("reason", ["closed", "cleanup_failed"])
def test_preexisting_cancellation_or_cleanup_error_cannot_become_owner_success(
    tmp_path, reason
):
    from types import SimpleNamespace

    reference = load_reference(REFERENCE)
    cls = episode_type(reference, historical_cli_config(reference.bridge.isolation))
    episode = cls(
        tmp_path / "episode",
        {},
        "fixture",
        "fixture",
        {},
        4,
        Path("/unused"),
        Path("/unused"),
        "fixture prompt",
    )
    episode.cancel_requested = False
    result = SimpleNamespace(
        error="unexpected shutdown failure",
        stop_reason=reason,
        error_kind="cancelled" if reason == "closed" else "infrastructure",
    )

    def wait(timeout):
        if not session.done:
            raise TimeoutError("cancellation is already in progress")
        return result

    def close(timeout):
        session.done = True

    session = SimpleNamespace(done=False, closed=True, wait=wait, close=close)
    episode.session = session
    with pytest.raises(RuntimeError, match="unexpected shutdown failure"):
        episode.finish()
    assert not episode.cancel_requested
    assert episode.finished is (reason != "cleanup_failed")
    if reason == "cleanup_failed":
        with pytest.raises(RuntimeError, match="unexpected shutdown failure"):
            episode.finish()
        assert episode.session is session and not episode.finished


def test_runner_rejects_native_success_after_final_cli_failure(tmp_path, task, package):
    from types import SimpleNamespace

    from roborsi.evaluation.media import CaseMediaChannel
    from roborsi.evaluation.messages import Provenance
    from roborsi.evaluation.runner import CaseRunner
    from roborsi.evaluation.store import ResultStore

    if not shutil.which("bwrap"):
        pytest.skip("Linux bubblewrap required")
    reference = load_reference(REFERENCE)
    binary = tmp_path / "codex"
    binary.write_text(
        FAKE_CLI[: FAKE_CLI.index('print(json.dumps({"type":"turn.completed"')]
        + 'print(json.dumps({"type":"turn.failed", "error":{"message":"late planner failure"}}), flush=True)\nraise SystemExit(1)\n'
    )
    binary.chmod(0o755)
    binary.with_name("codex-code-mode-host").write_text("offline fixture")
    auth = tmp_path / "auth.json"
    auth.write_text("{}")
    cls = episode_type(reference, historical_cli_config(reference.bridge.isolation))

    class FixtureEpisode(cls):
        def model_evidence(self):
            return {"fixture": True}

    boundary, native = adapter(tmp_path, task)
    original_advance = native.advance

    def advance(actions):
        observation = original_advance(actions)
        if observation is None:
            native.result.update(success=True, terminated_reason="success")
        return observation

    native.advance = advance
    policy = FrozenPolicyAgent(
        tmp_path / "agent",
        reference=REFERENCE,
        package=package,
        media_dir=tmp_path / "benchmark/media",
        binary=binary,
        auth=auth,
        episode_factory=FixtureEpisode,
    )
    # Use the actual Runner and Agent/Adapter implementations with local client
    # signatures. HTTP transport itself is covered by the framework suite.
    agent_client = SimpleNamespace(
        reset=lambda task, episode_id, timeout_s: policy.reset(task),
        act=lambda obs, timeout_s: policy.act(obs),
        observe_end=lambda result, timeout_s: policy.observe_end(result),
        close=policy.close,
    )

    def unexpected_finish(timeout_s):
        pytest.fail("Runner must not score after the terminal Agent error")

    def collect_observation(obs, media, timeout_s):
        source = CaseMediaChannel(tmp_path / "benchmark/media")
        for ref in obs.media:
            payload = source.read(ref, obs.episode_id, obs.step_index)
            destination = media.media_dir.parent / ref.ref
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)

    adapter_client = SimpleNamespace(
        start=lambda task, episode_id, timeout_s: boundary.start(task, episode_id),
        step=lambda action, timeout_s: boundary.step(action),
        finish=unexpected_finish,
        collect_observation=collect_observation,
        close=boundary.close,
    )
    store = ResultStore(tmp_path / "results")
    store.claim("run", "case", task, Provenance(agent_version="fixture"))
    result = CaseRunner(store).run(
        "run",
        "case",
        "episode-1",
        task,
        Provenance(agent_version="fixture"),
        agent_client,
        adapter_client,
        timeout_s=20,
        media_channel=CaseMediaChannel(tmp_path / "results/run/case/media"),
        preclaimed=True,
    )
    assert native.result["success"] is True
    assert (
        result.status == "agent_error" and not result.success and result.score is None
    )
    assert result.error.details["stage"] == "agent.observe_end"
    assert "late planner failure" in result.error.message
    assert store.load("run", "case").status == "agent_error"
    assert policy.closed and policy.episode.session.done and native.closed
    assert not (policy.episode.home / "auth.json").exists()
