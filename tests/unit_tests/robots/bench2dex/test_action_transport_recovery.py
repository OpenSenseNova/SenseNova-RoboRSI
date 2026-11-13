# Copyright 2026 The RoboRSI Contributors.
# Licensed under the Apache License, Version 2.0.

"""Kill real waiting helpers and verify action envelopes are never replayed."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import textwrap
import time
import zipfile
from pathlib import Path

import pytest

from robots.bench2dex.agents.baseline import step_helper

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux flock contract")


def write(path, value):
    temporary = path.with_suffix(".test-tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def wait_file(path):
    deadline = time.monotonic() + 5
    while not path.exists():
        assert time.monotonic() < deadline, f"Timed out waiting for {path.name}"
        time.sleep(0.01)


@pytest.fixture
def harness(tmp_path):
    workspace = tmp_path / "workspace"
    inputs = workspace / "input"
    inputs.mkdir(parents=True)
    transport = workspace / ".action_transport"
    transport.mkdir()
    payload = workspace / "action.json"
    write(payload, {"actions": [[0.25, -0.5]]})
    write(inputs / "observation.json", {"decision": 0, "policy_step": 0})
    processes = []

    def spawn(helper=Path(step_helper.__file__), *, wait_started=None, startup_delay=0):
        command = textwrap.dedent(
            """
            import json, runpy, sys, time
            from pathlib import Path

            helper, payload, workspace, wait_started, delay = sys.argv[1:]
            time.sleep(float(delay))
            module = runpy.run_path(helper)
            if wait_started:
                sleep = time.sleep

                def waiting(seconds):
                    Path(wait_started).touch()
                    sleep(seconds)

                # The first poll sleep is reached after submit has read the
                # observation and journal and entered its response wait loop.
                module["time"].sleep = waiting
            print(json.dumps(module["submit"](Path(payload), Path(workspace))))
            """
        )
        process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-c",
                command,
                str(helper),
                str(payload),
                str(workspace),
                str(wait_started) if wait_started else "",
                str(startup_delay),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(process)
        return process

    yield workspace, transport, payload, spawn
    for process in processes:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)


@pytest.mark.parametrize("number", [signal.SIGTERM, signal.SIGKILL])
def test_completed_chunk_recovers_after_helper_death_without_replaying(harness, number):
    workspace, transport, _, spawn = harness
    first = spawn()
    request = transport / "request.json"
    wait_file(request)
    original = json.loads(request.read_text())
    request.unlink()  # The evaluator accepted this envelope exactly once.
    first.send_signal(number)
    assert first.wait(timeout=5) == -number
    write(workspace / "input/observation.json", {"decision": 1, "policy_step": 1})
    second = spawn()
    wait_file(request)
    assert json.loads(request.read_text()) == {**original, "decision": 1}
    request.unlink()
    result = {"decision": 2, "policy_step": 2}
    write(workspace / "input/observation.json", result)
    out, err = second.communicate(timeout=5)
    assert second.returncode == 0, err
    assert json.loads(out) == result
    assert not request.exists()
    assert (transport / "submission.lock").is_file()


@pytest.mark.parametrize("startup_delay", [0, 0.3], ids=["immediate", "delayed"])
def test_same_inflight_payload_rejoins_wait_without_duplicate_request(
    harness, startup_delay
):
    workspace, transport, _, spawn = harness
    first = spawn()
    request = transport / "request.json"
    wait_file(request)
    request.unlink()
    first.kill()
    first.wait(timeout=5)
    waiting = workspace / "helper-waiting"
    second = spawn(wait_started=waiting, startup_delay=startup_delay)
    wait_file(waiting)
    assert second.poll() is None
    assert not request.exists()
    result = {"decision": 1, "policy_step": 8}
    write(workspace / "input/observation.json", result)
    out, err = second.communicate(timeout=5)
    assert second.returncode == 0, err
    assert json.loads(out) == result
    assert not request.exists()


def test_different_inflight_payload_is_rejected_after_owner_death(harness):
    _, transport, payload, spawn = harness
    first = spawn()
    request = transport / "request.json"
    wait_file(request)
    request.unlink()
    first.kill()
    first.wait(timeout=5)
    write(payload, {"actions": [[0.5, 0.5]]})
    second = spawn()
    _, err = second.communicate(timeout=5)
    assert second.returncode != 0
    assert "still in flight" in err
    assert not request.exists()


def test_live_owner_excludes_second_helper_and_terminal_releases_wait(harness):
    workspace, transport, _, spawn = harness
    first = spawn()
    request = transport / "request.json"
    wait_file(request)
    original = request.read_bytes()
    second = spawn()
    _, err = second.communicate(timeout=5)
    assert second.returncode != 0
    assert "already active" in err
    assert request.read_bytes() == original
    request.unlink()
    write(workspace / "input/episode_done.json", {"episode_done": True})
    out, err = first.communicate(timeout=5)
    assert first.returncode == 0, err
    assert json.loads(out) == {"episode_done": True}


def test_uninterrupted_request_and_response_match_frozen_helper(harness, tmp_path):
    workspace, transport, _, spawn = harness
    archive = (
        Path(__file__).resolve().parents[4]
        / "agents/bench2dex_compat/reference-code.zip"
    )
    old = tmp_path / "original_step.py"
    with zipfile.ZipFile(archive) as z:
        old.write_bytes(z.read("bench2dex-eval/src/gpt6_eval/step.py"))
    results = []
    for helper in [old, Path(step_helper.__file__)]:
        write(workspace / "input/observation.json", {"decision": 0, "policy_step": 0})
        child = spawn(helper)
        request = transport / "request.json"
        wait_file(request)
        wire = request.read_bytes()
        request.unlink()
        write(workspace / "input/observation.json", {"decision": 1, "policy_step": 7})
        out, err = child.communicate(timeout=5)
        assert child.returncode == 0, err
        results.append((wire, out))
    assert results[0] == results[1]


def test_corrupt_journal_never_replays_automatically(harness):
    workspace, transport, _, spawn = harness
    journal = transport / "submission.json"
    journal.write_text("not-json")
    child = spawn()
    child.communicate(timeout=5)
    assert child.returncode != 0
    assert not (transport / "request.json").exists()
    assert (
        json.loads((workspace / "input/observation.json").read_text())["decision"] == 0
    )


@pytest.mark.parametrize("startup_delay", [0, 0.3], ids=["immediate", "delayed"])
def test_interruption_before_request_publication_fails_closed(
    harness, monkeypatch, startup_delay
):
    workspace, transport, payload, spawn = harness
    original = Path.replace

    def interrupt(path, target):
        if path.name == "request.tmp":
            raise OSError("injected interruption before publication")
        return original(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", interrupt)
        with pytest.raises(OSError, match="injected interruption"):
            step_helper.submit(payload, workspace)
    assert (transport / "submission.json").exists()
    assert not (transport / "request.json").exists()
    waiting = workspace / "helper-waiting"
    child = spawn(wait_started=waiting, startup_delay=startup_delay)
    wait_file(waiting)
    assert child.poll() is None
    assert not (transport / "request.json").exists()
    write(workspace / "input/episode_done.json", {"episode_done": True})
    out, err = child.communicate(timeout=5)
    assert child.returncode == 0, err
    assert json.loads(out) == {"episode_done": True}
