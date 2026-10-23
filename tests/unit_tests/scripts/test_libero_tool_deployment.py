# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts.deploy import libero_tools

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/models/roborsi_27b.yaml"


def test_public_tool_plan_is_pinned_and_excludes_optimizer(tmp_path):
    d = libero_tools.settings(CONFIG, tmp_path)
    commands = libero_tools.preparation_commands(d, tmp_path)
    downloads = [json.loads(c[-1]) for c in commands if "-c" in c]
    assert downloads[0]["repo_id"] == "RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT"
    assert downloads[0]["revision"] == "6222623f635769bfc73c9472e29fab9b7fd8e027"
    assert "optimizer.pt" not in downloads[0]["allow_patterns"]
    assert (
        "physical-intelligence/libero/norm_stats.json" in downloads[0]["allow_patterns"]
    )
    assert downloads[1]["repo_id"] == "facebook/sam3"
    assert downloads[1]["allow_patterns"] == ["sam3.pt"]
    vla = libero_tools.service_command(d, tmp_path, "vla")
    assert vla[vla.index("--embodiment") + 1] == "libero"
    assert vla[vla.index("--model-backend") + 1] == "openpi_pytorch"
    assert "robots.libero.pi05_vla_server" in vla
    assert "robots.libero.sam3_server" in libero_tools.service_command(
        d, tmp_path, "sam3"
    )


def test_dry_run_never_installs_downloads_or_starts(monkeypatch, tmp_path, capsys):
    execute = Mock(side_effect=AssertionError("dry-run must not execute"))
    monkeypatch.setattr(libero_tools.subprocess, "run", execute)
    monkeypatch.setattr(libero_tools.subprocess, "Popen", execute)
    assert (
        libero_tools.main(
            [
                "run",
                "--model-config",
                str(CONFIG),
                "--work-dir",
                str(tmp_path / "new"),
                "--dry-run",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["services"]
    assert not (tmp_path / "new").exists()


def test_owned_process_cleanup_escalates_only_after_timeout():
    import subprocess

    child = Mock()
    child.poll.return_value = None
    child.wait.side_effect = [subprocess.TimeoutExpired("service", 15), 0]
    libero_tools.terminate(child)
    child.terminate.assert_called_once()
    child.kill.assert_called_once()
    finished = Mock()
    finished.poll.return_value = 0
    libero_tools.terminate(finished)
    finished.terminate.assert_not_called()


def test_refuses_unknown_ports_and_unpinned_weights(monkeypatch, tmp_path):
    import copy

    import yaml

    original = yaml.safe_load(CONFIG.read_text())
    for field, value in [
        ("port", 0),
        ("revision", "main"),
        ("checkpoint", "../escape"),
    ]:
        doc = copy.deepcopy(original)
        doc["vla"][field] = value
        with pytest.raises(ValueError):
            libero_tools.model_settings(doc)


def source_fixture(monkeypatch, tmp_path):
    import hashlib

    contents = b"pinned service source\n"
    plan = {
        "image": "registry.test/pro@sha256:" + "a" * 64,
        "destination": str(tmp_path / "image-sources" / "fixture"),
        "sha256": {"roborsi/service.py": hashlib.sha256(contents).hexdigest()},
    }
    monkeypatch.setattr(libero_tools, "image_source_plan", lambda _: plan)
    monkeypatch.setattr(libero_tools, "IMAGE_ROOTS", ())
    monkeypatch.delenv("ROBORSI_IMAGE_TOOLS_ROOT", raising=False)
    return plan, contents


def test_verified_cci_sources_need_no_docker(monkeypatch, tmp_path):
    plan, contents = source_fixture(monkeypatch, tmp_path)
    root = tmp_path / "cci-code"
    (root / "roborsi").mkdir(parents=True)
    (root / "roborsi/service.py").write_bytes(contents)
    monkeypatch.setattr(libero_tools, "IMAGE_ROOTS", (root,))
    monkeypatch.setattr(
        libero_tools.subprocess, "run", Mock(side_effect=AssertionError("no Docker"))
    )
    assert libero_tools.prepare_image_sources(tmp_path) == root


def test_explicit_invalid_source_is_not_silently_replaced(monkeypatch, tmp_path):
    source_fixture(monkeypatch, tmp_path)
    monkeypatch.setenv("ROBORSI_IMAGE_TOOLS_ROOT", str(tmp_path / "wrong"))
    monkeypatch.setattr(
        libero_tools.subprocess, "run", Mock(side_effect=AssertionError("no Docker"))
    )
    with pytest.raises(ValueError, match="differs from lock"):
        libero_tools.prepare_image_sources(tmp_path)


@pytest.mark.parametrize("valid", [True, False])
def test_export_hash_check_cache_and_container_cleanup(monkeypatch, tmp_path, valid):
    import subprocess

    plan, contents = source_fixture(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(libero_tools.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(
        libero_tools.subprocess, "check_output", lambda *a, **k: "owned-container\n"
    )

    def execute(command, **kwargs):
        calls.append(command)
        if command[:2] == ["docker", "cp"]:
            target = Path(command[-1]) / "roborsi/service.py"
            target.parent.mkdir(parents=True)
            target.write_bytes(contents if valid else b"wrong source")
        return subprocess.CompletedProcess(
            command, 1 if command[:3] == ["docker", "image", "inspect"] else 0
        )

    monkeypatch.setattr(libero_tools.subprocess, "run", execute)
    if valid:
        root = libero_tools.prepare_image_sources(tmp_path)
        assert root == Path(plan["destination"])
        assert json.loads((tmp_path / "image-source.json").read_text()) == plan
        prior = len(calls)
        assert libero_tools.prepare_image_sources(tmp_path) == root
        assert len(calls) == prior
    else:
        with pytest.raises(ValueError, match="differs from lock"):
            libero_tools.prepare_image_sources(tmp_path)
        assert not Path(plan["destination"]).exists()
    assert ["docker", "rm", "owned-container"] in calls
    assert not any(c[:2] in (["docker", "start"], ["docker", "run"]) for c in calls)


def test_service_children_receive_verified_sources(monkeypatch, tmp_path):
    import socket
    from unittest.mock import MagicMock

    d = libero_tools.settings(CONFIG, tmp_path)
    source = tmp_path / "verified-image-code"
    monkeypatch.setattr(libero_tools, "prepare_image_sources", lambda _: source)
    monkeypatch.setattr(socket, "socket", lambda: MagicMock())
    monkeypatch.setattr(libero_tools, "healthy", lambda *a, **k: True)
    children = []

    def spawn(command, **kwargs):
        assert kwargs["env"]["ROBORSI_IMAGE_TOOLS_ROOT"] == str(source)
        assert kwargs["env"]["SAM3_CHECKPOINT_PATH"] == str(
            tmp_path / d["sam3"]["checkpoint"]
        )
        child = Mock()
        child.poll.return_value = None
        children.append(child)
        return child

    monkeypatch.setattr(libero_tools.subprocess, "Popen", spawn)
    with libero_tools.services(d, tmp_path) as urls:
        assert set(urls) == {"vla", "sam3"}
        assert len(children) == 2
    for child in children:
        child.terminate.assert_called_once()
