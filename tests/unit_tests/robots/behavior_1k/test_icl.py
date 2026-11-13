# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Offline BEHAVIOR ICL workflow contracts ported from NewRPent."""

import json
from pathlib import Path

import pytest

from roborsi.agents.config import AgentConfig
from roborsi.components.icl.build import build_package
from roborsi.evaluation.messages import Task
from robots.behavior_1k.catalog import ACTION_SCHEMA

ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize(
    "global_args", [[], ["--search"], ["--ask-for-approval", "never"]]
)
def test_launcher_scoped_approvals_with_global_flags(monkeypatch, global_args):
    import sys

    from robots.behavior_1k.agents.baseline.workspace import codex_mcp

    monkeypatch.setattr(sys, "argv", ["launcher", *global_args, "exec", "--json", "-"])
    monkeypatch.setattr(codex_mcp.shutil, "which", lambda value: sys.executable)
    calls = []
    monkeypatch.setattr(codex_mcp.os, "execv", lambda path, argv: calls.append(argv))
    codex_mcp.main()
    assert calls == [
        [
            sys.executable,
            *global_args,
            "exec",
            "-c",
            'mcp_servers.roborsi.tools.icl_demonstration.approval_mode="approve"',
            "-c",
            'mcp_servers.roborsi.tools.roborsi_action.approval_mode="approve"',
            "--json",
            "-",
        ]
    ]


@pytest.mark.parametrize("nested", ["case", "output"])
def test_learning_rejects_paths_inside_materials_before_writes(tmp_path, nested):
    from types import SimpleNamespace

    from robots.behavior_1k.agents.baseline.icl import learn

    materials = tmp_path / "materials"
    materials.mkdir()
    case_dir = materials / "case" if nested == "case" else tmp_path / "case"
    output = materials / "package" if nested == "output" else tmp_path / "package"
    with pytest.raises(ValueError, match="disjoint"):
        learn(SimpleNamespace(), materials, case_dir, output)
    assert list(materials.iterdir()) == []
    assert not case_dir.exists()


def test_agent_requires_icl_and_returns_embodied_agent(tmp_path, monkeypatch):
    from roborsi.embodied_agent import EmbodiedAgent
    from robots.behavior_1k.agents.baseline.factory import make_agent

    config = AgentConfig.load(ROOT / "robots/behavior_1k/agent.yaml")
    assert config.document["agent"]["planner"] == "codex_cli"
    from roborsi.planner.codex_cli_config import CodexCliConfig

    monkeypatch.setenv("OPENAI_API_KEY", "offline-fixture")
    monkeypatch.setenv("HOME", str(tmp_path))
    cli = CodexCliConfig.from_dict(config.document["agent"]["codex_cli"])
    assert cli.subprocess_env()["PATH"]
    assert cli.subprocess_env()["HOME"]
    with pytest.raises(ValueError, match="resources.icl"):
        make_agent(tmp_path, config=config)
    learned = tmp_path / "learned"
    learned.mkdir()
    for name in ("TASK.md", "SOURCES.md"):
        (learned / name).write_text("Test task learning")
    package = build_package(
        learned,
        tmp_path / "package",
        package_id="test",
        metadata={
            "benchmark": "behavior_1k",
            "task_name": "turning_on_radio",
            "stage": "icl",
            "action_profile": "r1pro_eef21_v1",
        },
    )

    class Config:
        parameters = {}
        resources = {**config.resources, "icl": package.root}
        embodied_agent = config.embodied_agent

    for name in ("case-a", "case-b"):
        directory = tmp_path / name
        directory.mkdir()
        agent = make_agent(directory, config=Config())
        assert isinstance(agent, EmbodiedAgent)
        assert agent.planner == "codex_cli" and agent.codex_cli.tools == "mcp"
        task = Task(
            benchmark="behavior_1k",
            task_id="turning_on_radio",
            seed=0,
            instruction="Turn on the radio.",
            max_steps=2,
            capabilities={"action_schema": ACTION_SCHEMA},
        )
        for wrong in (
            task.model_copy(update={"task_id": "sorting_vegetables"}),
            task.model_copy(update={"benchmark": "another_benchmark"}),
        ):
            with pytest.raises(ValueError, match="ICL package is for"):
                agent.reset(wrong)
            assert agent._evaluation_connector is None
        media = directory / "media"
        media.mkdir()
        monkeypatch.setenv("ROBORSI_MEDIA_DIR", str(media))
        agent.reset(task)
        assert agent._evaluation_connector is not None
        with pytest.raises(ValueError, match="only one evaluation case"):
            agent.reset(task)
        agent.close()
    (tmp_path / "case-a/workspace/TASK.md").write_text("changed")
    assert (tmp_path / "case-b/workspace/TASK.md").read_text() == "Test task learning"
    assert (package.root / "payload/TASK.md").read_text() == "Test task learning"


def test_external_icl_preset_exports_learned_materials(tmp_path):
    import shutil

    from roborsi.agents.package import export_agent
    from robots.behavior_1k.agents.baseline.icl import attach_preset

    baseline = AgentConfig.load(ROOT / "robots/behavior_1k/agent.yaml")
    source = tmp_path / "source"
    for original in baseline.input_files:
        target = source / original.relative_to(baseline.source_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
    learned = tmp_path / "learned"
    learned.mkdir()
    (learned / "TASK.md").write_text("Synthetic packaging fixture")
    (learned / "SOURCES.md").write_text("Offline test; not a model result")
    package = build_package(learned, tmp_path / "package", package_id="fixture")
    preset = attach_preset(
        package.root,
        source / "robots/behavior_1k/agent.yaml",
        tmp_path / "prepared",
    )
    with pytest.raises(ValueError, match="outside source_root"):
        attach_preset(
            package.root, source / "robots/behavior_1k/agent.yaml", source / "generated"
        )
    shutil.rmtree(source)
    shutil.rmtree(package.root)
    resolved = AgentConfig.load(preset)
    assert resolved.source_root == tmp_path / "prepared"
    assert resolved.resources["icl"] / "payload/TASK.md" in resolved.input_files
    exported = export_agent(
        preset, tmp_path / "export", name="behavior_packaging_fixture"
    )
    assert any(
        p.read_text() == "Synthetic packaging fixture"
        for p in exported.rglob("TASK.md")
    )
    launchers = list(exported.rglob("codex_mcp.py"))
    assert len(launchers) == 1
    if __import__("os").name != "nt":
        assert launchers[0].stat().st_mode & 0o111


def test_icl_record_tool_rejects_other_episode_and_escape(tmp_path):
    import gzip

    from robots.behavior_1k.agents.baseline.tools.records import iter_records

    (tmp_path / "index.json").write_text(
        json.dumps({"episodes": [{"id": "practice-1", "records": "episode.jsonl.gz"}]})
    )
    with gzip.open(tmp_path / "episode.jsonl.gz", "wt") as stream:
        stream.write(json.dumps({"action": [0] * 21}) + "\n")
    assert list(iter_records("practice-1", root=tmp_path)) == [{"action": [0] * 21}]
    with pytest.raises(ValueError, match="Unknown"):
        list(iter_records("practice-2", root=tmp_path))
    with pytest.raises(ValueError, match="Invalid"):
        list(iter_records("../practice-1", root=tmp_path))


def test_icl_learn_seals_only_learned_files(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from benchmarks.behavior_1k import official_demonstrations
    from robots.behavior_1k.agents.baseline.icl import learn

    index = {
        "task_name": "turning_on_radio",
        "episodes": [{"id": f"practice-{i}"} for i in range(1, 201)],
    }
    materials = tmp_path / "materials"
    materials.mkdir()
    (materials / "index.json").write_text(json.dumps(index))
    monkeypatch.setattr(
        official_demonstrations,
        "validate_rich_materials",
        lambda *args, **kwargs: index,
    )
    base = AgentConfig.load(ROOT / "robots/behavior_1k/agents/baseline/icl.yaml")
    closed = []

    class Config:
        resources = base.resources
        fingerprint = base.fingerprint

        def embodied_agent(self, case_dir, *, working_dir, local_tools):
            assert (working_dir / "tools/frames.py").is_file()
            assert (working_dir / "training/index.json").is_file()
            assert len(local_tools) == 1
            tool = local_tools[0]
            assert tool.handler({"episode_id": "practice-1"}).result == {
                "episode": {"id": "practice-1"}
            }
            assert "error" in tool.handler({"episode_id": "../other"}).result
            # Use the real constructor/validation boundary that the old fixture
            # missed: offline ICL must expose a tool to EmbodiedAgent.
            agent = base.embodied_agent(
                case_dir, working_dir=working_dir, local_tools=local_tools
            )
            assert agent.local_tools

            def run(task, *, system_prompt):
                assert "turning_on_radio" in task and "records.py" in system_prompt
                for name in ("TASK.md", "SOURCES.md"):
                    (working_dir / "learned" / name).write_text("Offline fixture")
                cache = working_dir / "learned/__pycache__"
                cache.mkdir()
                (cache / "policy.cpython-311.pyc").write_bytes(b"generated bytecode")
                (working_dir / "learned/policy.pyc").write_bytes(b"legacy bytecode")
                return SimpleNamespace(error=None)

            def close():
                assert not (tmp_path / "sealed").exists()
                closed.append(True)

            return SimpleNamespace(run=run, close=close)

    package = learn(Config(), materials, tmp_path / "case", tmp_path / "sealed")
    assert closed == [True]
    assert {p.name for p in (package.root / "payload").iterdir()} == {
        "TASK.md",
        "SOURCES.md",
    }
    assert package.metadata["task_name"] == "turning_on_radio"


@pytest.mark.parametrize("raises", [False, True])
def test_icl_learning_failure_closes_agent_without_package(
    tmp_path, monkeypatch, raises
):
    from types import SimpleNamespace

    from benchmarks.behavior_1k import official_demonstrations
    from robots.behavior_1k.agents.baseline.icl import learn

    index = {
        "task_name": "turning_on_radio",
        "episodes": [{"id": f"practice-{i}"} for i in range(200)],
    }
    materials = tmp_path / "materials"
    materials.mkdir()
    (materials / "index.json").write_text(json.dumps(index))
    monkeypatch.setattr(
        official_demonstrations, "validate_rich_materials", lambda *a, **k: index
    )
    base = AgentConfig.load(ROOT / "robots/behavior_1k/agents/baseline/icl.yaml")
    closed = []

    def run(*args, **kwargs):
        if raises:
            raise RuntimeError("study failure")
        return SimpleNamespace(error="study failure")

    config = SimpleNamespace(
        resources=base.resources,
        fingerprint=base.fingerprint,
        embodied_agent=lambda *a, **k: SimpleNamespace(
            run=run, close=lambda: closed.append(True)
        ),
    )
    with pytest.raises(RuntimeError, match="study failure"):
        learn(config, materials, tmp_path / "case", tmp_path / "sealed")
    assert closed == [True]
    assert not (tmp_path / "sealed").exists()


def test_materials_select_actual_noncontiguous_training_instances(monkeypatch):
    from benchmarks.behavior_1k import materials

    instances = [*range(1, 9), *range(10, 202)]
    monkeypatch.setattr(materials, "_task_id", lambda *args: 20)
    rows = [{"task_instance_id": value} for value in instances]
    monkeypatch.setattr(materials, "_episode_metadata", lambda *args: ([], rows))
    episodes = materials.training_episodes(Path("unused"), "sorting_vegetables")
    assert [row["instance_index"] for row in episodes] == instances
    assert episodes[8] == {
        "id": "practice-9",
        "task_name": "sorting_vegetables",
        "mode": "train",
        "instance_index": 10,
    }
    for invalid in (
        rows[:-1],
        [*rows[:-1], rows[0]],
        [*rows[:-1], {"task_instance_id": 301}],
    ):
        monkeypatch.setattr(materials, "_episode_metadata", lambda *args: ([], invalid))
        with pytest.raises(ValueError, match="200 distinct public training"):
            materials.training_episodes(Path("unused"), "sorting_vegetables")


def test_offline_fk_preserves_layout_and_grippers():
    from benchmarks.behavior_1k.r1pro_fk import R1ProFK

    fk = object.__new__(R1ProFK)
    fk.limits = [(-1, 1)] * 18
    observed = []

    def poses(targets):
        observed.append(targets)
        return {"left": [1, 2, 3, 4, 5, 6], "right": [7, 8, 9, 10, 11, 12]}

    fk.poses = poses
    original = [0.1, 0.2, 0.3] + [0.5] * 11 + [-0.5] + [0.75] * 7 + [1.0]
    assert fk.convert(original) == (
        [0.1, 0.2, 0.3] + [0.5] * 4 + [1, 2, 3, 4, 5, 6, -0.5, 7, 8, 9, 10, 11, 12, 1.0]
    )
    assert observed == [[0.5] * 11 + [0.75] * 7]
    with pytest.raises(ValueError):
        fk.convert([0] * 21)


@pytest.mark.parametrize("timestamp", [-0.1, 2, float("nan"), True])
def test_frames_reject_timestamps_outside_episode(tmp_path, monkeypatch, timestamp):
    import hashlib
    import sys
    from types import SimpleNamespace

    from robots.behavior_1k.agents.baseline.tools.frames import extract_frames

    monkeypatch.setitem(
        sys.modules, "imageio_ffmpeg", SimpleNamespace(get_ffmpeg_exe=lambda: "ffmpeg")
    )
    source = tmp_path / "training"
    source.mkdir()
    (source / "video.mp4").write_bytes(b"video")
    index = {
        "episodes": [
            {
                "id": "practice-1",
                "videos": {"head": {"rgb": "video.mp4"}},
                "intervals": {
                    "head": {"rgb": {"fps": 10, "frames": 20, "start_timestamp": 30}}
                },
            }
        ],
        "files": [
            {"path": "video.mp4", "sha256": hashlib.sha256(b"video").hexdigest()}
        ],
    }
    (source / "index.json").write_text(json.dumps(index))
    output = tmp_path / "frames"
    with pytest.raises(ValueError, match="inside the selected episode"):
        extract_frames("practice-1", "head", "rgb", [timestamp], output, root=source)
    assert not output.exists()
    (source / "video.mp4").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        extract_frames("practice-1", "head", "rgb", [0], output, root=source)
