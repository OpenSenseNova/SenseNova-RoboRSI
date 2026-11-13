# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Context projection and bounded, confined reference reads."""

import base64
import sys
from pathlib import Path

import pytest

from roborsi.components.icl import ContextExports, build_package, materialize_package
from roborsi.components.icl.adapters import context_from, make_icl_tools


@pytest.fixture
def package(tmp_path):
    source = tmp_path / "source"
    (source / "skills/check").mkdir(parents=True)
    (source / "notes.txt").write_text(
        "参考经验一二三\n", encoding="utf-8", newline="\n"
    )
    (source / "skills/check/SKILL.md").write_text(
        "---\nname: check\ndescription: Inspect observations.\n---\nRead first.\n"
    )
    (source / "photo.jpg").write_bytes(b"\xff\xd8\xff\xe0sample\xff\xd9")
    return build_package(
        source,
        tmp_path / "package",
        package_id="references",
        exports=ContextExports(
            memory=(("Notes", "notes.txt"),),
            skill_paths=("skills/check",),
            initial_context=(
                {"type": "image", "path": "photo.jpg", "media_type": "image/jpeg"},
            ),
        ),
    )


@pytest.mark.parametrize("planner", ["api", "codex", "claude_code", "codex_cli"])
def test_explicit_preload_and_none(package, planner):
    ctx = context_from(package, planner=planner, skill_mode="preload")
    assert ctx.memory[0].text == "参考经验一二三\n"
    assert [p.name for p in ctx.skills] == ["SKILL.md"]
    assert ctx.skill_paths == ()
    assert ctx.initial_context[0].media_type == "image/jpeg"
    empty = context_from(package, planner=planner, skill_mode="none")
    assert empty.skills == empty.skill_paths == ()
    assert empty.memory == ctx.memory
    assert empty.initial_context == ctx.initial_context


@pytest.mark.parametrize("planner", ["codex", "claude_code", "codex_cli"])
def test_non_api_requires_explicit_skill_mode(package, planner):
    with pytest.raises(ValueError, match="skill"):
        context_from(package, planner=planner)
    with pytest.raises(ValueError, match="api"):
        context_from(package, planner=planner, skill_mode="on_demand")


def test_api_declared_and_invalid_modes(package):
    context = context_from(package, planner="api")
    assert context.skills == ()
    assert context.skill_paths == package.skill_paths
    assert context_from(package, planner="api", skill_mode="on_demand").skill_paths
    with pytest.raises(ValueError):
        context_from(package, planner="typo")
    with pytest.raises(ValueError):
        context_from(package, planner="api", skill_mode="typo")


def test_unicode_paging_and_list(package):
    tools = {tool.name: tool for tool in make_icl_tools(package, max_read_bytes=6)}
    first = tools["icl_read_text"].handler({"path": "notes.txt", "limit": 100}).result
    assert first == {
        "path": "notes.txt",
        "text": "参考",
        "next_offset": 2,
        "truncated": True,
    }
    second = (
        tools["icl_read_text"]
        .handler({"path": "notes.txt", "offset": 2, "limit": 2})
        .result
    )
    assert second["text"] == "经验"
    listing = tools["icl_list"].handler({"limit": 1}).result
    assert len(listing["entries"]) == 1
    assert listing["next_offset"] == 1
    assert listing["truncated"] is True
    assert all(tool.propagate_errors for tool in tools.values())


@pytest.mark.parametrize(
    "path", ["../secret", "/tmp/secret", "C:/secret", "a\\b", "//host/path"]
)
def test_tool_path_errors_are_recoverable(package, path):
    read = make_icl_tools(package)[1]
    result = read.handler({"path": path}).result
    assert result["retryable"] is True
    assert "text" not in result


def test_reads_current_workspace_and_rejects_late_links(
    package, tmp_path, symlink_support
):
    workspace = materialize_package(package, tmp_path / "workspace")
    read = make_icl_tools(workspace)[1]
    (workspace.root / "notes.txt").write_text("updated", encoding="utf-8")
    assert read.handler({"path": "notes.txt"}).result["text"] == "updated"
    external = tmp_path / "secret"
    external.write_text("PRIVATE")
    (workspace.root / "notes.txt").unlink()
    (workspace.root / "notes.txt").symlink_to(external)
    assert "text" not in read.handler({"path": "notes.txt"}).result


def test_sealed_tampering_is_an_operational_failure(package):
    read = make_icl_tools(package)[1]
    (package.root / "payload/notes.txt").write_text("tampered")
    with pytest.raises(ValueError, match="integrity|hash"):
        read.handler({"path": "notes.txt"})


@pytest.mark.parametrize(
    "data,mime",
    [
        (b"\xff\xd8\xff\xe0sample\xff\xd9", "image/jpeg"),
        (b"RIFF\x04\x00\x00\x00WEBPsample", "image/webp"),
    ],
)
def test_image_blocks_keep_actual_mime(package, tmp_path, data, mime):
    workspace = materialize_package(package, tmp_path / "workspace")
    (workspace.root / "photo.jpg").write_bytes(data)
    result = make_icl_tools(workspace)[2].handler({"path": "photo.jpg"})
    block = result.content_blocks[-1]
    assert block["type"] == "image"
    assert block["source"]["media_type"] == mime
    assert base64.b64decode(block["source"]["data"]) == data
    from roborsi.planner.api_loop import _content_blocks_to_pydantic

    _, content = _content_blocks_to_pydantic(result.content_blocks)
    assert any(getattr(item, "media_type", None) == mime for item in content)
    from roborsi.planner.claude_code import _tool_result_to_mcp

    sdk_image = _tool_result_to_mcp(result)["content"][-1]
    assert sdk_image["mimeType"] == mime
    assert base64.b64decode(sdk_image["data"]) == data


def test_read_image_limits_and_bad_arguments(package):
    tools = make_icl_tools(package, max_read_bytes=3)
    assert "error" in tools[2].handler({"path": "photo.jpg"}).result
    for args in (
        {"path": "notes.txt", "offset": -1},
        {"path": "notes.txt", "limit": True},
        {"path": "notes.txt", "unknown": 1},
    ):
        assert "error" in tools[1].handler(args).result
    with pytest.raises(ValueError):
        make_icl_tools(package, name_prefix="bad-name")


def test_large_text_page_remains_model_readable(package, tmp_path):
    import json

    workspace = materialize_package(package, tmp_path / "workspace")
    (workspace.root / "notes.txt").write_text("\x01" * 70000, encoding="utf-8")
    result = make_icl_tools(workspace)[1].handler({"path": "notes.txt"})
    visible = json.loads(result.content_blocks[0]["text"])
    assert visible["text"] == result.result["text"]
    assert visible["truncated"] is True
    assert visible["next_offset"] > 0


def test_export_image_mime_mismatch_fails(package, tmp_path):
    from dataclasses import replace

    workspace = materialize_package(package, tmp_path / "workspace")
    workspace = replace(
        workspace,
        exports=ContextExports(
            initial_context=(
                {"type": "image", "path": "photo.jpg", "media_type": "image/png"},
            )
        ),
    )
    with pytest.raises(ValueError, match="media_type"):
        context_from(workspace, planner="api")


def test_late_intermediate_directory_link(package, tmp_path, symlink_support):
    import shutil

    workspace = materialize_package(package, tmp_path / "workspace")
    read = make_icl_tools(workspace)[1]
    shutil.rmtree(workspace.root / "skills")
    external = tmp_path / "external"
    external.mkdir()
    (external / "secret.txt").write_text("PRIVATE")
    (workspace.root / "skills").symlink_to(external, target_is_directory=True)
    assert "text" not in read.handler({"path": "skills/secret.txt"}).result


def test_preload_discovers_declared_collection_and_deduplicates(package, tmp_path):
    from dataclasses import replace

    workspace = materialize_package(package, tmp_path / "workspace")
    workspace = replace(
        workspace,
        exports=ContextExports(
            skills=("skills/check/SKILL.md",), skill_paths=("skills", "skills/check")
        ),
    )
    context = context_from(workspace, planner="codex", skill_mode="preload")
    assert context.skills == (workspace.root / "skills/check/SKILL.md",)
    assert context.skill_paths == ()


def test_list_accepts_integral_float_limit_without_mutating_arguments(package):
    arguments = {"limit": 1.0}
    result = make_icl_tools(package)[0].handler(arguments).result
    assert result == {
        "entries": [{"path": "notes.txt", "type": "file"}],
        "next_offset": 1,
        "truncated": True,
    }
    assert type(arguments["limit"]) is float


def test_text_accepts_integral_float_range_without_mutating_arguments(package):
    arguments = {"path": "notes.txt", "offset": 1.0, "limit": 1.0}
    result = make_icl_tools(package)[1].handler(arguments).result
    assert result == {
        "path": "notes.txt",
        "text": "考",
        "next_offset": 2,
        "truncated": True,
    }
    assert type(result["next_offset"]) is int
    assert type(arguments["offset"]) is float
    assert type(arguments["limit"]) is float


@pytest.mark.parametrize("tool_index", [0, 1])
@pytest.mark.parametrize("field", ["offset", "limit"])
@pytest.mark.parametrize("value", [True, False, 1.5])
def test_noninteger_ranges_are_recoverable(package, tool_index, field, value):
    arguments = {field: value}
    if tool_index == 1:
        arguments["path"] = "notes.txt"
    result = make_icl_tools(package)[tool_index].handler(arguments).result
    assert result["retryable"] is True
    assert "error" in result
    assert "text" not in result
    assert "entries" not in result


def test_nul_path_is_recoverable(package):
    result = make_icl_tools(package)[1].handler({"path": "bad\x00name"}).result
    assert result["retryable"] is True
    assert "error" in result
    assert "text" not in result


def test_supplementary_unicode_page_remains_model_readable(package, tmp_path):
    import json

    workspace = materialize_package(package, tmp_path / "workspace")
    text = "🚀" * 10000
    (workspace.root / "notes.txt").write_text(text, encoding="utf-8")
    read = make_icl_tools(workspace)[1]
    first = read.handler({"path": "notes.txt"})
    visible = json.loads(first.content_blocks[0]["text"])
    assert visible == first.result
    assert visible["truncated"] is True
    assert 0 < visible["next_offset"] < len(text)
    assert visible["text"] == text[: visible["next_offset"]]

    pieces = [visible["text"]]
    offset = visible["next_offset"]
    while offset is not None:
        next_page = read.handler({"path": "notes.txt", "offset": offset})
        next_visible = json.loads(next_page.content_blocks[0]["text"])
        assert next_visible == next_page.result
        end = next_visible["next_offset"] or len(text)
        assert offset < end <= len(text)
        assert next_visible["text"] == text[offset:end]
        assert next_visible["truncated"] is (next_visible["next_offset"] is not None)
        pieces.append(next_visible["text"])
        offset = next_visible["next_offset"]
    assert "".join(pieces) == text


@pytest.mark.parametrize("planner", ["codex", "claude_code", "codex_cli"])
def test_declared_skill_files_keep_memory_and_initial_content(
    package, tmp_path, planner
):
    from dataclasses import replace

    workspace = materialize_package(package, tmp_path / "workspace")
    workspace = replace(
        workspace,
        exports=ContextExports(
            memory=(("Notes", "notes.txt"),),
            skills=("skills/check/SKILL.md",),
            initial_context=(
                {"type": "text", "path": "notes.txt"},
                {"type": "image", "path": "photo.jpg", "media_type": "image/jpeg"},
            ),
        ),
    )
    context = context_from(workspace, planner=planner)
    assert context.skills == (workspace.root / "skills/check/SKILL.md",)
    assert context.skill_paths == ()
    assert context.memory[0].text == "参考经验一二三\n"
    assert context.initial_context[0] == "参考经验一二三\n"
    assert context.initial_context[1].data == b"\xff\xd8\xff\xe0sample\xff\xd9"
    assert context.initial_context[1].media_type == "image/jpeg"


@pytest.mark.skipif(sys.platform != "linux", reason="Codex CLI runtime requires Linux")
def test_codex_cli_receives_preloaded_icl_context(package, tmp_path):
    import json
    from dataclasses import replace

    from roborsi.embodied_agent import EmbodiedAgent
    from roborsi.planner.codex_cli_config import CodexCliConfig

    workspace = materialize_package(package, tmp_path / "workspace")
    image = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
    )
    (workspace.root / "photo.png").write_bytes(image)
    (workspace.root / "initial.txt").write_text(
        "Initial reference text.", encoding="utf-8"
    )
    workspace = replace(
        workspace,
        exports=ContextExports(
            memory=(("Notes", "notes.txt"),),
            skills=("skills/check/SKILL.md",),
            skill_paths=("skills", "skills/check"),
            initial_context=(
                {"type": "text", "path": "initial.txt"},
                {"type": "image", "path": "photo.png", "media_type": "image/png"},
            ),
        ),
    )
    context = context_from(workspace, planner="codex_cli", skill_mode="preload")
    assert context.skills == (workspace.root / "skills/check/SKILL.md",)
    assert context.skill_paths == ()
    captured = tmp_path / "cli-input.json"
    executable = tmp_path / "fake-codex"
    executable.write_text(
        f"#!{sys.executable}\n"
        + """
import json
import os
import sys
from pathlib import Path

if sys.argv[1:] == ["--version"]:
    print("codex-cli icl-test")
    raise SystemExit(0)
Path(os.environ["ICL_CAPTURE_PATH"]).write_text(
    json.dumps({"prompt": sys.stdin.read(), "argv": sys.argv}), encoding="utf-8"
)
print(json.dumps({"type": "thread.started", "thread_id": "icl-offline"}))
print(json.dumps({"type": "item.completed", "item": {
    "id": "answer", "type": "agent_message", "text": "References received"
}}))
print(json.dumps({"type": "turn.completed"}))
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    agent = EmbodiedAgent(
        mcp_servers=(),
        output_dir=tmp_path / "output",
        working_dir=workspace.root,
        planner="codex_cli",
        model="offline-model",
        max_turns=None,
        max_tokens=None,
        planner_timeout_s=10,
        codex_cli=CodexCliConfig(
            expected_version="codex-cli icl-test",
            executable=str(executable),
            inherit_env=False,
            env={"ICL_CAPTURE_PATH": str(captured)},
        ),
        local_tools=make_icl_tools(workspace),
    )
    try:
        result = agent.run(
            "Read the selected materials.",
            system_prompt="Use references as context.",
            memory=context.memory,
            skills=context.skills,
            skill_paths=context.skill_paths,
            initial_context=context.initial_context,
        )
        assert result.error is None
        actual = json.loads(captured.read_text(encoding="utf-8"))
        assert "参考经验一二三\n" in actual["prompt"]
        assert "Initial reference text." in actual["prompt"]
        assert actual["prompt"].count("Read first.") == 1
        argv = actual["argv"]
        images = [
            Path(argv[index + 1])
            for index, value in enumerate(argv)
            if value == "--image"
        ]
        assert len(images) == 1
        assert images[0].read_bytes() == image
    finally:
        agent.close()
