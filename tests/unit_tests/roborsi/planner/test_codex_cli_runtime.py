# Copyright 2026 The RoboRSI Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exercise CLI subprocess ownership and the real loopback MCP boundary."""

from __future__ import annotations

import base64
import json
import os
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from pydantic_ai.messages import BinaryContent

from roborsi.planner.codex_cli import CodexCliPlanner, preflight_codex_cli
from roborsi.planner.codex_cli_config import CodexCliConfig
from roborsi.tools.toolkit import ToolResult

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux CLI runtime")

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)


_PROGRAM = r"""
import json, os, pathlib, signal, subprocess, sys, time, urllib.request
if sys.argv[1:] == ["--version"]:
    print("codex-cli test.1")
    raise SystemExit(0)
pathlib.Path("started.json").write_text(json.dumps({"argv": sys.argv, "pid": os.getpid(), "stdin": sys.stdin.read(), "env": os.environ.get("EXAMPLE_VALUE")}))
def emit(value):
    print(json.dumps(value), flush=True)
emit({"type":"thread.started", "thread_id":"thread-test"})
mode = os.environ.get("STUB_MODE", "text")
for item in json.loads(os.environ.get("STUB_TOOL_ITEMS", "[]")):
    emit({"type":"item.started", "item":item})
    emit({"type":"item.updated", "item":item})
    emit({"type":"item.completed", "item":item})
    emit({"type":"item.completed", "item":item})
if mode in ("hang", "detached"):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(90)"], start_new_session=mode == "detached")
    pathlib.Path("grandchild.pid").write_text(str(child.pid))
    while True:
        time.sleep(.1)
if mode == "stderr":
    sys.stderr.write("e" * 200000)
    sys.stderr.flush()
if mode == "capture":
    args = sys.argv
    override = next(args[i+1] for i in range(len(args)-1) if args[i] == "-c" and args[i+1].startswith("model_providers.test.base_url="))
    url = json.loads(override.split("=",1)[1]) + "/responses"
    request = urllib.request.Request(url, b'{"model":"test-model","stream":false,"include":["reasoning.encrypted_content"],"store":false}',headers={"Content-Type":"application/json", "Authorization":"Bearer hidden"})
    with urllib.request.urlopen(request) as response:
        body = response.read()
    pathlib.Path("provider-body.bin").write_bytes(body)
if mode in ("mcp", "no_ack", "wrong_ack", "blocked", "finish_only", "finish_only_no_ack"):
    args = sys.argv
    override = next(args[i + 1] for i in range(len(args)-1) if args[i] == "-c" and args[i + 1].startswith("mcp_servers.roborsi.url="))
    url = json.loads(override.split("=",1)[1])
    def rpc(method, params, req_id):
        request = urllib.request.Request(url, json.dumps({"jsonrpc":"2.0","id":req_id,"method":method,"params":params}).encode(), headers={"Content-Type":"application/json", "Accept":"application/json, text/event-stream"})
        with urllib.request.urlopen(request) as response:
            return json.load(response)["result"]
    rpc("initialize", {"protocolVersion":"2024-11-05", "capabilities":{}, "clientInfo":{"name":"cli-test","version":"1"}}, 1)
    names = ["finish"] if mode.startswith("finish_only") else ["move", "finish", "move"] if mode != "blocked" else ["blocked"]
    for i, name in enumerate(names):
        result = rpc("tools/call", {"name":name,"arguments":{}}, i+2)
        if mode in ("no_ack", "finish_only_no_ack") and name == "finish":
            continue
        if mode == "wrong_ack" and name == "finish":
            result = {"content":[{"type":"text","text":json.dumps({"_finish":True,"_roborsi_terminal_ack":"wrong"})}]}
        emit({"type":"item.completed", "item":{"id":str(i),"type":"mcp_tool_call","server":"roborsi","tool":name,"arguments":{},"result":result,"status":"completed"}})
if mode == "malformed":
    print('{"type":', flush=True)
    time.sleep(30)
if mode == "failed":
    emit({"type":"turn.failed", "error":{"message":"provider refused"}})
    raise SystemExit(3)
emit({"type":"future.event", "extra":42})
emit({"type":"item.completed", "item":{"id":"answer","type":"agent_message","text":"done"}})
if mode != "missing_usage":
    usage = {"input_tokens":100,"cached_input_tokens":40,"output_tokens":12}
    if mode == "cache_write_usage":
        usage["cache_write_input_tokens"] = 17
    emit({"type":"turn.completed", "usage":usage})
    if mode == "duplicate_usage":
        emit({"type":"turn.completed", "usage":{"input_tokens":100,"cached_input_tokens":40,"output_tokens":12}})
else:
    emit({"type":"turn.completed"})
"""


def make_planner(tmp_path, *, mode="text", tools="external", timeout_s=5, **changes):
    executable = tmp_path / "fake cli"
    executable.write_text(f"#!{sys.executable}\n" + _PROGRAM, encoding="utf-8")
    executable.chmod(0o755)
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    config = CodexCliConfig.from_dict(
        {
            "executable": str(executable),
            "expected_version": "codex-cli test.1",
            "env": {"STUB_MODE": mode, "EXAMPLE_VALUE": "private-value"},
            "tools": tools,
            **changes,
        }
    )
    planner = CodexCliPlanner(
        output_dir=tmp_path / "output",
        repo_root=workspace,
        model="test-model",
        timeout_s=timeout_s,
        dashboard_events=None,
        reasoning_effort="max",
        config=config,
    )
    return planner, workspace


def solve(planner, **kwargs):
    return planner.solve(
        system_prompt="",
        user_message="original\ntext\n",
        toolkit=None,
        max_turns=None,
        **kwargs,
    )


class RecordingToolkit:
    def __init__(self):
        self.calls = []
        self.cancelled = threading.Event()
        self.active = threading.Event()

    def get_tools_spec(self):
        return [
            {"name": name, "description": name, "input_schema": {"type": "object"}}
            for name in ("move", "finish", "blocked")
        ]

    def execute_tool(self, name, args):
        self.calls.append(name)
        if name == "blocked":
            self.active.set()
            assert self.cancelled.wait(10), "deferred tool was not released"
            return ToolResult(name, {"cancelled": True})
        if name == "finish":
            return ToolResult(
                name,
                {"_finish": True, "status": "success", "summary": "native terminal"},
            )
        return ToolResult(name, {"observation": "new", "_image_bytes": b"image-bytes"})

    def cancel_pending(self):
        self.cancelled.set()

    def cancel_active_and_wait(self):
        self.cancelled.set()


def test_real_process_preserves_prompt_cwd_env_and_separate_invocations(tmp_path):
    planner, workspace = make_planner(tmp_path)
    first = solve(planner)
    second = solve(planner)
    assert first.error is None and second.error is None
    launch = json.loads((workspace / "started.json").read_text())
    assert launch["stdin"] == "original\ntext\n"
    assert launch["env"] == "private-value"
    assert launch["argv"][1:5] == ["exec", "--json", "--color", "never"]
    assert launch["argv"][-1] == "-"
    calls = sorted((tmp_path / "output" / "codex-cli").iterdir())
    assert [p.name for p in calls] == ["call-0001", "call-0002"]
    assert (calls[0] / "input.txt").read_bytes() == b"original\ntext\n"
    assert '"future.event"' in (calls[0] / "events.jsonl").read_text()
    manifest = (calls[0] / "runtime.json").read_text()
    assert "private-value" not in manifest
    assert json.loads(manifest)["version"] == "codex-cli test.1"
    assert first.stats["total_input_tokens"] == 100
    assert first.stats["total_output_tokens"] == 12
    assert first.stats["usage_status"] == "complete"
    assert first.stats["total_tokens"] == 112


@pytest.mark.parametrize("mode,expected", [("cache_write_usage", 17), ("text", None)])
def test_cli_cache_write_usage_is_preserved_without_guessing_missing_counts(
    tmp_path, mode, expected
):
    planner, _ = make_planner(tmp_path, mode=mode)
    result = solve(planner)
    assert result.error is None
    assert result.stats["total_cache_creation_input_tokens"] == expected
    assert result.stats["total_input_tokens"] == 100
    assert result.stats["total_tokens"] == 112


@pytest.mark.parametrize(
    "mode,reason", [("text", "capture_not_observed"), ("failed", "turn_failed")]
)
def test_capture_without_provider_request_is_visible_and_keeps_primary_errors(
    tmp_path, mode, reason
):
    planner, _ = make_planner(
        tmp_path,
        mode=mode,
        provider_capture={
            "upstream_base_url": "http://127.0.0.1:1/v1",
            "model_provider": "test",
        },
    )
    result = solve(planner)
    assert result.stop_reason == reason
    assert result.error_kind == "infrastructure"
    directory = tmp_path / "output/codex-cli/call-0001"
    assert '"thread.started"' in (directory / "events.jsonl").read_text()
    runtime = json.loads((directory / "runtime.json").read_text())
    assert runtime["provider_capture"]["requests"] == 0
    if mode == "text":
        assert result.messages[-1]["item"]["text"] == "done"


@pytest.mark.parametrize(
    "changes", [{"expected_version": "codex-cli other"}, {"expected_sha256": "0" * 64}]
)
def test_preflight_mismatch_never_starts_exec(tmp_path, changes):
    planner, workspace = make_planner(tmp_path, **changes)
    result = solve(planner)
    assert result.error_kind == "configuration"
    assert not (workspace / "started.json").exists()
    assert (tmp_path / "output/codex-cli/call-0001/result.json").exists()


def test_preflight_returns_same_absolute_executable(tmp_path):
    planner, workspace = make_planner(tmp_path)
    identity = preflight_codex_cli(planner.config, repo_root=workspace)
    assert identity["executable"] == str(tmp_path / "fake cli")
    assert identity["version"] == "codex-cli test.1"
    assert len(identity["sha256"]) == 64


@pytest.mark.parametrize(
    "spelling,host_binary",
    [
        ("./codex", False),
        ("./codex", True),
        ("./codex with spaces", True),
        ("bin/codex", True),
        ("absolute", True),
        ("codex", True),
        ("relative_path", True),
        ("empty_path", True),
    ],
)
def test_executable_resolution_uses_working_directory_or_explicit_path(
    tmp_path, monkeypatch, spelling, host_binary
):
    host = tmp_path / "host"
    host.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    path_dir = tmp_path / "path-bin"
    path_dir.mkdir()
    search_path = str(path_dir)
    if spelling == "absolute":
        expected = workspace / "codex"
        spelling = str(expected)
    elif spelling == "codex":
        expected = path_dir / "codex"
    elif spelling in {"relative_path", "empty_path"}:
        search_path = "bin" if spelling == "relative_path" else ""
        expected = workspace / search_path / "codex"
        spelling = "codex"
    else:
        expected = workspace / spelling
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.write_text(f"#!{sys.executable}\n" + _PROGRAM, encoding="utf-8")
    expected.chmod(0o755)
    if host_binary:
        decoy = host / ("bin" if search_path == "bin" else "") / expected.name
        decoy.parent.mkdir(parents=True, exist_ok=True)
        decoy.write_text(expected.read_text(), encoding="utf-8")
        decoy.chmod(0o755)
    monkeypatch.chdir(host)
    planner, _ = make_planner(
        tmp_path,
        executable=spelling,
        inherit_env=False,
        env={"PATH": search_path},
    )
    identity = planner.preflight()
    assert identity["executable"] == str(expected.resolve())
    result = solve(planner)
    assert result.error is None
    launch = json.loads((workspace / "started.json").read_text())
    assert launch["argv"][0] == str(expected.resolve())


def test_bare_executable_requires_path_in_the_launch_environment(tmp_path):
    planner, workspace = make_planner(
        tmp_path, executable="codex", inherit_env=False, env={}
    )
    executable = workspace / "codex"
    executable.write_text(f"#!{sys.executable}\n" + _PROGRAM, encoding="utf-8")
    executable.chmod(0o755)
    with pytest.raises(ValueError, match="executable was not found"):
        planner.preflight()


_COMMAND_ITEM = {
    "id": "command-1",
    "type": "command_execution",
    "command": "pwd",
    "aggregated_output": "/workspace\n",
    "exit_code": 0,
    "status": "completed",
}
_OTHER_TOOL_ITEMS = [
    {
        "id": "edit-1",
        "type": "file_change",
        "changes": [{"path": "result.txt", "kind": "add"}],
        "status": "completed",
    },
    {"id": "search-1", "type": "web_search", "query": "robot tasks"},
    {
        "id": "external-mcp-1",
        "type": "mcp_tool_call",
        "server": "external",
        "tool": "lookup",
        "arguments": {},
        "result": {"content": [{"type": "text", "text": "found"}]},
        "status": "completed",
    },
]


@pytest.mark.parametrize(
    "mode,items,expected",
    [
        ("finish_only", [_COMMAND_ITEM], 2),
        ("finish_only", [_COMMAND_ITEM, *_OTHER_TOOL_ITEMS], 5),
        ("finish_only_no_ack", [_COMMAND_ITEM], 2),
    ],
)
def test_mcp_tool_statistics_combine_dispatches_and_unique_cli_tools(
    tmp_path, mode, items, expected
):
    planner, _ = make_planner(
        tmp_path,
        tools="mcp",
        env={"STUB_MODE": mode, "STUB_TOOL_ITEMS": json.dumps(items)},
    )
    toolkit = RecordingToolkit()
    result = planner.solve(
        system_prompt="", user_message="task", toolkit=toolkit, max_turns=None
    )
    assert toolkit.calls == ["finish"]
    assert result.stop_reason == (
        "terminal_feedback_missing" if mode.endswith("no_ack") else "completed"
    )
    assert result.stats["tool_calls"] == expected
    saved = json.loads(
        (tmp_path / "output/codex-cli/call-0001/result.json").read_text()
    )
    assert saved["stats"]["tool_calls"] == expected


def test_external_mode_counts_all_unique_cli_tools_without_a_toolkit(tmp_path):
    roborsi_item = {
        **_OTHER_TOOL_ITEMS[-1],
        "id": "independent-roborsi-mcp",
        "server": "roborsi",
    }
    planner, _ = make_planner(
        tmp_path,
        env={
            "STUB_TOOL_ITEMS": json.dumps(
                [_COMMAND_ITEM, *_OTHER_TOOL_ITEMS, roborsi_item]
            )
        },
    )
    result = solve(planner)
    assert result.error is None
    assert result.stats["tool_calls"] == 5
    saved = json.loads(
        (tmp_path / "output/codex-cli/call-0001/result.json").read_text()
    )
    assert saved["stats"]["tool_calls"] == 5


def test_global_cli_options_precede_exec_and_runtime_records_full_argv(tmp_path):
    planner, workspace = make_planner(
        tmp_path,
        cli_args=[
            "--ask-for-approval",
            "never",
            "--search",
            "--sandbox",
            "workspace-write",
        ],
    )
    result = solve(planner)
    assert result.error is None
    argv = json.loads((workspace / "started.json").read_text())["argv"]
    assert argv[1:5] == ["--ask-for-approval", "never", "--search", "exec"]
    assert argv.index("--sandbox") > argv.index("exec")
    runtime = json.loads(
        (tmp_path / "output/codex-cli/call-0001/runtime.json").read_text()
    )
    assert runtime["argv"] == argv


def test_preflight_environment_is_consumed_once_by_the_next_solve(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("EXAMPLE_VALUE", "first")
    planner, workspace = make_planner(
        tmp_path, env={"STUB_MODE": "text"}, env_vars=["EXAMPLE_VALUE"]
    )
    identity = planner.preflight()
    assert identity["version"] == "codex-cli test.1"
    monkeypatch.setenv("EXAMPLE_VALUE", "second")
    assert solve(planner).error is None
    assert json.loads((workspace / "started.json").read_text())["env"] == "first"
    assert solve(planner).error is None
    assert json.loads((workspace / "started.json").read_text())["env"] == "second"


@pytest.mark.parametrize(
    "mode,status,tokens",
    [
        ("missing_usage", "unavailable", None),
        ("duplicate_usage", "complete", 112),
        ("stderr", "complete", 112),
    ],
)
def test_usage_snapshots_and_stderr_are_not_lost(tmp_path, mode, status, tokens):
    planner, _ = make_planner(tmp_path, mode=mode)
    result = solve(planner)
    assert result.error is None
    assert result.stats["usage_status"] == status
    assert result.stats["total_tokens"] == tokens
    if mode == "stderr":
        assert (
            tmp_path / "output/codex-cli/call-0001/stderr.log"
        ).stat().st_size == 200000


@pytest.mark.parametrize(
    "mode,reason", [("malformed", "protocol_error"), ("failed", "turn_failed")]
)
def test_protocol_and_provider_errors_preserve_partial_evidence(tmp_path, mode, reason):
    planner, _ = make_planner(tmp_path, mode=mode)
    result = solve(planner)
    assert result.stop_reason == reason
    assert result.error
    assert (
        "thread-test"
        in (tmp_path / "output/codex-cli/call-0001/events.jsonl").read_text()
    )


@pytest.mark.parametrize("mode", ["mcp", "no_ack", "wrong_ack"])
def test_real_mcp_terminal_requires_the_current_token(tmp_path, mode):
    planner, _ = make_planner(tmp_path, mode=mode, tools="mcp")
    toolkit = RecordingToolkit()
    result = planner.solve(
        system_prompt="system", user_message="task", toolkit=toolkit, max_turns=None
    )
    assert toolkit.calls == ["move", "finish"]
    assert result.finish_result["summary"] == "native terminal"
    if mode == "mcp":
        assert result.error is None
        assert (
            '"image"'
            in (tmp_path / "output/codex-cli/call-0001/events.jsonl").read_text()
        )
    else:
        assert result.stop_reason == "terminal_feedback_missing"


def test_cancellation_releases_deferred_tool_before_waiting_for_server(tmp_path):
    planner, _ = make_planner(tmp_path, mode="blocked", tools="mcp")
    toolkit = RecordingToolkit()
    cancel = threading.Event()
    results = []
    thread = threading.Thread(
        target=lambda: results.append(
            planner.solve(
                system_prompt="",
                user_message="task",
                toolkit=toolkit,
                max_turns=None,
                cancel_event=cancel,
            )
        )
    )
    thread.start()
    assert toolkit.active.wait(10)
    cancel.set()
    thread.join(10)
    assert not thread.is_alive()
    assert toolkit.cancelled.is_set()
    assert results[0].error_kind == "cancelled"


def _running(pid):
    try:
        state = (
            Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].strip().split()[0]
        )
        return state != "Z"
    except FileNotFoundError:
        return False


@pytest.mark.parametrize("cancelled", [True, False])
def test_cancel_and_timeout_kill_grandchildren(tmp_path, cancelled):
    planner, workspace = make_planner(
        tmp_path, mode="hang", timeout_s=0 if cancelled else 3
    )
    cancel = threading.Event()
    results = []
    thread = threading.Thread(
        target=lambda: results.append(solve(planner, cancel_event=cancel))
    )
    thread.start()
    deadline = time.monotonic() + 10
    while not (workspace / "grandchild.pid").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    child = int((workspace / "grandchild.pid").read_text())
    if cancelled:
        cancel.set()
    thread.join(10)
    assert not thread.is_alive()
    assert not _running(child)
    assert results[0].stop_reason == ("closed" if cancelled else "timeout")


def test_detached_child_is_reaped_with_owned_process_tree(tmp_path):
    # The deadline includes version checking and interpreter startup. Give the
    # child time to exist under suite load before testing timeout-driven cleanup.
    planner, workspace = make_planner(tmp_path, mode="detached", timeout_s=3)
    result = solve(planner)
    child = int((workspace / "grandchild.pid").read_text())
    try:
        assert not _running(child)
        assert result.stop_reason == "timeout"
    finally:
        if _running(child):
            os.kill(child, 9)


@pytest.mark.parametrize("use_proxy", [False, True])
def test_capture_routes_real_cli_http_and_preserves_provider_bytes(
    tmp_path, monkeypatch, use_proxy
):
    body = b'{"id":"r1","status":"completed","output":[{"id":"encrypted1","type":"reasoning","encrypted_content":"opaque-value"}]}'
    seen = []

    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(
                (
                    self.path,
                    self.headers.get("Authorization"),
                    self.rfile.read(int(self.headers["Content-Length"])),
                )
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        upstream_url = f"http://127.0.0.1:{server.server_port}"
        environment = {}
        if use_proxy:
            monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
            environment = {
                "inherit_env": False,
                "env": {"STUB_MODE": "capture", "HTTP_PROXY": upstream_url},
            }
        planner, workspace = make_planner(
            tmp_path,
            mode="capture",
            provider_capture={
                "upstream_base_url": (
                    "http://provider.invalid/v1" if use_proxy else f"{upstream_url}/v1"
                ),
                "model_provider": "test",
            },
            **environment,
        )
        planner.preflight()
        result = solve(planner)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert result.error is None
    assert seen[0][0:2] == (
        "http://provider.invalid/v1/responses" if use_proxy else "/v1/responses",
        "Bearer hidden",
    )
    if use_proxy:
        assert os.environ["HTTP_PROXY"] == "http://127.0.0.1:1"
    assert (workspace / "provider-body.bin").read_bytes() == body
    capture = tmp_path / "output/codex-cli/call-0001/provider-capture"
    assert next(capture.glob("*.response.bin")).read_bytes() == body
    assert "opaque-value" in (capture / "reasoning.jsonl").read_text()
    assert "hidden" not in (capture / "000001.request.json").read_text()


def test_pending_cleanup_is_bounded_and_reports_failure(tmp_path, monkeypatch):
    import roborsi.planner.codex_cli as module

    monkeypatch.setattr(module, "_CLEANUP_TIMEOUT_S", 0.15)
    planner, _ = make_planner(tmp_path, tools="mcp")
    toolkit = RecordingToolkit()
    toolkit.cancel_pending = lambda: time.sleep(0.6)
    started = time.monotonic()
    result = planner.solve(
        system_prompt="", user_message="task", toolkit=toolkit, max_turns=None
    )
    assert time.monotonic() - started < 2
    assert result.stop_reason == "cleanup_failed"


def test_images_are_owned_files_and_unsupported_controls_fail_before_exec(tmp_path):
    planner, workspace = make_planner(tmp_path)
    result = planner.solve(
        system_prompt="system",
        user_message=["task", BinaryContent(_PNG, media_type="image/png")],
        toolkit=None,
        max_turns=None,
    )
    assert result.error is None
    argv = json.loads((workspace / "started.json").read_text())["argv"]
    assert Path(argv[argv.index("--image") + 1]).read_bytes() == _PNG
    with pytest.raises(ValueError, match="max_turns"):
        planner.solve(system_prompt="", user_message="task", toolkit=None, max_turns=1)
    with pytest.raises(ValueError, match="steer"):
        solve(planner, input_queue=queue.Queue())
    with pytest.raises(ValueError, match="toolkit"):
        planner.solve(
            system_prompt="",
            user_message="task",
            toolkit=RecordingToolkit(),
            max_turns=None,
        )


def test_corrupt_initial_image_is_rejected_before_exec(tmp_path):
    planner, workspace = make_planner(tmp_path)
    result = planner.solve(
        system_prompt="",
        user_message=[BinaryContent(b"not-an-image", media_type="image/png")],
        toolkit=None,
        max_turns=None,
    )
    assert result.error_kind == "configuration"
    assert not (workspace / "started.json").exists()
