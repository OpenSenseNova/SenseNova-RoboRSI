# Copyright 2026 The RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI presets preserve explicit execution identity and budget semantics."""

import json
import sys

import pytest

from roborsi.agents.config import AgentConfig


def _preset(tmp_path, **options):
    path = tmp_path / "agent.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "agent": {
                    "planner": "codex_cli",
                    "codex_cli": {"expected_version": "codex-cli 0.154.0"},
                    **options,
                },
            }
        )
    )
    return path


def test_cli_preset_normalizes_unavailable_budgets_and_constructs_agent(tmp_path):
    config = AgentConfig.from_file(
        _preset(tmp_path, planner_timeout_s=0, action_timeout_s=None)
    )
    agent = config.embodied_agent(tmp_path)
    assert agent.planner == "codex_cli"
    assert agent.max_turns is None
    assert agent.max_tokens is None
    assert agent.planner_timeout_s == 0
    assert agent.action_timeout_s is None
    assert agent.codex_cli.expected_version == "codex-cli 0.154.0"
    assert config.document["agent"]["max_turns"] is None


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"max_turns": 1}, "max_turns"),
        ({"max_tokens": 32}, "max_tokens"),
        ({"planner_timeout_s": -1}, "planner_timeout_s"),
        ({"codex_cli": {}}, "expected_version"),
        ({"reasoning_effort": "invented"}, "reasoning_effort"),
        ({"reasoning_effort": []}, "reasoning_effort"),
        ({"codex_cli": {"expected_version": "v", "tools": "external"}}, "custom"),
        ({"planner": "api"}, "codex_cli"),
        ({"planner": "codex"}, "codex_cli"),
    ],
)
def test_cli_preset_rejects_unsupported_settings_at_load(tmp_path, options, message):
    with pytest.raises(ValueError, match=message):
        AgentConfig.from_file(_preset(tmp_path, **options))


def test_working_directory_stays_relative_to_declaring_preset(tmp_path):
    parent = tmp_path / "baseline"
    parent.mkdir()
    (parent / "workspace").mkdir()
    source = _preset(parent, working_dir="workspace", reasoning_effort="max")
    data = json.loads(source.read_text())
    data["source_root"] = ".."
    source.write_text(json.dumps(data))
    child = tmp_path / "variant.json"
    child.write_text(
        json.dumps({"schema_version": 1, "extends": "baseline/agent.json"})
    )
    config = AgentConfig.from_file(child)
    agent = config.embodied_agent(tmp_path / "outputs")
    assert agent.working_dir == parent / "workspace"
    assert agent.output_dir == tmp_path / "outputs"
    assert agent.reasoning_effort == "max"
    # A factory can create a case-specific writable copy of the template.
    instance = config.embodied_agent(
        tmp_path / "outputs", working_dir=tmp_path / "case"
    )
    assert instance.working_dir == tmp_path / "case"


def test_cli_config_uses_named_secrets_without_serializing_values(monkeypatch):
    from roborsi.planner.codex_cli_config import CodexCliConfig

    monkeypatch.setenv("CLI_TEST_CREDENTIAL", "secret-value")
    monkeypatch.setenv("CLI_UNRELATED", "unselected")
    raw = {
        "expected_version": "codex-cli 0.154.0",
        "inherit_env": False,
        "env_vars": ["CLI_TEST_CREDENTIAL"],
        "env": {"CODEX_HOME": "/workspace/home"},
        "config_overrides": {"features.example": False},
    }
    config = CodexCliConfig.from_dict(raw)
    raw["env"]["CODEX_HOME"] = "changed"
    assert config.subprocess_env() == {
        "CLI_TEST_CREDENTIAL": "secret-value",
        "CODEX_HOME": "/workspace/home",
    }
    assert "secret-value" not in repr(config)
    with pytest.raises(TypeError):
        config.env["CODEX_HOME"] = "changed"


@pytest.mark.parametrize(
    "patch",
    [
        {"expected_version": ""},
        {"expected_sha256": "not-a-hash"},
        {"executable": ""},
        {"cli_args": ["--json"]},
        {"cli_args": ["--color", "always"]},
        {"cli_args": ["-m", "other-model"]},
        {"cli_args": ["--config=model='other'"]},
        {"cli_args": ["resume"]},
        {"config_overrides": {"mcp_servers.roborsi.url": "other"}},
        {"config_overrides": {"model": "other"}},
        {"config_overrides": {"model_providers.custom.query_params.token": "secret"}},
        {
            "config_overrides": {
                "model_providers.custom.query_params.access_key": "secret"
            }
        },
        {"config_overrides": {"model_providers.custom.query_params.bearer": "secret"}},
        {"config_overrides": {"nested": {"value": 1}}},
        {"env": {"OPENAI_API_KEY": "secret"}},
        {"env_vars": ["A=B"]},
        {"inherit_env": "false"},
        {"unknown": True},
    ],
)
def test_cli_config_rejects_conflicting_or_unserializable_inputs(patch):
    from roborsi.planner.codex_cli_config import CodexCliConfig

    with pytest.raises(ValueError):
        CodexCliConfig.from_dict({"expected_version": "codex-cli 0.154.0", **patch})


def test_cli_preset_preserves_explicit_simulation_action_approval(tmp_path):
    key = "mcp_servers.roborsi.tools.roborsi_action.approval_mode"
    config = AgentConfig.from_file(
        _preset(
            tmp_path,
            codex_cli={
                "expected_version": "codex-cli 0.154.0",
                "tools": "mcp",
                "config_overrides": {"approval_policy": "never", key: "approve"},
            },
        )
    )
    agent = config.embodied_agent(tmp_path / "outputs")
    assert agent.codex_cli.config_overrides[key] == "approve"
    assert agent.codex_cli.config_overrides["approval_policy"] == "never"


@pytest.mark.parametrize(
    ("tools", "key", "value"),
    [
        (
            "external",
            "mcp_servers.roborsi.tools.roborsi_action.approval_mode",
            "approve",
        ),
        ("mcp", "mcp_servers.roborsi.tools.roborsi_action.approval_mode", "deny"),
        ("mcp", "mcp_servers.roborsi.tools.roborsi_action.approval_mode", True),
        ("mcp", "mcp_servers.roborsi.tools.other_action.approval_mode", "approve"),
        ("mcp", "mcp_servers.roborsi.tools.roborsi_action.enabled", True),
        ("mcp", "mcp_servers.roborsi.url", "http://localhost:8000"),
    ],
)
def test_cli_config_keeps_simulation_approval_exception_narrow(tools, key, value):
    from roborsi.planner.codex_cli_config import CodexCliConfig

    with pytest.raises(ValueError, match="reserved codex_cli.config_overrides key"):
        CodexCliConfig.from_dict(
            {
                "expected_version": "codex-cli 0.154.0",
                "tools": tools,
                "config_overrides": {key: value},
            }
        )


@pytest.mark.parametrize("planner", ["api", "codex", "claude_code"])
def test_unlimited_action_wait_remains_cli_only(tmp_path, planner):
    path = tmp_path / "agent.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "agent": {"planner": planner, "action_timeout_s": None},
            }
        )
    )
    with pytest.raises(ValueError, match="action_timeout_s"):
        AgentConfig.from_file(path)


def test_python_cli_agent_does_not_silently_ignore_default_budget(tmp_path):
    from roborsi.embodied_agent import EmbodiedAgent, LocalToolSpec
    from roborsi.planner.codex_cli_config import CodexCliConfig

    agent = EmbodiedAgent(
        mcp_servers=[],
        output_dir=tmp_path,
        planner="codex_cli",
        codex_cli=CodexCliConfig(expected_version="v"),
        local_tools=[
            LocalToolSpec("observe", "Observe", {"type": "object"}, lambda _: {})
        ],
    )
    with pytest.raises(ValueError, match="max_turns"):
        agent.run("Task", system_prompt="Rules")


def test_cli_result_retains_unknown_usage_and_working_directory(tmp_path, monkeypatch):
    from roborsi import embodied_agent as module
    from roborsi.embodied_agent import LocalToolSpec
    from roborsi.planner.base import PlannerResult

    workdir = tmp_path / "workspace"
    workdir.mkdir()
    config = AgentConfig.from_file(_preset(tmp_path, working_dir="workspace"))

    class Planner:
        def preflight(self):
            pass

        def solve(self, **kwargs):
            assert kwargs["max_turns"] is None
            return PlannerResult(stats={"usage_status": "unavailable"})

    def construct(backend, **kwargs):
        assert backend == "codex_cli"
        assert kwargs["working_dir"] == workdir
        assert kwargs["codex_cli"].expected_version == "codex-cli 0.154.0"
        return Planner()

    monkeypatch.setattr(module, "build_planner", construct)
    agent = config.embodied_agent(
        tmp_path / "logs",
        local_tools=[
            LocalToolSpec("observe", "Observe", {"type": "object"}, lambda _: {})
        ],
    )
    result = agent.run("Task", system_prompt="Rules")
    assert result.stats["llm_usage"]["status"] == "unavailable"
    assert result.stats["llm_usage"]["input_tokens"] is None
    assert result.stats["llm_usage"]["total_tokens"] is None


def test_cli_action_wait_can_be_unlimited_while_cleanup_is_bounded(tmp_path):
    from types import SimpleNamespace

    from roborsi.evaluation.embodied_bridge import (
        _EmbodiedToolConnector,
        roborsi_action_tool,
    )
    from roborsi.evaluation.messages import Observation, StepResult, Task
    from roborsi.planner.base import PlannerResult

    timeouts = []

    class Episode:
        def next_call(self, *, timeout):
            timeouts.append(("action", timeout))
            return SimpleNamespace(name="roborsi_action", arguments={})

        def complete(self, call, reply):
            pass

        def wait(self, *, timeout):
            timeouts.append(("cleanup", timeout))
            return PlannerResult()

        def close(self, **kwargs):
            timeouts.append(("close", kwargs.get("timeout")))

    schema = {"type": "object"}
    planner = SimpleNamespace(
        local_tools=[roborsi_action_tool(schema)],
        start_episode=lambda *args, **kwargs: Episode(),
    )
    connector = _EmbodiedToolConnector(
        planner,
        media_dir=tmp_path / "media",
        system_prompt="Rules",
        output_dir=tmp_path / "logs",
        action_timeout_s=None,
        cleanup_timeout_s=30,
    )
    connector.reset(
        Task(
            benchmark="test",
            task_id="0",
            seed=0,
            instruction="Task",
            max_steps=1,
            capabilities={"action_schema": schema},
        )
    )
    connector.act(Observation(benchmark="test", episode_id="episode", step_index=0))
    connector.observe_end(
        StepResult(
            observation=Observation(
                benchmark="test", episode_id="episode", step_index=1
            ),
            terminated=True,
            truncated=False,
            termination_reason="finished",
        )
    )
    assert timeouts == [("action", None), ("cleanup", 30)]
    connector.close()
    assert timeouts[-1] == ("close", 30)


def test_partial_cli_usage_does_not_turn_missing_fields_into_zero(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from roborsi import embodied_agent as module
    from roborsi.embodied_agent import LocalToolSpec
    from roborsi.planner.base import PlannerResult

    monkeypatch.setattr(
        module,
        "build_planner",
        lambda *a, **kw: SimpleNamespace(
            preflight=lambda: None,
            solve=lambda **kw: PlannerResult(
                stats={
                    "usage_status": "partial",
                    "total_input_tokens": 20,
                    "total_output_tokens": None,
                    "total_cached_input_tokens": None,
                    "total_reasoning_output_tokens": None,
                }
            ),
        ),
    )
    config = AgentConfig.from_file(_preset(tmp_path))
    agent = config.embodied_agent(
        tmp_path,
        local_tools=[
            LocalToolSpec("observe", "Observe", {"type": "object"}, lambda _: {})
        ],
    )
    usage = agent.run("Task", system_prompt="Rules").stats["llm_usage"]
    assert usage["status"] == "partial"
    assert usage["input_tokens"] == 20
    assert usage["output_tokens"] is None
    assert usage["total_tokens"] is None
    assert usage["cache_read_tokens"] is None


def test_provider_capture_preserves_selected_provider_and_endpoint():
    from roborsi.planner.codex_cli_config import CodexCliConfig

    capture = {
        "model_provider": "local",
        "upstream_base_url": "https://example.test/v1",
    }
    for overrides in (
        {"model_provider": "different"},
        {"model_providers.local.base_url": "https://different.test/v1"},
    ):
        with pytest.raises(ValueError, match="provider_capture"):
            CodexCliConfig.from_dict(
                {
                    "expected_version": "v",
                    "provider_capture": capture,
                    "config_overrides": overrides,
                }
            )


def test_provider_credentials_use_environment_references():
    from roborsi.planner.codex_cli_config import CodexCliConfig

    config = CodexCliConfig.from_dict(
        {
            "expected_version": "v",
            "config_overrides": {
                "model_providers.custom.env_key": "CUSTOM_API_KEY",
                "model_providers.custom.env_http_headers.Authorization": "CUSTOM_AUTH",
                "model_providers.custom.query_params.api-version": "2026-01-01",
            },
        }
    )
    assert (
        config.config_overrides["model_providers.custom.env_http_headers.Authorization"]
        == "CUSTOM_AUTH"
    )
    assert (
        config.config_overrides["model_providers.custom.query_params.api-version"]
        == "2026-01-01"
    )


def test_build_cli_planner_uses_cli_timeout_fallback_and_explicit_workdir(
    tmp_path, monkeypatch
):
    from roborsi.dashboard.events import NullDashboardEventSink
    from roborsi.planner.base import build_planner
    from roborsi.planner.codex_cli_config import CodexCliConfig

    monkeypatch.setenv("CODEX_TIMEOUT_S", "37")
    planner = build_planner(
        "codex_cli",
        output_dir=tmp_path,
        recipe_tag="case",
        robot_name="robot",
        model="model",
        max_tokens=None,
        reasoning_effort="max",
        working_dir=tmp_path / "work",
        codex_cli=CodexCliConfig(expected_version="v"),
        dashboard_events=NullDashboardEventSink(),
    )
    assert planner.timeout_s == 37
    assert planner.repo_root == tmp_path / "work"


@pytest.mark.parametrize("planner", ["codex", "codex_cli"])
def test_provider_capture_is_shared_by_codex_backends(tmp_path, planner):
    from roborsi.planner.provider_capture import ProviderCaptureConfig

    options = {
        "planner": planner,
        "provider_capture": {
            "upstream_base_url": "https://example.test/v1",
            "model_provider": "custom",
        },
    }
    if planner == "codex_cli":
        options["codex_cli"] = {"expected_version": "v"}
    path = tmp_path / "agent.json"
    path.write_text(json.dumps({"schema_version": 1, "agent": options}))
    agent = AgentConfig.from_file(path).embodied_agent(tmp_path)
    assert isinstance(agent.provider_capture, ProviderCaptureConfig)
    assert agent.provider_capture.model_provider == "custom"


@pytest.mark.parametrize("planner", ["api", "claude_code"])
def test_provider_capture_rejects_unsupported_backends(tmp_path, planner):
    path = tmp_path / "agent.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "agent": {
                    "planner": planner,
                    "provider_capture": {
                        "upstream_base_url": "https://example.test/v1",
                        "model_provider": "custom",
                    },
                },
            }
        )
    )
    with pytest.raises(ValueError, match="provider_capture.*codex"):
        AgentConfig.from_file(path)


def test_cli_capture_rejects_two_configuration_sources(tmp_path):
    capture = {
        "upstream_base_url": "https://example.test/v1",
        "model_provider": "custom",
    }
    with pytest.raises(
        ValueError, match="provider_capture.*both|both.*provider_capture"
    ):
        AgentConfig.from_file(
            _preset(
                tmp_path,
                provider_capture=capture,
                codex_cli={
                    "expected_version": "v",
                    "provider_capture": capture,
                },
            )
        )


@pytest.mark.skipif(
    sys.platform != "linux", reason="CLI subprocess lifecycle targets Linux"
)
def test_cli_preflight_runs_before_external_tool_discovery(tmp_path, monkeypatch):
    from roborsi import embodied_agent as module
    from roborsi.embodied_agent import McpServer

    def unexpected_start(toolkit):
        pytest.fail("external MCP discovery started before the version check")

    executable = tmp_path / "codex-stub"
    executable.write_text(f"#!{sys.executable}\nprint('codex-cli wrong-version')\n")
    executable.chmod(0o755)
    monkeypatch.setattr(module._McpToolkit, "start", unexpected_start)
    agent = AgentConfig.from_file(
        _preset(
            tmp_path,
            codex_cli={
                "executable": str(executable),
                "expected_version": "codex-cli expected-version",
            },
        )
    ).embodied_agent(
        tmp_path,
        mcp_servers=[McpServer(name="external", url="http://127.0.0.1:1/mcp")],
    )
    with pytest.raises(ValueError, match="version mismatch"):
        agent.run("Task", system_prompt="Rules")


def test_shared_capture_reaches_cli_runtime_config(tmp_path):
    from roborsi.dashboard.events import NullDashboardEventSink
    from roborsi.planner.base import build_planner
    from roborsi.planner.codex_cli_config import CodexCliConfig
    from roborsi.planner.provider_capture import ProviderCaptureConfig

    capture = ProviderCaptureConfig(
        upstream_base_url="https://example.test/v1", model_provider="custom"
    )
    planner = build_planner(
        "codex_cli",
        output_dir=tmp_path,
        recipe_tag="case",
        robot_name="robot",
        model="model",
        max_tokens=None,
        working_dir=tmp_path,
        codex_cli=CodexCliConfig(expected_version="v"),
        provider_capture=capture,
        dashboard_events=NullDashboardEventSink(),
    )
    assert planner.config.provider_capture is capture


@pytest.mark.parametrize(
    "provider",
    ["openai", "ollama", "lmstudio", "amazon-bedrock", "amazon-bedrock-runtime"],
)
def test_capture_requires_custom_provider_without_restricting_ordinary_cli(provider):
    from roborsi.planner.codex_cli_config import CodexCliConfig

    ordinary = CodexCliConfig.from_dict(
        {
            "expected_version": "v",
            "config_overrides": {"model_provider": provider},
        }
    )
    assert ordinary.config_overrides["model_provider"] == provider
    with pytest.raises(ValueError, match="custom.*provider|provider.*custom"):
        CodexCliConfig.from_dict(
            {
                "expected_version": "v",
                "provider_capture": {
                    "model_provider": provider,
                    "upstream_base_url": "https://example.test/v1",
                },
            }
        )


@pytest.mark.parametrize("planner", ["codex", "codex_cli"])
@pytest.mark.parametrize(
    "provider",
    ["openai", "ollama", "lmstudio", "amazon-bedrock", "amazon-bedrock-runtime"],
)
def test_shared_capture_rejects_builtin_provider_during_preset_load(
    tmp_path, planner, provider
):
    options = {
        "planner": planner,
        "provider_capture": {
            "upstream_base_url": "https://example.test/v1",
            "model_provider": provider,
        },
    }
    if planner == "codex_cli":
        options["codex_cli"] = {"expected_version": "v"}
    path = tmp_path / "agent.json"
    path.write_text(json.dumps({"schema_version": 1, "agent": options}))
    with pytest.raises(ValueError, match="custom.*provider|provider.*custom"):
        AgentConfig.from_file(path)


@pytest.mark.parametrize("planner_type", ["codex", "codex_cli"])
def test_shared_python_capture_rejects_builtin_provider_before_construction(
    tmp_path, planner_type
):
    from roborsi.dashboard.events import NullDashboardEventSink
    from roborsi.planner.base import build_planner
    from roborsi.planner.codex_cli_config import CodexCliConfig
    from roborsi.planner.provider_capture import ProviderCaptureConfig

    capture = ProviderCaptureConfig(
        upstream_base_url="https://example.test/v1", model_provider="openai"
    )
    with pytest.raises(ValueError, match="custom.*provider|provider.*custom"):
        build_planner(
            planner_type,
            output_dir=tmp_path,
            recipe_tag="case",
            robot_name="robot",
            model="model",
            max_tokens=None,
            provider_capture=capture,
            dashboard_events=NullDashboardEventSink(),
            **(
                {"codex_cli": CodexCliConfig(expected_version="v")}
                if planner_type == "codex_cli"
                else {}
            ),
        )
