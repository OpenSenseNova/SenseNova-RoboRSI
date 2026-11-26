"""Offline checks of the unique skillpack owner and scoped launch hooks."""
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
import ast
import hashlib
import json
import threading

import pytest
import yaml

skillpack = import_module("robots.robodojo.agents.submission.skillpack_agent")
factory = import_module("robots.robodojo.agents.submission.sense_nova_roborsi_factory")
execution_profile = import_module("robots.robodojo.agents.submission.execution_profile")
EmbodiedAgent = import_module("roborsi.embodied_agent").EmbodiedAgent
P = Path(__import__('robots.robodojo.agents.submission', fromlist=['__file__']).__file__).parent


@pytest.mark.parametrize("effort", ["medium", "xhigh"])
def test_unique_factory_is_real_embodied_agent_with_selected_effort(tmp_path, effort):
    document = yaml.safe_load((P/"agent.yaml").read_text())
    document["agent"]["model"] = "fixture-model"
    document["agent"]["reasoning_effort"] = effort
    agent = factory.make_agent(tmp_path, config=SimpleNamespace(document=document))
    assert isinstance(agent, EmbodiedAgent)
    assert isinstance(agent, skillpack.SkillpackEmbodiedAgent)
    assert agent.reasoning_effort == effort
    assert agent.codex_cli.config_overrides["model_auto_compact_token_limit"] == 200000
    assert agent.max_tokens is None


def test_old_prompt_and_factory_are_removed_and_base_cannot_launch():
    hybrid = import_module("robots.robodojo.agents.submission.hybrid_agent")
    assert not hasattr(hybrid, "build_evaluation_prompt")
    assert not hasattr(hybrid, "make_agent")
    agent = object.__new__(hybrid.HybridEmbodiedAgent)
    with pytest.raises(NotImplementedError, match="skillpack execution profile"):
        agent.evaluation_prompt(None, {})


def test_derived_controller_keeps_native_records_and_recovery_ast():
    root = P/"vendor/robodojo_adaptive"
    manifest = json.loads((root/"provenance/source-files.json").read_text())
    change = manifest["skillpack_prompt_cleanup"]
    tree = ast.parse((root/"source/code/policylab/runner.py").read_text())
    # Python 3.12 adds empty type_params fields; pins use Python 3.10 AST dumps.
    for node in ast.walk(tree):
        if hasattr(node, "type_params") and not node.type_params:
            node._fields = tuple(name for name in node._fields if name != "type_params")
    definitions = {n.name: n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    assert "run_experiment" not in definitions
    assert "INSTRUCTIONS" not in (root/"source/code/policylab/runner.py").read_text()
    for name, expected in change["preserved_controller_ast_sha256"].items():
        actual = hashlib.sha256(ast.dump(definitions[name], include_attributes=False).encode()).hexdigest()
        assert actual == expected, name
    pins = json.loads((P/"submission.json").read_text())
    assert pins["hybrid"]["manifest_sha256"] == hashlib.sha256((root/"provenance/source-files.json").read_bytes()).hexdigest()
    for row in change["patches"]:
        current = root/row["path"]
        assert hashlib.sha256(current.read_bytes()).hexdigest() == row["after_sha256"]


def test_scoped_experiment_rejects_python_socket_bypass(monkeypatch, tmp_path):
    calls = []

    class InvalidAction(Exception):
        pass

    class Experiment:
        def __init__(self, cfg, run, backend):
            self.cfg, self.run, self.backend = cfg, run, backend

    def original(self, req):
        calls.append((self, req))
        return {"observation": {"control_step": 0}}

    # Use the frozen make_request with a tiny owner, not the old CaseRunner.
    runtime = execution_profile.runtime_modules(P)
    reference = SimpleNamespace(runner=SimpleNamespace(Experiment=Experiment, InvalidAction=InvalidAction),
                                hybrid_host=SimpleNamespace(request=original, ORIGINAL=original))
    monkeypatch.setattr(execution_profile, "runtime_modules", lambda _: runtime)
    agent = object.__new__(skillpack.SkillpackEmbodiedAgent)
    context = SimpleNamespace(reference=reference, backend=object())
    episode = agent.make_experiment(context, {}, tmp_path)
    with pytest.raises(InvalidAction, match="registered robot MCP"):
        episode.request({"op": "observe"})
    agent._skillpack_toolkit = SimpleNamespace(operation_thread=threading.get_ident())
    result = episode.request({"op": "observe"})
    assert result["manual_tcp_targets_enabled"] is True
    assert len(calls) == 1
    assert reference.runner.Experiment is Experiment
    assert not hasattr(Experiment, "request")


def test_prompt_and_staging_do_not_use_original_evaluation_text(tmp_path):
    run = tmp_path/"controller"
    (run/"tools").mkdir(parents=True)
    (run/"tools/robot.py").write_text("legacy")
    staged = []
    profile = SimpleNamespace(manifest={}, stage_live_workspace=lambda *a, **kw: staged.append((a, kw)),
                              episode_prompt=lambda cfg: "current episode only")
    context = SimpleNamespace(execution_profile=profile, episode_dir=tmp_path,
                              prepared_workspace=tmp_path, model_cfg={})
    agent = object.__new__(skillpack.SkillpackEmbodiedAgent)
    agent.stage_workspace(context, tmp_path, run)
    assert not (run/"tools/robot.py").exists()
    assert staged and agent.evaluation_prompt(context, {}) == "current episode only"




def test_controller_cleanup_is_bounded(monkeypatch):
    calls = []
    monkeypatch.setattr(skillpack.HybridController, "cancel", lambda self, timeout: calls.append(timeout))
    controller = object.__new__(skillpack.SkillpackController)
    controller.close()
    assert calls == [15.0]


def test_failed_capture_stop_is_retried_with_the_same_owner(monkeypatch, tmp_path):
    monkeypatch.setattr(skillpack.HybridCodexCliPlanner, 'cancel', lambda self: None)
    calls = []

    class Capture:
        def stop(self):
            calls.append('capture')
            if calls.count('capture') == 1:
                raise RuntimeError('Synthetic writer is still flushing')

    capture = Capture()
    planner = object.__new__(skillpack.SkillpackCodexCliPlanner)
    planner._resource_lock = threading.RLock()
    planner.context = SimpleNamespace(episode_dir=tmp_path)
    planner._capture = capture
    planner._robot_server = SimpleNamespace(stop=lambda: calls.append('mcp'))
    planner.cancel()
    assert planner._capture is capture
    assert planner._robot_server is None
    planner.cancel()
    assert calls == ['mcp', 'capture', 'capture']
    assert planner._capture is None


def test_launch_binds_mcp_capture_and_short_prompt_on_instance(monkeypatch, tmp_path):
    @dataclass
    class Config:
        config_overrides: dict
        env: dict
        env_vars: tuple = ()

    server = SimpleNamespace(url="http://127.0.0.1:12345/mcp/", start=lambda: None)
    capture = SimpleNamespace(base_url="http://127.0.0.1:12346/v1")
    monkeypatch.setattr(execution_profile, "start_capture", lambda profile, run, cfg: capture)
    planner = object.__new__(skillpack.SkillpackCodexCliPlanner)
    planner._resource_lock = threading.RLock()
    planner._cancelled = threading.Event()
    planner._robot_server = planner._capture = None
    planner.profile = SimpleNamespace(manifest={}, profile_id="frozen", task="arrange_largest_number",
                                      expected_tools={"observe", "vla_propose", "act"},
                                      episode_prompt=lambda cfg: "Carry out the current episode task.")
    planner.context = SimpleNamespace(model_cfg={"smoke_act_limit": 3})
    planner.toolkit = SimpleNamespace(mcp_server=lambda: server,
                                     get_tools_spec=lambda: [{"name": n, "input_schema": {"type": "object"}}
                                                            for n in ["observe", "vla_propose", "act"]])
    planner.config = Config({"model_providers.runtime.base_url": "https://upstream.invalid/v1"}, {})
    previous = planner.config
    seen = []
    def launch(self, cfg, run, home, segment, prompt, session):
        seen.append((self.config, prompt))
        return "owned-process"
    monkeypatch.setattr(skillpack.HybridCodexCliPlanner, "launch_segment", launch)
    prompt = execution_profile.episode_prompt(planner.profile, {})
    assert planner.launch_segment({}, tmp_path, tmp_path, tmp_path/"segment", prompt, None) == "owned-process"
    assert planner.config is previous
    current, prompt = seen[0]
    assert current.config_overrides["model_providers.runtime.base_url"] == capture.base_url
    assert current.config_overrides["mcp_servers.robot.url"] == server.url
    assert current.config_overrides["mcp_servers.robot.required"] is True
    assert current.config_overrides["mcp_servers.robot.omit_tools_from"] == []
    assert current.config_overrides["features.code_mode"] is True
    assert current.config_overrides["features.code_mode_only"] is False
    assert current.config_overrides["model_context_window"] == 200000
    assert "3" not in prompt and "obsolete" not in prompt
