"""Execution material selection, independent of simulator or paid provider calls."""
from importlib import import_module
import hashlib
import json
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace

import pytest

profile = import_module("robots.robodojo.agents.submission.execution_profile")
P = Path(__import__('robots.robodojo.agents.submission', fromlist=['__file__']).__file__).parent


def test_robot_namespace_preserves_recursive_bounds_and_replaces_lossy_cli_schema():
    schema = {'type': 'object', 'required': ['points'], 'additionalProperties': False,
        'properties': {'points': {'type': 'array', 'minItems': 1, 'maxItems': 12,
            'items': {'type': 'object', 'properties': {
                'step': {'type': 'integer', 'minimum': 1, 'maximum': 160},
                'roll': {'anyOf': [{'type': 'number', 'minimum': -90, 'maximum': 90}, {'type': 'null'}]}}}}}}
    underlying = SimpleNamespace(expected_tools={'localize'}, transform=lambda request: (request, {}))
    wrapped = profile.RobotCodeModeProfile(underlying, [{'name': 'localize', 'input_schema': schema}])
    request = {'input': [{'type': 'additional_tools', 'tools': [
        {'name': 'mcp__robot', 'tools': [{'name': 'localize', 'parameters': {}}]},
        {'name': 'functions', 'tools': []}]}]}
    original = deepcopy(request)
    actual, receipt = wrapped.transform(request)
    tools = actual['input'][0]['tools']
    robots = [tool for tool in tools if tool['name'] == 'mcp__robot']
    assert len(robots) == 1 and request == original
    projected = robots[0]['tools'][0]['parameters']
    assert projected['required'] == ['points'] and projected['additionalProperties'] is False
    points = projected['properties']['points']
    assert (points['minItems'], points['maxItems']) == (1, 12)
    assert projected == schema
    assert points['items']['properties']['step']['maximum'] == 160
    assert points['items']['properties']['roll']['anyOf'][0]['minimum'] == -90
    assert receipt['robot_constraint_schema_sha256'] == wrapped.schema_sha256










def test_fixed_references_reject_unregistered_or_changed_workspace_links(tmp_path):
    context=profile.prepare_skillpack_context(P,'align_blocks',tmp_path)
    derived=profile.load_skillpack_profile(P,'align_blocks',tmp_path)
    derived.validate_staged_references(context.workspace)
    link=context.workspace/'.agents/skills/robo-general/SKILL.md'
    foreign=tmp_path/'foreign.md';foreign.write_bytes(link.read_bytes())
    link.unlink();link.symlink_to(foreign)
    with pytest.raises(ValueError,match='Unregistered workspace reference'):
        derived.validate_staged_references(context.workspace)




def test_shared_bytes_and_tools_are_unchanged(tmp_path):
    context = profile.prepare_skillpack_context(P, "arrange_largest_number", tmp_path)
    derived = profile.load_skillpack_profile(P, "arrange_largest_number", tmp_path)
    common = P/"skillpack/common"
    for name in ("agent_core.md", "native_tools.json", "robot_tools.json", "actual_robot_declarations.json", "skills_instruction.md"):
        assert (derived.directory/name).read_bytes() == (common/name).read_bytes()
    general = context.workspace/".agents/skills/robo-general/SKILL.md"
    assert general.read_bytes() == (common/"skills/robo-general/SKILL.md").read_bytes()
    cfg = {"options": {"review_window_steps": 1050, "review_max_windows": 1}, "episode_seconds": 14000}
    prompt = derived.episode_prompt(cfg)
    assert "smoke" not in prompt.lower() and "three" not in prompt.lower()
    assert derived.expected_tools == {"observe", "vla_propose", "act",
                                      "hold_current", "close_in_place", "move_relative",
                                      "spatial_localize_pixel"}


def test_published_exec_description_uses_registered_host_time_command(tmp_path):
    derived=profile.load_skillpack_profile(P,'align_blocks',tmp_path)
    group=next(g for g in derived.tools if g['name']=='functions')
    description=next(t['description']for t in group['tools']if t['name']=='exec')
    assert 'clock__curr_time' not in description
    assert 'date -u' in description and 'exec_command' in description


@pytest.mark.parametrize("suffix,media_type", [("jpg", "image/jpeg"), ("jpeg", "image/jpeg"), ("png", "image/png")])
def test_observe_attaches_original_image_bytes_with_matching_mime(tmp_path, suffix, media_type):
    import base64
    from PIL import Image

    runtime = profile.runtime_modules(P)
    cameras = {}
    for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
        name = f"00000_{camera}.{suffix}"
        Image.new("RGB", (13, 9), (240, 20, 10)).save(tmp_path/name)
        cameras[camera] = "/observations/"+name
    native = {"observation": {"cameras": cameras, "control_step": 0}, "done": False}
    tools = runtime.robot_tools.RobotTools(lambda request: native, setting="simate_vla", image_root=tmp_path)
    result = tools.execute_tool("observe", {})
    images = [block["source"] for block in result.content_blocks if block["type"] == "image"]
    assert len(images) == 3
    for camera, encoded in zip(cameras, images):
        assert encoded["media_type"] == media_type
        assert base64.b64decode(encoded["data"]) == (tmp_path/f"00000_{camera}.{suffix}").read_bytes()
    assert native["observation"]["cameras"] == cameras


def test_unknown_task_and_cross_task_profile_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown public"):
        profile.load_skillpack_profile(P, "made_up_task", tmp_path)
    profile.load_skillpack_profile(P, "arrange_largest_number", tmp_path)
    with pytest.raises(ValueError, match="another task"):
        profile.load_skillpack_profile(P, "stack_bowls", tmp_path)


def test_cci_reasoning_null_content_does_not_rewrite_visible_history():
    raw = {"input": [{"type": "reasoning", "content": None, "encrypted_content": "opaque"},
                     {"role": "assistant", "content": [{"type": "output_text", "text": "visible"}]}]}
    frozen = SimpleNamespace(transform=lambda value: (deepcopy(value), {}))
    result, evidence = profile.CciPromptProfile(frozen).transform(raw)
    assert raw["input"][0]["content"] is None
    assert result["input"][0]["content"] == []
    assert result["input"][0]["encrypted_content"] == "opaque"
    assert result["input"][1] == raw["input"][1]
    assert evidence["cci_null_content_as_empty_array"] == [0]


@pytest.mark.parametrize("episode_seconds", [None, 14400, 86400])
def test_cci_attempt_timeout_is_not_replaced_by_four_hour_episode(tmp_path, monkeypatch, episode_seconds):
    cci = import_module("robots.robodojo.agents.submission.cci_protocol")
    guard = import_module("robots.robodojo.agents.submission.output_retry")
    seen = {}

    class Adapter:
        def __init__(self, endpoint, model, directory, request_timeout=300, sampling=None, enable_thinking=True, save_raw=True, reasoning_directory=None):
            seen["attempt_timeout"] = request_timeout
            seen["sampling"] = sampling
            seen['save_raw'] = save_raw
            self.base_url = "http://127.0.0.1:12345/v1"
        def start(self): pass
        def stop(self): pass

    class Capture:
        def __init__(self, *args, **kwargs):
            seen["deadline"] = kwargs["case_deadline"]
            seen["expected_effort"] = kwargs["expected_effort"]
            seen['storage_mode'] = kwargs['storage_mode']
        def start(self): pass
        def stop(self): pass

    monkeypatch.setattr(cci, "CciAdapter", Adapter)
    monkeypatch.setattr(guard, "GuardedCapture", Capture)
    monkeypatch.setenv("ROBORSI_CODEX_MODEL", "test-cci")
    monkeypatch.setenv("ROBORSI_CODEX_BASE_URL", "http://127.0.0.1:819/v1")
    sampling = {"temperature": 1.0, "top_k": 20, "top_p": 0.95}
    capture = profile.start_capture(None, tmp_path/"controller", {
        "provider_kind": "cci", "episode_seconds": episode_seconds, "case_wall_deadline": 20000,
        "reasoning_effort": "medium", "sampling": sampling})
    assert seen == {"attempt_timeout": 300, "deadline": 20000,
                    "expected_effort": "medium", "sampling": sampling,
                    'storage_mode':'training', 'save_raw':False}
    capture.stop()


def test_medium_catalog_does_not_fall_back_to_xhigh(tmp_path):
    (tmp_path/"preparation").mkdir()
    path = profile.stage_cci_catalog(tmp_path, "fixture", reasoning_effort="medium")
    model = json.loads(path.read_text())["models"][0]
    assert model["default_reasoning_level"] == "medium"
    assert model["slug"] == "fixture"


def test_native_robot_namespace_is_preserved_with_exec_wait(tmp_path):
    derived = profile.load_skillpack_profile(P, "arrange_largest_number", tmp_path)
    robot = {"type": "namespace", "name": "mcp__robot", "tools": [
        {"type": "function", "name": name, "parameters": {"type": "object"}}
        for name in sorted(derived.expected_tools)]}
    result, evidence = derived.transform({"input": [
        {"type": "additional_tools", "tools": [robot]},
        {"role": "developer", "content": [{"type": "input_text", "text": derived.base_rules}]},
        {"role": "developer", "content": [{"type": "input_text", "text": derived.core}]},
        {"role": "user", "content": [{"type": "input_text", "text": "episode"}]}]})
    groups = result["input"][0]["tools"]
    assert next(g for g in groups if g["name"] == "mcp__robot") == robot
    assert {t["name"] for g in groups if g["name"] == "functions" for t in g["tools"]} == {"exec", "wait"}
    assert "automatic text and image" in evidence["robot_binding"]
    assert "mcp__robot__observe" not in groups[0]["tools"][0]["description"]
    with pytest.raises(ValueError, match="robot declarations missing"):
        derived.transform({"input": [{"type": "additional_tools", "tools": []},
            {"role": "developer", "content": [{"type": "input_text", "text": derived.core}]}]})
