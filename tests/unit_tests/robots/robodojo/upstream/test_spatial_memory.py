"""Task isolation, recovery and RGB geometry-to-memory integration, offline."""
from importlib import import_module, util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

execution = import_module("robots.robodojo.agents.submission.execution_profile")
POLICY = Path(__import__('robots.robodojo.agents.submission', fromlist=['__file__']).__file__).parent
SKILL = POLICY / "skillpack/common/skills/robodojo-spatial-localize"
spec = util.spec_from_file_location("spatial_memory_helper", SKILL / "scripts/space_memory.py")
memory_helper = util.module_from_spec(spec)
spec.loader.exec_module(memory_helper)






def test_geometry_rgb_cli_import_and_invalidate(tmp_path):
    from PIL import Image
    Image.new("RGB", (100, 100), (100, 20, 30)).save(tmp_path / "rgb.png")
    case = {"frame": "environment_origin", "units": "meters", "cameras": {
        "head": {"image": "rgb.png", "K": [[100, 0, 50], [0, 100, 50], [0, 0, 1]],
                 "T_base_camera_cv": [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 1], [0, 0, 0, 1]]}},
        "support_planes": {"table": {"frame": "environment_origin", "normal": [0, 0, 1],
            "offset_m": 0, "height_sigma_m": 0, "source": "synthetic known plane"}}}
    targets = {"targets": [{"id": "contact", "object_id": "red_block", "description": "visible table contact",
        "kind": "support_plane", "support_plane": "table", "height_above_plane_m": 0,
        "observations": {"head": {"points_px": [[50, 50]]}},
        "plane_options": {"max_position_sigma_m": 0.02}}]}
    for name, data in [("case.json", case), ("targets.json", targets)]:
        (tmp_path / name).write_text(json.dumps(data))
    subprocess.run([sys.executable, "-B", str(SKILL / "scripts/localize_rgb.py"), "--case", str(tmp_path / "case.json"),
                    "--targets", str(tmp_path / "targets.json"), "--output", str(tmp_path / "geometry")], check=True)
    result = json.loads((tmp_path / "geometry/points3d.json").read_text())
    assert result["points"][0]["estimate"]["accepted"] is True
    state = {"schema_version": 1, "task": "classify_objects", "frame": "environment_origin",
             "units": "meters", "objects": {"unrelated": {}}}
    memory_helper.import_result(state, result, evidence="geometry/points3d.json", step=42)
    point = state["objects"]["red_block"]["contact"]
    assert point["position_m"] == pytest.approx([0, 0, 0])
    assert point["status"] == "localized" and point["control_step"] == 42
    assert "unrelated" in state["objects"]
    memory = tmp_path / "space.json"
    memory_helper.save(memory, state)
    subprocess.run([sys.executable, str(SKILL / "scripts/space_memory.py"), "stale", "--memory", str(memory),
                    "--object", "red_block"], check=True)
    assert memory_helper.load(memory)["objects"]["red_block"]["contact"]["status"] == "stale"
    result["frame"] = "other_robot_base"
    with pytest.raises(ValueError, match="frame/units"):
        memory_helper.import_result(state, result, evidence="wrong.json", step=43)


def test_rejected_estimate_never_promotes_a_position():
    state = {"frame": "environment_origin", "units": "meters", "objects": {
        "block": {"top": {"position_m": [1, 2, 3], "status": "localized"}}}}
    result = {"frame": "environment_origin", "units": "meters", "points": [{
        "object_id": "block", "id": "top", "estimate": {"accepted": False, "position_base_m": [9, 9, 9]}}]}
    memory_helper.import_result(state, result, evidence="rejected.json", step=4)
    assert state["objects"]["block"]["top"]["position_m"] == [1, 2, 3]
    assert state["objects"]["block"]["top"]["status"] == "stale"


def test_stale_preserves_legacy_scalar_notes_and_rejected_scalar_point(tmp_path):
    memory = tmp_path / "space.json"
    state = {"schema_version": 1, "task": "classify_objects", "frame": "environment_origin",
             "units": "meters", "objects": {"block": {
                 "top": {"position_m": [0.1, 0.2, 0.8], "status": "localized"},
                 "status_note": "model-written annotation", "empty": {}}}}
    memory_helper.save(memory, state)
    result = {"frame": "environment_origin", "units": "meters", "points": [{
        "object_id": "block", "id": "status_note", "estimate": {"accepted": False}}]}
    memory_helper.import_result(state, result, evidence="rejected.json", step=1)
    memory_helper.save(memory, state)
    subprocess.run([sys.executable, str(SKILL / "scripts/space_memory.py"), "stale",
                    "--memory", str(memory), "--object", "block"], check=True, capture_output=True)
    points = memory_helper.load(memory)["objects"]["block"]
    assert points["top"]["status"] == "stale"
    assert points["status_note"] == "model-written annotation"
    assert points["empty"] == {}


def test_mode2_staging_does_not_create_or_update_memory(tmp_path):
    context = execution.prepare_skillpack_context(POLICY, "classify_objects", tmp_path)
    memory = context.workspace / "space.json"
    assert not memory.exists()
    assert not (context.workspace / ".agents/skills/robodojo-spatial-localize").exists()
    profile = execution.load_skillpack_profile(POLICY, "classify_objects", tmp_path)
    assert "spatial_localize_pixel" in profile.expected_tools
    memory.write_text('{"sentinel": "untouched"}')
    profile.stage_live_workspace(context.workspace, owner=tmp_path)
    assert memory.read_text() == '{"sentinel": "untouched"}'
