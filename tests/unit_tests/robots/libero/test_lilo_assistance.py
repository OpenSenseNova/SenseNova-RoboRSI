# Copyright 2026 The RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""The LiLo assistance channel is exact, live and score-read-only."""

from types import SimpleNamespace

import numpy as np
import pytest

from robots.libero import lilo_assistance as assistance


def test_instance_resolution_never_uses_ambiguous_prefixes():
    base = SimpleNamespace(
        objects_dict={"bowl_1": None, "bowl_10": None},
        fixtures_dict={},
        object_states_dict={"bowl_1_site": SimpleNamespace(parent_name="bowl_1")},
    )
    assert assistance.resolve_object(base, "bowl_1_main") == "bowl_1"
    assert assistance.resolve_object(base, "bowl_1_site") == "bowl_1"
    with pytest.raises(ValueError):
        assistance.resolve_object(base, "bowl_2_main")


def test_geometry_height_uses_live_rotated_collision_mesh():
    rotation = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    model = SimpleNamespace(
        geom_name2id=lambda _: 0,
        geom_type=[7],
        geom_size=[[1, 1, 1]],
        geom_dataid=[0],
        mesh_vertadr=[0],
        mesh_vertnum=[2],
        mesh_vert=np.array([[-0.2, 0, 0], [0.1, 0, 0]]),
    )
    data = SimpleNamespace(geom_xpos=[[0, 0, 1]], geom_xmat=[rotation.ravel()])
    base = SimpleNamespace(sim=SimpleNamespace(model=model, data=data))
    assert assistance.geometry_heights(
        base, SimpleNamespace(contact_geoms=["mesh"])
    ) == pytest.approx((0.9, 1.2))


def test_context_reads_exact_place_payload_without_advancing_score(monkeypatch):
    base = SimpleNamespace(
        objects_dict={"bowl_1": None, "plate_1": None},
        fixtures_dict={},
        obj_body_id={"bowl_1": 0, "plate_1": 1},
        object_states_dict={},
    )
    records = {
        "bowl_1": {
            "name": "bowl_1",
            "position": [0.4, 0.2, 0.5],
            "top_z": 0.55,
            "bottom_z": 0.45,
            "held": True,
        },
        "plate_1": {
            "name": "plate_1",
            "position": [0.4, 0.2, 0.5],
            "top_z": 0.55,
            "bottom_z": 0.45,
            "held": False,
        },
    }
    monkeypatch.setattr(assistance, "object_record", lambda _base, name: records[name])
    tracker = SimpleNamespace(
        skills=["pick bowl 1", "place bowl 1"], completed=1, success=False
    )
    mappings = {
        "pick bowl 1": {
            "language": "pick bowl",
            "target_object": "bowl_1_main",
            "bddl_predicates": [["pickedup", "bowl_1_main"]],
        },
        "place bowl 1": {
            "language": "place bowl",
            "target_object": "plate_1_main",
            "bddl_predicates": [["on", "bowl_1_main", "plate_1_main"]],
        },
    }
    result = assistance.task_context(
        SimpleNamespace(env=base),
        tracker,
        mappings,
        {"robot0_eef_pos": np.array([0.4, 0.2, 0.6])},
    )
    assert tracker.completed == 1
    assert result["contract"] == "lilo-instance-pose-feedback-v1"
    assert result["held_objects"] == ["bowl_1"]
    assert result["current"]["payload"] == "bowl_1"
    assert result["current"]["pickup_language"] == "pick bowl"
    assert result["current"]["release_eef_position"] == pytest.approx([0.4, 0.2, 0.71])


def test_context_restores_live_stove_contact_geometry(monkeypatch):
    base = SimpleNamespace(
        objects_dict={},
        fixtures_dict={"flat_stove_2": None},
        obj_body_id={"flat_stove_2": 0},
        object_states_dict={},
    )
    record = {
        "name": "flat_stove_2",
        "position": [-0.02, 0.3078, 0.905],
        "top_z": 0.955,
        "bottom_z": 0.885,
        "held": False,
    }
    monkeypatch.setattr(assistance, "object_record", lambda _base, _name: record)
    tracker = SimpleNamespace(
        skills=["turn on the stove 2"], completed=0, success=False
    )
    mappings = {
        "turn on the stove 2": {
            "language": "turn on the stove",
            "target_object": "flat_stove_2_main",
            "bddl_predicates": [["turnon", "flat_stove_2_main"]],
        }
    }

    result = assistance.task_context(
        SimpleNamespace(env=base), tracker, mappings, {"robot0_eef_pos": np.zeros(3)}
    )

    assert result["current"]["recommended_contact_eef_position"] == pytest.approx(
        [-0.02, 0.3078, 0.947]
    )
    assert "+0.55" in result["current"]["recommended_contact_sequence"]


def test_context_restores_bottom_drawer_push_geometry(monkeypatch):
    base = SimpleNamespace(
        objects_dict={},
        fixtures_dict={"wooden_cabinet_1": None},
        obj_body_id={"wooden_cabinet_1": 0},
        object_states_dict={},
    )
    record = {
        "name": "wooden_cabinet_1",
        "position": [0.0, 0.3, 0.905],
        "top_z": 1.128,
        "bottom_z": 0.91,
        "held": False,
    }
    monkeypatch.setattr(assistance, "object_record", lambda _base, _name: record)
    tracker = SimpleNamespace(
        skills=["close the bottom drawer of the cabinet 1"],
        completed=0,
        success=False,
    )
    mappings = {
        "close the bottom drawer of the cabinet 1": {
            "language": "close the bottom drawer of the cabinet",
            "target_object": "wooden_cabinet_1_main",
            "bddl_predicates": [["close", "wooden_cabinet_1_bottom_region"]],
        }
    }

    result = assistance.task_context(
        SimpleNamespace(env=base), tracker, mappings, {"robot0_eef_pos": np.zeros(3)}
    )

    assert np.asarray(
        result["current"]["recommended_contact_eef_waypoints"]
    ) == pytest.approx(
        np.asarray(
            [
                [0.0, 0.012, 0.952],
                [0.0, 0.085, 0.952],
                [0.0, 0.15, 0.952],
                [0.0, 0.202, 0.952],
            ]
        )
    )
    assert "pitch 0.8" in result["current"]["recommended_contact_sequence"]
