# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

from roborsi.evaluation.batch import BatchEvaluator, expand_cases
from roborsi.evaluation.store import ResultStore
from tests.unit_tests.roborsi.evaluation.test_batch import _PublishedCases, _registry


def test_progress_reports_task_seed_and_completed_count(tmp_path):
    registry = _registry(tmp_path / "registry")
    store = ResultStore(tmp_path / "results")
    cells = expand_cases(registry, ["demo"], seeds=[0, 1])
    events = []
    evaluator = BatchEvaluator(registry, store, _PublishedCases(registry, store))
    evaluator.run("first", cells, "agent-v1", progress=events.append)
    assert sorted(event["event"] for event in events) == [
        "finished",
        "finished",
        "started",
        "started",
    ]
    finished = [event for event in events if event["event"] == "finished"]
    assert [event["completed"] for event in finished] == [1, 2]
    assert {event["seed"] for event in finished} == {0, 1}
    assert all(event["task_id"] and event["total"] == 2 for event in events)
    assert all(event["elapsed_s"] >= 0 for event in events)
    events.clear()
    evaluator.run(
        "second", cells, "agent-v1", resume_from="first", progress=events.append
    )
    assert [event["event"] for event in events] == ["reused", "reused"]
    assert events[-1]["completed"] == 2
