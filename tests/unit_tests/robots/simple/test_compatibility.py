# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline integrity and registration checks for the restored upstream task."""

import hashlib
import sys
from types import SimpleNamespace

import pytest

from robots.simple.contracts import SOURCES
from robots.simple.runtime_setup import register_compatibility_task


def test_pinned_extension_registration_rejects_tampering_and_duplicates(
    tmp_path, monkeypatch
):
    alias = "xmove_bendpick"
    spec = dict(SOURCES["compatibility"][alias])
    body = (
        f"class {spec['class_name']}:\n"
        f"    uid = {spec['task_uid']!r}\n"
        "    metadata = {'max_episode_steps': 800}\n"
    ).encode()
    spec["sha256"] = hashlib.sha256(body).hexdigest()
    monkeypatch.setitem(SOURCES["compatibility"], alias, spec)
    path = tmp_path / spec["path"].split("/")[-1]
    path.write_bytes(body + b"# changed\n")
    with pytest.raises(ValueError, match="SHA256"):
        register_compatibility_task(alias, tmp_path)
    path.write_bytes(body)
    registry = {}

    def register(**kwargs):
        registry[kwargs["id"]] = kwargs

    monkeypatch.setitem(
        sys.modules, "gymnasium", SimpleNamespace(registry=registry, register=register)
    )
    module_name = "simple.tasks." + path.stem
    try:
        assert register_compatibility_task(alias, tmp_path) == spec
        native = registry["simple/" + SOURCES["tasks"][alias]["env_id"]]
        assert native["kwargs"] == {"task": spec["task_uid"]}
        assert native["entry_point"] == spec["entry_point"]
        with pytest.raises(ValueError, match="already registered"):
            register_compatibility_task(alias, tmp_path)
    finally:
        sys.modules.pop(module_name, None)


def test_other_tasks_do_not_need_compatibility_installation(tmp_path):
    assert register_compatibility_task("tabletop", tmp_path) == {}
    with pytest.raises(ValueError, match="install the pinned"):
        register_compatibility_task("xmove_bendpick", tmp_path)
