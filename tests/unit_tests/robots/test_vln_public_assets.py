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

"""Independent public preparation must preserve tasks and reject changed inputs."""

import pytest

from robots.vln_shared.public_assets import prepare
from robots.vln_shared.storage import file_digest, read_json, write_json

IMAGE = "ghcr.io/example/navigation@sha256:" + "1" * 64


@pytest.fixture
def inputs(tmp_path):
    package = tmp_path / "package"
    scenes = tmp_path / "scenes"
    scene = scenes / "mp3d/room/room.glb"
    scene.parent.mkdir(parents=True)
    scene.write_bytes(b"fixture scene bytes; no simulator")
    annotation = package / "r2r/data/episodes.json"
    write_json(annotation, {"original": True})
    catalog = {
        "spec": {
            "dataset": "r2r",
            "python": "/opt/native/bin/python",
            "seed": 0,
            "split": "rand100",
            "native_commit": "a" * 40,
            "data_root": "data",
            "scenes_root": "scenes",
            "inputs": {"data/episodes.json": file_digest(annotation)},
        },
        "episodes": [
            {
                "id": "r2r-7",
                "dataset": "r2r",
                "episode_id": "7",
                "episode_index": 19,
                "instruction": "Navigate",
                "scene": "mp3d/room/room.glb",
            }
        ],
    }
    write_json(package / "r2r/catalog.json", catalog)

    def row(path, root):
        return {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_digest(path),
        }

    entry = {
        "benchmark": "vlnce",
        "task_count": 1,
        "scene_subdir": "scenes",
        "files": [
            row(p, package / "r2r") for p in [annotation, package / "r2r/catalog.json"]
        ],
        "scenes": [row(scene, scenes)],
    }
    write_json(
        package / "input-index.json", {"schema_version": 1, "datasets": {"r2r": entry}}
    )
    return package, scenes, scene, catalog


@pytest.mark.parametrize("budget", [500, 5000])
def test_public_prepare_keeps_frozen_task_and_reuses_scene(inputs, tmp_path, budget):
    package, scenes, scene, original = inputs
    result = prepare(
        package, "r2r", scenes, tmp_path / "registry", IMAGE, budget=budget
    )
    manifest = read_json(result)
    assert read_json(result.parent / "assets/catalog.json") == original
    assert manifest["tasks"][0]["task_id"] == "r2r-7"
    assert manifest["tasks"][0]["seeds"] == [0]
    assert manifest["tasks"][0]["native"]["episode_index"] == 19
    assert manifest["tasks"][0]["max_steps"] == budget
    assert (
        result.parent / "assets/scenes/mp3d/room/room.glb"
    ).stat().st_ino == scene.stat().st_ino
    with pytest.raises(FileExistsError):
        prepare(package, "r2r", scenes, result.parent.parent, IMAGE)


@pytest.mark.parametrize("modified", ["scene", "annotation"])
def test_corrupt_input_rejected_before_output_creation(inputs, tmp_path, modified):
    package, scenes, scene, _ = inputs
    target = scene if modified == "scene" else package / "r2r/data/episodes.json"
    target.write_bytes(b"changed")
    output = tmp_path / "registry"
    with pytest.raises(ValueError, match="checksum"):
        prepare(package, "r2r", scenes, output, IMAGE)
    assert not output.exists()


@pytest.mark.parametrize("bad", ["../outside", "/absolute", "a/../b", "a\\b", "a//b"])
def test_unsafe_index_path_rejected(inputs, tmp_path, bad):
    package, scenes, _, _ = inputs
    index = read_json(package / "input-index.json")
    index["datasets"]["r2r"]["scene_subdir"] = bad
    write_json(package / "input-index.json", index)
    with pytest.raises(ValueError, match="path"):
        prepare(package, "r2r", scenes, tmp_path / "registry", IMAGE)


def test_explicit_copy_does_not_share_mutable_scene_inode(inputs, tmp_path):
    package, scenes, scene, _ = inputs
    result = prepare(
        package, "r2r", scenes, tmp_path / "registry", IMAGE, copy_scenes=True
    )
    copied = result.parent / "assets/scenes/mp3d/room/room.glb"
    assert copied.read_bytes() == scene.read_bytes()
    assert copied.stat().st_ino != scene.stat().st_ino


def test_output_symlink_rejected(inputs, tmp_path):
    package, scenes, _, _ = inputs
    (tmp_path / "linked").symlink_to(scenes, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        prepare(package, "r2r", scenes, tmp_path / "linked/registry", IMAGE)
