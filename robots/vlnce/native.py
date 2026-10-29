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

"""VLN-CE evaluation binding adapted from gpt6-test navigation_worker.py.

Official Habitat/VLN-CE owns reset, action semantics and metric computation.
The source file hash and pinned upstream revisions are recorded in provenance.json.
Executed by the Python 3.8 worker; optional imports stay inside methods.
"""

import base64
import hashlib
import io
import os
import sys
from types import ModuleType


def file_sha256(filename):
    with open(filename, "rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def encode_image(array):
    from PIL import Image

    output = io.BytesIO()
    Image.fromarray(array).convert("RGB").save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


class Episode:
    def __init__(self, spec):
        sys.path.insert(0, spec["repo_root"])
        import habitat

        # The official package initializer imports all training baselines, whose
        # optional LMDB/TensorBoard dependencies may compile into a shared env.
        # Load only the official evaluation modules from the pinned package path.
        # No simulator, dataset, action or metric implementation is replaced.
        package = ModuleType("habitat_extensions")
        package.__path__ = [os.path.join(spec["repo_root"], "habitat_extensions")]
        sys.modules["habitat_extensions"] = package
        import habitat_extensions.measures  # noqa: F401 (official metric registration)
        from habitat_extensions.config.default import get_extended_config
        from habitat_extensions.task import RxRVLNCEDatasetV1, VLNCEDatasetV1

        self.spec = spec
        rxr = spec.get("dataset", "r2r") == "rxr"
        config_name = "rxr_vlnce_english_task.yaml" if rxr else "vlnce_task.yaml"
        cfg = get_extended_config(
            os.path.join(spec["repo_root"], "habitat_extensions/config", config_name)
        )
        cfg.defrost()
        cfg.SEED = spec["seed"]
        cfg.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = spec.get("gpu_device_id", 0)
        cfg.SIMULATOR.RGB_SENSOR.WIDTH = 512
        cfg.SIMULATOR.RGB_SENSOR.HEIGHT = 512
        cfg.ENVIRONMENT.MAX_EPISODE_STEPS = spec["max_steps"]
        cfg.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        cfg.DATASET.DATA_PATH = os.path.join(
            spec["data_root"], "{split}/{split}.json.gz"
        )
        cfg.DATASET.SCENES_DIR = spec["scenes_root"]
        cfg.DATASET.SPLIT = spec["split"]
        cfg.DATASET.EPISODES_ALLOWED = [spec["episode_id"]]
        cfg.TASK.NDTW.GT_PATH = os.path.join(
            spec["data_root"], "{split}/{split}_gt.json.gz"
        )
        cfg.TASK.NDTW.SPLIT = spec["split"]
        if rxr:
            role = spec.get("role", "guide")
            if role not in RxRVLNCEDatasetV1.annotation_roles:
                raise ValueError("invalid RxR role")
            cfg.DATASET.ROLES = [role]
            # All official language configs share simulator settings; override
            # the English filter so Hindi/Telugu are never silently excluded.
            cfg.DATASET.LANGUAGES = (
                [spec["language"]] if spec.get("language") else ["*"]
            )
            cfg.DATASET.DATA_PATH = os.path.join(
                spec["data_root"], "{split}/{split}_{role}.json.gz"
            )
            # Resolve the selected role first. Otherwise the official NDTW
            # loader merges both roles even for a single-role evaluation.
            cfg.TASK.NDTW.GT_PATH = os.path.join(
                spec["data_root"], "{split}/{split}_" + role + "_gt.json.gz"
            )
        # Oracle actions/progress are training-only sensors. Keep official
        # measures intact, but never compute or expose these to the model.
        # The public instruction is read directly from current_episode below.
        # Omitting its redundant legacy Discrete(0) space also avoids Gym's
        # newer positive-size assertion without patching Gym or Habitat.
        cfg.TASK.SENSORS = []
        cfg.freeze()
        dataset_class = RxRVLNCEDatasetV1 if rxr else VLNCEDatasetV1
        dataset = dataset_class(cfg.DATASET)
        if len(dataset.episodes) != 1:
            raise ValueError(
                "dataset/split/role/language/episode_id must select exactly one official episode"
            )
        self.actions = tuple(action.lower() for action in cfg.TASK.POSSIBLE_ACTIONS)
        self.turn_degrees = cfg.SIMULATOR.TURN_ANGLE
        self.tilt_degrees = cfg.SIMULATOR.TILT_ANGLE if rxr else None
        self.forward_metres = cfg.SIMULATOR.FORWARD_STEP_SIZE
        scene = dataset.episodes[0].scene_id
        if not os.path.isfile(scene):
            raise FileNotFoundError(scene)
        self.env = habitat.Env(config=cfg, dataset=dataset)
        self.env.seed(spec["seed"])
        self.observations = self.env.reset()
        self.steps = 0
        self.stopped = False
        file = cfg.DATASET.DATA_PATH.format(
            split=spec["split"], role=spec.get("role", "guide")
        )
        self.provenance = {
            "dataset": spec.get("dataset", "r2r"),
            "episode_id": dataset.episodes[0].episode_id,
            "scene": scene,
            "initial_position": dataset.episodes[0].start_position,
            "initial_rotation_xyzw": dataset.episodes[0].start_rotation,
            "initialization": "official_dataset_start",
            "effective_habitat_config": cfg.dump(),
            "dataset_sha256": file_sha256(file),
            "reference_sha256": file_sha256(
                cfg.TASK.NDTW.GT_PATH.format(split=spec["split"])
            ),
        }
        if rxr:
            self.provenance.update(
                {
                    "role": role,
                    "language": dataset.episodes[0].instruction.language,
                    "instruction_id": dataset.episodes[0].instruction.instruction_id,
                }
            )

    def observe(self):
        return {
            "instruction": self.env.current_episode.instruction.instruction_text,
            "step_index": self.steps,
            "terminated": self.stopped,
            "truncated": self.env.episode_over and not self.stopped,
            "actions": list(self.actions),
            "turn_degrees": self.turn_degrees,
            "tilt_degrees": self.tilt_degrees,
            "language": getattr(self.env.current_episode.instruction, "language", None),
            "forward_metres": self.forward_metres,
            "images": {"front": encode_image(self.observations["rgb"])},
        }

    def step(self, action):
        if self.env.episode_over:
            raise ValueError("episode_done")
        self.observations = self.env.step(self.actions.index(action))
        self.steps += 1
        self.stopped = action == "stop"
        return {
            "ok": True,
            "step_index": self.steps,
            "terminated": self.stopped,
            "truncated": self.env.episode_over and not self.stopped,
        }

    def evaluate(self):
        if not (self.stopped or self.env.episode_over):
            raise ValueError("unfinished VLNCE episode cannot be scored")
        metrics = self.env.get_metrics()
        return {
            "success": bool(metrics["success"]),
            "steps": self.steps,
            "terminated": self.stopped,
            "truncated": self.env.episode_over and not self.stopped,
            "metrics": metrics,
            "provenance": self.provenance,
        }

    def close(self):
        self.env.close()
