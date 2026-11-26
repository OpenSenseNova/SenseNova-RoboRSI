"""Native owner: preserve physics/scoring, archive JPEG, send exact RGB."""
from pathlib import Path
import importlib.util
import hashlib
import json

from native_rgb import TRANSPORT, archive_rgb, encode_rgb

SOURCE = Path(__file__).resolve().parent/"vendor/robodojo_adaptive/source"
BASE = SOURCE/"code/configurations/robodojo/20261005_standard_1x_10x/adapters/adapter.py"
pins = json.loads((Path(__file__).resolve().parent/"submission.json").read_text())["hybrid"]
manifest = Path(__file__).resolve().parent/pins["manifest_path"]
if hashlib.sha256(manifest.read_bytes()).hexdigest() != pins["manifest_sha256"]:
    raise ValueError("Native RGB original-source manifest differs")
entry = next(row for row in json.loads(manifest.read_text())["files"]
             if row["path"] == "source/"+BASE.relative_to(SOURCE).as_posix())
if hashlib.sha256(BASE.read_bytes()).hexdigest() != entry["sha256"] or BASE.stat().st_size != entry["bytes"]:
    raise ValueError("Native RGB original adapter differs from pinned physics source")
spec = importlib.util.spec_from_file_location("_native_rgb_original_adapter", BASE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Adapter(module.Adapter):
    """Preserve RGB/storage/scoring and dispatch private native cuRobo requests."""
    def __init__(self, environment_dir, options):
        self.save_video = options.get('save_video', False)
        if type(self.save_video) is not bool:
            raise ValueError('save_video must be boolean')
        self.archive_observations = options.get('archive_native_observations', True)
        if type(self.archive_observations) is not bool:
            raise ValueError('archive_native_observations must be boolean')
        super().__init__(environment_dir, options)

    def _native_result(self):
        # Same pinned native scoring calculation, with opt-in file publication.
        if not self.finalized:
            success = bool(self.env.success[0])
            score = 1.0 if success else (float(self.env.reward_manager.get_score()[0]) / 100.0
                                         if hasattr(self.env, 'get_score') else 0.0)
            if getattr(self, 'save_video', False):
                self.env.save_video(0, str(self.record_dir/'episode_0000000.mp4'),
                                    'success' if success else 'fail')
            self.result = {'success':success,'score':score,'layout_id':int(self.env.env_seeds[0])}
            self.finalized = True
        return self.result

    def _transition(self, applied_action=None):
        import numpy as np

        raw = self.env.get_obs()
        if hasattr(self.env, "camera_manager") and not raw.get("additional_info", {}).get("spatial_calibration"):
            import os
            if os.environ.get("ROBORSI_SPATIAL_CALIBRATION", "1") == "1":
                from native_camera import calibrated_observation
                calibrated_observation(self.env, raw)
        # Freeze renderer buffers once, so the archive and live bytes describe
        # the same frame even if the renderer later reuses its array storage.
        vision = {camera: dict(color=np.array(fields["color"], copy=True, order="C"))
                  for camera, fields in raw["vision"].items()}
        payload = encode_rgb(vision)
        # Training-first callers already retain the exact model input in the
        # inner training store. Avoid encoding/writing a second image archive
        # for every physical control step; legacy callers keep their archive.
        archive = getattr(self, 'archive_observations', True)
        cameras, artifacts = (archive_rgb(vision, self.observation_dir, self.frame)
                              if archive else ({}, {}))
        state = {key: np.asarray(value).tolist() for key, value in raw["state"].items()}
        self._current_state = state
        step = int(self.env.take_action_cnt[0])
        meta = dict(transport=TRANSPORT, frame=self.frame, control_step=step,
                    rgb_sha256={name: item["sha256"] for name, item in payload.items()},
                    archive=artifacts)
        if archive:
            with (self.record_dir/"rgb_frames.jsonl").open("a") as stream:
                stream.write(json.dumps(meta, allow_nan=False)+"\n")
        observation = dict(instruction=raw["instruction"], state=state, cameras=cameras,
                           control_step=step,
                           control_frequency_hz=int(raw["additional_info"]["frequency"]))
        extra = raw.get("extra_info")
        if isinstance(extra, dict) and "detailed_instruction" in extra:
            observation["extra_info"] = {"detailed_instruction": extra["detailed_instruction"]}
        if raw["additional_info"].get("spatial_calibration") is not None:
            observation["spatial_calibration"] = raw["additional_info"]["spatial_calibration"]
        self.frame += 1
        done = bool(self.env.end_flag[0])
        result = dict(observation=observation, done=done,
                      metrics=self._native_result() if done else {},
                      native_rgb=meta, _native_rgb=payload)
        if applied_action is not None:
            result["applied_action"] = applied_action
        return result

    def step(self, action):
        if isinstance(action, dict) and action.get("_roborsi_control") == "bounded_curobo":
            from native_curobo import execute_control
            return execute_control(self.env, action, step_joint=self._step_joint,
                initial_observation={"state": self._current_state}, record_dir=self.record_dir)
        return super().step(action)
