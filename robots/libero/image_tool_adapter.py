# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
"""PRO camera/state bridge for the image's read-only tools.
Image functions produce physical actions; ThreeBenchmarkAdapter owns score and step limits.
"""

from __future__ import annotations

import math

from roborsi.evaluation.messages import StepResult
from roborsi.evaluation.result import write_json_atomic
from robots.libero.image_tool_records import CameraRecords
from robots.libero.image_tools import (
    PROFILE,
    READ_ONLY_NAMES,
    OpenPiExecutionClient,
    call_image_tool,
    image_manifest,
    validate_tool_urls,
)
from robots.libero.image_tools import action_schema as observation_action_schema
from robots.libero.three_benchmark_adapter import ThreeBenchmarkAdapter
from robots.libero.three_benchmark_env import _ProEnvironment


class ObservationEnvironment(_ProEnvironment):
    def camera_artifacts(self):
        """Capture both camera resolutions without advancing the simulator."""
        import numpy as np

        from robots.libero.image_tool_records import _metric_depth, _world_from_depth

        raw = self._facade.raw_obs()
        artifacts = {}
        for camera, native in (
            ("agentview", "agentview"),
            ("wrist", "robot0_eye_in_hand"),
        ):
            rgb = np.asarray(raw[f"{native}_image"], dtype=np.uint8)
            meta = self._facade.get_camera_meta(
                camera_name=native, height=rgb.shape[0], width=rgb.shape[1]
            )
            depth = _metric_depth(raw[f"{native}_depth"], meta)[::-1]
            artifacts[f"{camera}.png"] = rgb[::-1]
            artifacts[f"{camera}_metadata.json"] = meta
            artifacts[f"{camera}_depth.npz"] = depth.astype(np.float32)
            artifacts[f"{camera}_world.npz"] = _world_from_depth(depth, meta).astype(
                np.float32
            )
            high_rgb, high_depth = self._facade.render_camera(
                camera_name=native, height=1024, width=1024, depth=True
            )
            high_meta = self._facade.get_camera_meta(
                camera_name=native, height=1024, width=1024
            )
            artifacts[f"{camera}_high.png"] = np.asarray(high_rgb, dtype=np.uint8)[::-1]
            artifacts[f"{camera}_high_metadata.json"] = high_meta
            artifacts[f"{camera}_world_high.npz"] = _world_from_depth(
                _metric_depth(high_depth, high_meta)[::-1], high_meta
            ).astype(np.float32)
        return artifacts


class ObservationAdapter(ThreeBenchmarkAdapter):
    def __init__(self, directory, artifacts=None, *, env_factory=None, tool_urls=None):
        super().__init__(
            directory,
            artifacts,
            env_factory=env_factory or ObservationEnvironment,
        )
        self._tool_clients = {}
        if tool_urls:
            from roborsi.utils.rpc.http_rpc import HttpRpcClient

            self._tool_urls = validate_tool_urls(tool_urls)
            self._tool_clients = {
                "vla": OpenPiExecutionClient(tool_urls["vla"]),
                "sam3": HttpRpcClient(tool_urls["sam3"]),
            }
        self._image_session = None
        self._tool_feedback = {}
        self._image_manifest = image_manifest()
        write_json_atomic(
            self._native_dir / "image-tool-provenance.json", self._image_manifest
        )
        self._action_schema = observation_action_schema(bool(self._tool_clients))
        self._camera_records = CameraRecords(self._native_dir / "states")

    def start(self, task, episode_id):
        if (
            task.benchmark != "libero_pro"
            or task.capabilities.get("image_observation_tools") != PROFILE
        ):
            raise ValueError("Image observation tools require the matching PRO service")
        # Original adapter validates its original schema. Preserve that code,
        # then expose only this opt-in profile's extended schema to the caller.
        from robots.libero.three_benchmark_adapter import ACTION_SCHEMA

        base_task = task.model_copy(
            update={
                "capabilities": {**task.capabilities, "action_schema": ACTION_SCHEMA}
            }
        )
        result = super().start(base_task, episode_id)
        self._task = task
        return result.model_copy(
            update={
                "task": task,
                "capabilities": {
                    **result.capabilities,
                    "action_schema": self._action_schema,
                },
            }
        )

    def _observation(self, values, *, state_step=-1):
        self._camera_records.capture(
            self._env,
            self._latest or values,
            self._env_steps,
            terminated=self._primary_success,
            truncated=self._ended and not self._primary_success,
        )
        metadata, images = self._camera_records.view(state_step)
        values = {
            k: v for k, v in values.items() if k not in {"main_images", "wrist_images"}
        }
        values.update(images)
        values.update(
            camera_state=metadata,
            image_coordinates=(
                "row=0 top, col=0 left; back_project high uses "
                "agentview_high/wrist_high; low uses agentview"
            ),
        )
        return super()._observation(values)

    def _read_camera(self, data):
        name = data["primitive"]
        state_step = -1
        try:
            if not hasattr(self._env, "camera_artifacts"):
                raise ValueError("camera records are unavailable for this environment")
            args = {k: v for k, v in data.items() if k != "primitive"}
            result = call_image_tool(name, args, self._camera_records.directory)
            if name == "view_env_state" and "error" not in result:
                state_step = self._camera_records.get(args.get("step", -1)).step_idx
                # Images are published by _observation from these same immutable
                # records; don't duplicate the image bytes in the JSON feedback.
                result = {k: v for k, v in result.items() if not k.startswith("_image")}
        except (ValueError, FileNotFoundError) as exc:
            result = {"error": str(exc)}
        self._decision_steps += 1
        return StepResult(
            observation=self._observation(
                {**self._latest, name + "_result": result}, state_step=state_step
            ),
            terminated=False,
            truncated=False,
            reward=self._last_reward,
            native={"action_kind": name, "native_env_steps": self._env_steps},
        )

    def _segment(self, data):
        args = {k: v for k, v in data.items() if k != "primitive"}
        result = call_image_tool(
            "segment",
            args,
            self._camera_records.directory,
            endpoint=self._tool_urls["sam3"],
        )
        result = {k: v for k, v in result.items() if not k.startswith("_image")}
        self._decision_steps += 1
        return StepResult(
            observation=self._observation({**self._latest, "segment_result": result}),
            terminated=False,
            truncated=False,
            reward=self._last_reward,
            native={"action_kind": "segment", "native_env_steps": self._env_steps},
        )

    def _primitive_vectors(self, data):
        from robots.libero.image_vla_bridge import DIRECT_ACTIONS, ImageSkillSession

        name = data.get("primitive")
        if name in DIRECT_ACTIONS or (
            self._tool_clients and name in {"pi0_pick", "pi0_doubled"}
        ):
            self._tool_feedback = {}
            self._image_session = ImageSkillSession(self, data)
            for vector in self._image_session.events():
                yield vector
                if name == "release":
                    self._release_peak = max(
                        self._release_peak, self._gripper_opening()
                    )
        else:
            yield from super()._primitive_vectors(data)

    def _gripper_opening(self):
        state = self._state()
        return float(abs(state[6]) + abs(state[7]))

    def _direct_action_result(self, data, before, result):
        """Supplement image-tool diagnostics with measured physical steps.

        steps_used counts actual native steps, including early benchmark stops
        and zero-step convergence; it does not count the servo's final check.
        """
        name = data["primitive"]
        final = self._state()
        pitch, yaw = self._orientation()
        feedback = {
            "name": name,
            "steps_used": self._env_steps - before["steps"],
            "terminated": result.terminated,
            "truncated": result.truncated,
        }
        if name in {"move_to", "move_pose"}:
            target = [float(x) for x in data["xyz"]]
            feedback.update(
                target_xyz=target,
                final_eef_pos=[round(float(x), 4) for x in final[:3]],
                final_dist_m=round(math.dist(target, final[:3]), 4),
                max_steps=data.get("max_steps", 80 if name == "move_to" else 150),
            )
            if name == "move_pose":
                feedback["final_pitch"] = round(pitch, 4)
            for axis, value in (("pitch", pitch), ("yaw", yaw)):
                key = "target_" + axis
                if key in data:
                    feedback[key] = data[key]
                    feedback["final_" + axis] = round(value, 4)
                    feedback["final_" + axis + "_err"] = round(
                        self._wrapped_error(data[key], value), 4
                    )
        elif name in {"rotate_pitch", "rotate_wrist"}:
            axis = "pitch" if name == "rotate_pitch" else "yaw"
            index = 0 if axis == "pitch" else 1
            start = before["orientation"][index]
            target = data["target"] if "target" in data else start + data["delta"]
            angle = (pitch, yaw)[index]
            feedback.update(
                {
                    "start_" + axis: round(start, 4),
                    "target_" + axis: round(target, 4),
                    "final_" + axis: round(angle, 4),
                    "final_err": round(self._wrapped_error(target, angle), 4),
                }
            )
        elif name == "release":
            opening = self._gripper_opening()
            feedback.update(
                start_gripper_opening=round(before["opening"], 4),
                peak_gripper_opening=round(max(self._release_peak, opening), 4),
                final_gripper_opening=round(opening, 4),
            )
        elif name == "set_gripper":
            feedback.update(
                gripper=float(data["gripper"]), steps=feedback["steps_used"]
            )
        return feedback

    def step(self, action):
        if self._env is None or self._task is None or self._episode_id is None:
            raise ValueError("LIBERO observation adapter has not started")
        if self._ended:
            raise ValueError("LIBERO episode already ended")
        if (
            action.benchmark != self._task.benchmark
            or action.episode_id != self._episode_id
            or action.step_index != self._decision_steps
        ):
            raise ValueError("LIBERO action identity or decision step differs")
        from jsonschema import validate

        validate(action.data, self._action_schema)
        name = action.data.get("primitive")
        if name in READ_ONLY_NAMES:
            return self._read_camera(action.data)
        if name == "segment" and self._tool_clients:
            return self._segment(action.data)
        before = None
        if name in {
            "move_to",
            "move_pose",
            "rotate_pitch",
            "rotate_wrist",
            "release",
            "set_gripper",
        }:
            before = {
                "steps": self._env_steps,
                "orientation": self._orientation(),
                "opening": self._gripper_opening(),
            }
            self._release_peak = before["opening"]
        try:
            result = super().step(action)
            if self._image_session is not None:
                self._image_session.finish()
            if before is not None:
                feedback = {
                    **self._direct_action_result(action.data, before, result),
                    **self._tool_feedback,
                    # Legacy servos count the final convergence check as a step.
                    # Evaluation feedback reports real env.step calls instead.
                    "steps_used": self._env_steps - before["steps"],
                }
                if name == "set_gripper":
                    feedback["steps"] = feedback["steps_used"]
                obs = result.observation.model_copy(
                    update={
                        "data": {
                            **result.observation.data,
                            "action_result": feedback,
                        }
                    }
                )
                result = result.model_copy(update={"observation": obs})
            if self._image_session is not None and before is None:
                obs = result.observation.model_copy(
                    update={
                        "data": {
                            **result.observation.data,
                            "vla_result": {
                                **self._tool_feedback,
                                "task_success": self._primary_success,
                            },
                        }
                    }
                )
                result = result.model_copy(update={"observation": obs})
            return result
        finally:
            if self._image_session is not None:
                self._image_session.close()
                self._image_session = None
