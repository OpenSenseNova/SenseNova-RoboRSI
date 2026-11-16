# Copyright 2026 The SenseNova-RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run LIBERO directly, without the RoboRSI Adapter or RLinf environment wrapper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.transform_utils import quat2axisangle

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--actions", help="JSON list of explicit native action vectors")
    args = parser.parse_args()
    actions = json.loads(args.actions) if args.actions is not None else None
    if actions is not None and len(actions) != args.max_steps:
        parser.error("--actions must contain max-steps vectors")
    suite = benchmark.get_benchmark(args.suite)()
    task = suite.get_task(args.task)
    initial_states = suite.get_task_init_states(args.task)
    env = OffScreenRenderEnv(
        bddl_file_name=str(
            Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        ),
        camera_heights=256,
        camera_widths=256,
        camera_depths=True,
        horizon=args.max_steps,
        ignore_done=True,
    )
    try:
        env.seed(args.seed)
        env.reset()
        env.set_init_state(initial_states[args.seed % len(initial_states)])
        action = [0.0] * 6 + [-1.0]
        # Use the same initial settling as the supported RLinf runtime.
        for _ in range(15):
            env.step(action)
        states = []
        for step in range(1, args.max_steps + 1):
            observation, reward, done, _ = env.step(
                actions[step - 1] if actions is not None else action
            )
            if actions is not None:
                states.append(
                    np.concatenate(
                        [
                            observation["robot0_eef_pos"],
                            quat2axisangle(observation["robot0_eef_quat"]),
                            observation["robot0_gripper_qpos"],
                        ]
                    ).tolist()
                )
            success = bool(env.check_success())
            assert bool(done) == success
            if success:
                break
        values = {
            "suite": args.suite,
            "task_id": str(args.task),
            "seed": args.seed,
            "success": success,
            "score": float(success),
            "last_reward": float(reward),
            "steps": step,
        }
        if actions is not None:
            controller = env.env.robots[0].part_controllers["right"]
            values.update(
                states=states,
                control={
                    "name": controller.name,
                    "input_type": controller.input_type,
                    "input_ref_frame": controller.input_ref_frame,
                    "input_min": controller.input_min.tolist(),
                    "input_max": controller.input_max.tolist(),
                    "output_min": controller.output_min.tolist(),
                    "output_max": controller.output_max.tolist(),
                    "frequency_hz": env.env.control_freq,
                    "action_low": env.env.action_spec[0].tolist(),
                    "action_high": env.env.action_spec[1].tolist(),
                },
            )
        record = json.dumps(values, allow_nan=False)
        if args.output is not None:
            args.output.write_text(record)
        else:
            print("ROBORSI_NATIVE_REFERENCE=" + record, flush=True)

    finally:
        env.close()


if __name__ == "__main__":
    main()
