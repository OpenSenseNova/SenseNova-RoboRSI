# Copyright 2026 The SenseNova-RoboRSI Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Own each simulator in a separate process, independent of HTTP threads."""

from __future__ import annotations

import io
import multiprocessing
import os
import random
import subprocess
import sys
from pathlib import Path

from robots.dexjoco.contracts import COMMIT, LIMITS, environment_action


def verify_upstream(root: Path) -> None:
    """Require the unchanged evaluated simulator revision."""

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True
        ).strip()

    if git("rev-parse", "HEAD") != COMMIT or git("status", "--porcelain"):
        raise ValueError("DexJoCo upstream must be clean at the pinned commit")


def task_config(root: Path, task: str) -> dict:
    """Load the original rand-full instruction and camera mapping."""
    import yaml

    if task not in LIMITS:
        raise ValueError("Unknown DexJoCo task")
    cfg = yaml.safe_load((root / "configs/rand_full" / f"{task}.yaml").read_text())
    if (
        cfg["env_name"] != task
        or cfg["robot_type"] not in {"single_arm", "dual_arm"}
        or not cfg["camera_mapping"]
        or not cfg["prompt"]
    ):
        raise ValueError("Invalid pinned task configuration")
    return cfg


class NativeSession:
    """Construct and drive the original task without model dependencies."""

    def __init__(self, root: Path, task: str, seed: int, multiplier: int) -> None:
        import numpy as np

        from robots.dexjoco.horizon import extend_episode_limit

        verify_upstream(root)
        cfg = task_config(root, task)
        os.environ.setdefault("MUJOCO_GL", "osmesa")
        random.seed(seed)
        np.random.seed(seed)
        sys.path.insert(0, str(root / "dexjoco"))
        from dexjoco.tasks import CONFIG_MAPPING

        kwargs = {
            "policy_mode": True,
            "render_mode": "rgb_array",
            "randomize": True,
            "seed": seed,
            "randomize_dynamics": False,
        }
        if "password" in cfg:
            kwargs["password"] = cfg["password"]
        self.env = CONFIG_MAPPING[task]().get_environment(**kwargs)
        self.dual = cfg["robot_type"] == "dual_arm"
        try:
            self.horizon = (
                extend_episode_limit(self.env, LIMITS[task], multiplier)
                if multiplier > 1
                else None
            )
        except BaseException:
            self.env.close()
            raise

    def reset(self):
        """Return original reset data to the trusted Adapter."""
        return self.env.reset()

    def step(self, row):
        """Execute one unmodified absolute target at native control frequency."""
        return self.env.step(environment_action(row, self.dual))

    def close(self) -> None:
        """Release the native render context and simulation."""
        self.env.close()

    def checkpoint(self) -> bytes:
        """Capture the original private audit arrays without resetting the scene."""
        import numpy as np

        target = self.env
        while hasattr(target, "env"):
            target = target.env
        data, model = target._data, target._model
        arrays = {
            name: np.array(getattr(data, name))
            for name in (
                "qpos",
                "qvel",
                "act",
                "ctrl",
                "qacc_warmstart",
                "qfrc_applied",
                "xfrc_applied",
                "mocap_pos",
                "mocap_quat",
                "userdata",
            )
        }
        arrays.update(
            {
                name: np.array(getattr(model, name))
                for name in ("body_pos", "body_quat", "geom_rgba")
            }
        )
        stream = io.BytesIO()
        np.savez_compressed(
            stream,
            env_step=np.array([target.env_step]),
            time=np.array([data.time]),
            **arrays,
        )
        return stream.getvalue()


def _worker(connection, root, task, seed, multiplier):
    session = None
    try:
        session = NativeSession(root, task, seed, multiplier)
        connection.send((True, session.horizon))
        while True:
            method, args = connection.recv()
            if method == "close":
                break
            if method not in {"reset", "step", "checkpoint"}:
                raise ValueError("Unknown native operation")
            connection.send((True, getattr(session, method)(*args)))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException as exc:
        try:
            connection.send((False, f"{type(exc).__name__}: {exc}"))
        except (EOFError, BrokenPipeError):
            pass
    finally:
        if session is not None:
            session.close()
        connection.close()


class ProcessNative:
    """Serialize simulator operations in a private spawn process."""

    def __init__(self, root: Path, task: str, seed: int, multiplier: int) -> None:
        ctx = multiprocessing.get_context("spawn")
        self.connection, child = ctx.Pipe()
        self.process = ctx.Process(
            target=_worker, args=(child, root, task, seed, multiplier)
        )
        self.closed = False
        self.process.start()
        child.close()
        try:
            self.horizon = self._receive()
        except BaseException:
            self.close()
            raise

    def _receive(self):
        if not self.connection.poll(120):
            raise TimeoutError("DexJoCo native worker timed out")
        ok, value = self.connection.recv()
        if not ok:
            raise RuntimeError(value)
        return value

    def reset(self):
        """Reset on the owning process."""
        self.connection.send(("reset", ()))
        return self._receive()

    def step(self, row):
        """Advance on the owning process."""
        self.connection.send(("step", (row,)))
        return self._receive()

    def checkpoint(self) -> bytes:
        """Capture private state on its owning process."""
        self.connection.send(("checkpoint", ()))
        return self._receive()

    def close(self) -> None:
        """Reap the worker even after a failed request; repeated close is safe."""
        if self.closed:
            return
        try:
            if self.process.is_alive():
                self.connection.send(("close", ()))
        except (OSError, BrokenPipeError):
            pass
        finally:
            self.connection.close()
            self.process.join(5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()
            self.closed = True
