# Copyright 2026 The HEART Authors.
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

"""Bridge the original isolated CLI policy to RoboRSI's Agent contract.

The trusted bridge receives Task metadata and terminal feedback. The model
receives only the historical file interface inside its mount namespace.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import deque
from pathlib import Path

from roborsi.evaluation.media import CaseMediaChannel
from roborsi.evaluation.messages import (
    Action,
    EpisodeStop,
    Observation,
    StepResult,
    Task,
)
from robots.dexjoco.agents.baseline.session import CliSession
from robots.dexjoco.contracts import ALIGNMENT, CAPABILITIES, LIMITS, PROTOCOL, SEEDS
from robots.dexjoco.reference import load_reference


class FrozenPolicyAgent:
    """Retain chunking, Click Mouse alignment and fresh-session continuation."""

    def __init__(
        self,
        case_dir: Path,
        archive: Path,
        assets: Path,
        *,
        prompts=None,
        cli_config=None,
        config_fingerprint=None,
        session_factory=CliSession,
    ) -> None:
        self.root = Path(case_dir) / "dexjoco-policy"
        self.root.mkdir(parents=True, exist_ok=False)
        self.workspace = self.root / "policy"
        self.private = self.root / "private"
        self.inputs = self.workspace / "input"
        self.interface = self.private / "interface"
        for path in (self.inputs, self.interface, self.private / "action-transport"):
            path.mkdir(parents=True, exist_ok=True)
        self.media = CaseMediaChannel(
            Path(os.environ.get("ROBORSI_MEDIA_DIR", Path(case_dir) / "media"))
        )
        self.assets = Path(assets)
        self.ref = load_reference(Path(archive))
        self.cfg = self.ref.load_config(
            self.ref.ROOT
            / "configs/dexjoco_icl_5ep_rand_full_robot_v2_two_session.json"
        )
        self.cli_config = cli_config
        self.config_fingerprint = config_fingerprint
        if prompts is None:
            folder = Path(__file__).parent / "prompts"
            common = (folder / "policy.md").read_text()
            prompts = {
                phase: (folder / (phase + ".md")).read_text() + common
                for phase in ("initial", "continuation")
            }
        self.prompts = dict(prompts)
        self.prompt_hashes = {
            phase: hashlib.sha256(text.encode()).hexdigest()
            for phase, text in self.prompts.items()
        }
        self.factory = session_factory
        self.task = None
        self.session = None
        self.queue = deque()
        self.decision = 0
        self.phase = "initial"
        self.records = []
        self.last_row = None
        self.next_step = 0
        self.last_saved_step = -1
        self.episode = None
        self.closed = False

    def reset(self, task: Task) -> None:
        """Install only this task's verified payload and public robot assets."""
        if self.task is not None or self.closed:
            raise ValueError("A fresh DexJoCo Agent is required for every episode")
        if (
            task.benchmark != "dexjoco"
            or task.task_id not in LIMITS
            or task.seed not in SEEDS
            or task.native
            != {"regime": "rand_full", "protocol": PROTOCOL, "multiplier": 10}
            or task.max_steps != 10 * LIMITS[task.task_id]
        ):
            raise ValueError(
                "Compatibility Agent requires the pinned two-session cohort"
            )
        self.task = task
        self.dual = task.task_id.startswith("bimanual_")
        if task.capabilities != CAPABILITIES:
            raise ValueError("Unexpected action schema")
        self.standard_limit = LIMITS[task.task_id]
        self.package = self.assets / "rand_full" / task.task_id
        catalog = json.loads(
            (self.ref.ROOT / "provenance/knowledge_packages.json").read_text()
        )["packages"]
        self.expected_hash = catalog[task.task_id]["manifest_sha256"]
        manifest = self.ref.verify_package(
            self.package, task.task_id, self.cfg, self.expected_hash
        )
        shutil.copytree(self.package / "payload", self.inputs / "task_knowledge")
        self._verify_payload()
        robot_type = "dual_arm" if self.dual else "single_arm"
        reference = self.assets / "robot_reference" / robot_type
        robot_manifest = json.loads((reference / "manifest.json").read_text())
        if (
            self.ref.file_hash(reference / "manifest.json")
            != manifest["robot_reference_sha256"]
        ):
            raise ValueError("Robot reference manifest differs from preparation")
        # Validate every robot asset against the hashed reference manifest.
        expected = {item["path"]: item["sha256"] for item in robot_manifest["files"]}
        expected["README.md"] = (
            "6a2e08cfc88d7eec8fb585e53fc3e8aea9ccd051c7af672185003f4102e544a4"
        )
        actual = self.ref.tree_hashes(reference)
        if {k: v for k, v in actual.items() if k != "manifest.json"} != expected:
            raise ValueError("Robot reference payload changed")
        shutil.copytree(reference, self.inputs / "robot_reference")
        shutil.copyfile(
            self.ref.ROOT / "src/gpt6_eval/step.py", self.interface / "step.py"
        )
        self.ref.write_json(
            self.inputs / "protocol.json",
            {
                "revision": self.cfg["protocol_revision"],
                "accept_bare_action_arrays": True,
                "robot_reference_sha256": manifest["robot_reference_sha256"],
            },
        )
        self.ref.write_json(
            self.inputs / "robot.json",
            {
                "robot_type": robot_type,
                "public_state_dim": 46 if self.dual else 23,
                "action_dim": 44 if self.dual else 22,
                "control_hz": 30,
                "action_layout": "[r_xyz3,r_rotvec3,r_hand16,l_xyz3,l_rotvec3,l_hand16]"
                if self.dual
                else "[xyz3,rotvec3,hand16]",
                "state_layout": "[r_pose7,l_pose7,r_hand16,l_hand16]"
                if self.dual
                else "[pose7,hand16]",
                "quaternion_order_in_state": "wxyz",
                "robot_reference": "robot_reference/README.md",
            },
        )
        self.ref.write_json(
            self.private / "protocol.json",
            {
                "protocol": PROTOCOL,
                "profile": "rand-full-v2",
                "config": self.cfg,
                "package_sha256": self.expected_hash,
                "task": task.task_id,
                "seed": task.seed,
                "policy_runtime": "roborsi_embodied_codex_cli_external",
                "agent_config_fingerprint": self.config_fingerprint,
                "prompt_sha256": self.prompt_hashes,
                "fresh_sessions": True,
            },
        )

    def _verify_payload(self) -> None:
        self.ref.verify_package(
            self.package, self.task.task_id, self.cfg, self.expected_hash
        )
        if self.ref.tree_hashes(self.inputs / "task_knowledge") != self.ref.tree_hashes(
            self.package / "payload"
        ):
            raise ValueError("Copied frozen knowledge changed")

    def _save(self, observation: Observation) -> dict:
        import io

        import numpy as np
        from PIL import Image

        if self.task is None or observation.benchmark != "dexjoco":
            raise ValueError("Agent must reset before observing DexJoCo")
        if self.episode is None:
            self.episode = observation.episode_id
        if (
            observation.episode_id != self.episode
            or observation.step_index != self.next_step
        ):
            raise ValueError("Policy received a stale or foreign observation")
        step = observation.step_index
        if step and self.last_row is not None:
            self.ref.write_json(
                self.private / "steps" / f"{step:06d}.json",
                {
                    "action": self.last_row,
                    "phase": "startup_alignment"
                    if self.task.task_id == "click_mouse" and step <= 30
                    else "policy_" + self.phase,
                },
            )
        state = observation.data["state"]
        width = 46 if self.dual else 23
        if len(state) != width or not np.isfinite(state).all():
            raise ValueError("Invalid public state width")
        names = observation.data["camera_names"]
        if (
            len(names) != len(observation.media)
            or len(set(names)) != len(names)
            or set(names)
            != (
                {"random_camera", "wrist_left", "wrist_right"}
                if self.dual
                else {"random_camera", "wrist"}
            )
            or any(not name or not name.replace("_", "").isalnum() for name in names)
        ):
            raise ValueError("Unexpected official camera mapping")
        folder = self.inputs / "observations" / f"{step:06d}"
        folder.mkdir(parents=True, exist_ok=False)
        cameras = []
        for name, media in zip(names, observation.media, strict=True):
            payload = self.media.read(media, observation.episode_id, step)
            rgb = np.asarray(Image.open(io.BytesIO(payload)))
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError("Expected RGB camera image")
            path = folder / f"{name}.png"
            path.write_bytes(payload)
            cameras.append(
                {
                    "name": name,
                    "path": str(path.relative_to(self.inputs)),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                }
            )
        limit = self.standard_limit if self.phase == "initial" else self.task.max_steps
        public = {
            "prompt": self.task.instruction,
            "state": state,
            "step": step,
            "decision": self.decision,
            "remaining_steps": limit - step,
            "cameras": cameras,
            "action_format": "rotvec_22_or_44",
        }
        self.ref.write_json(folder / "observation.json", public)
        self.last_saved_step = step
        return public

    def _start(self) -> None:
        self.session = self.factory(self, self.phase, self.prompts[self.phase])

    def _finish(self) -> None:
        if self.session is not None:
            session = self.session
            record = None
            try:
                record = session.close()
            finally:
                if record is None:
                    record = getattr(session, "record", None)
                if record is not None and record not in self.records:
                    self.records.append(record)
                self.ref.write_json(self.private / "sessions.json", self.records)
            self.session = None
        else:
            self.ref.write_json(self.private / "sessions.json", self.records)

    def act(self, observation: Observation):
        """Drain complete policy chunks before publishing another decision."""
        if self.closed:
            raise ValueError("DexJoCo policy is closed")
        public = self._save(observation)
        step = observation.step_index
        if self.task.task_id == "click_mouse" and step < 30:
            row = list(ALIGNMENT)
        else:
            if step == self.standard_limit and self.phase == "initial":
                if self.queue:
                    raise RuntimeError("Policy chunk crossed the original budget")
                self._finish()
                self._verify_payload()
                transport = self.private / "action-transport"
                for name in ("request.json", "request.tmp", "submission.lock"):
                    (transport / name).unlink(missing_ok=True)
                history_sha = self.ref._export_first_session_history(
                    self.inputs, self.private, self.standard_limit
                )
                self.ref.write_json(
                    self.inputs / "continuation.json",
                    {
                        "phase": "continuation",
                        "standard_step_limit": self.standard_limit,
                        "additional_step_budget": self.task.max_steps
                        - self.standard_limit,
                        "public_history": "first_session_history.jsonl",
                        "history_sha256": history_sha,
                    },
                )
                (self.inputs / "episode_done.json").unlink(missing_ok=True)
                self.phase = "continuation"
                public.update(
                    remaining_steps=self.task.max_steps - step, continuation=True
                )
            if not self.queue:
                self.ref.write_json(self.inputs / "observation.json", public)
                self.ref.write_json(
                    self.private / "decisions" / f"{self.decision:06d}.json", public
                )
                self.ref._release_completed_submission_lock(
                    self.private / "action-transport"
                )
                if self.session is None:
                    self._start()
                try:
                    actions, evidence = self.session.actions(
                        self.decision, public["remaining_steps"]
                    )
                    # Also enforce the contract for alternate session implementations.
                    actions = self.ref.validate_actions(
                        {"actions": actions}, self.dual, public["remaining_steps"], 30
                    )
                except (self.ref.PolicyError, ValueError) as exc:
                    self.ref.write_json(
                        self.private / "policy_failure.json", {"message": str(exc)}
                    )
                    return EpisodeStop(
                        benchmark="dexjoco",
                        episode_id=self.episode,
                        step_index=step,
                        reason="policy_failure",
                    )
                self.ref.write_json(
                    self.private / "actions" / f"{self.decision:06d}.json",
                    {
                        "start_step": step,
                        "decision": self.decision,
                        "session_phase": self.phase,
                        "actions": actions.tolist(),
                        **evidence,
                    },
                )
                self.queue.extend(actions.tolist())
                self.decision += 1
            row = self.queue.popleft()
        self.last_row = row
        self.next_step = step + 1
        return Action(
            benchmark="dexjoco",
            episode_id=self.episode,
            step_index=step,
            data={"vector": row},
        )

    def observe_end(self, result: StepResult) -> None:
        """Archive the final public frame; reveal only generic episode_done."""
        try:
            if (
                result.observation.step_index == self.next_step
                and self.last_saved_step != self.next_step
            ):
                self._save(result.observation)
        finally:
            self.close()

    def close(self) -> None:
        """Stop sessions on every completion, error or cancellation path."""
        if self.closed:
            return
        try:
            self.ref.write_json(
                self.inputs / "episode_done.json", {"episode_done": True}
            )
        finally:
            try:
                self._finish()
            finally:
                self.queue.clear()
        self.closed = True
