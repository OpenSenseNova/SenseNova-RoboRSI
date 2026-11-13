# Copyright 2026 The RoboRSI Contributors.
# Licensed under the Apache License, Version 2.0.

"""Connect the historical file protocol to RoboRSI's shared CLI backend."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from roborsi.evaluation.media import CaseMediaChannel
from roborsi.evaluation.messages import (
    Action,
    EpisodeStop,
    Observation,
    StepResult,
    Task,
)
from roborsi.planner.codex_cli_config import CodexCliConfig
from roborsi.utils.config import require_project_env

from ...contracts import TASKS, file_sha256, validate_task
from ...frozen_reference import REFERENCE_SHA256, load_reference
from ...icl import (
    MODEL,
    PROTOCOL,
    SETTING,
    extract_verified,
    package_entry,
    verify_package,
)
from .cli_episode import episode_type, historical_cli_config

CLI_SHA256 = "3188814c35471432d4123203e0eb38e5bddc60226e3d7ddf0e59e649ea140022"
ROBOT_KEYS = {
    "robot_key",
    "active_joint_names",
    "position_limits",
    "action_semantics",
    "joint_units",
    "physics_hz",
    "policy_hz",
}


class FrozenPolicyAgent:
    """Retain the original Episode, including its isolation and action mailbox."""

    def __init__(
        self,
        directory: Path,
        *,
        reference: Path,
        package: Path,
        media_dir: Path,
        binary: Path,
        auth: Path,
        episode_factory=None,
        prompt: str | None = None,
        cli_config: CodexCliConfig | None = None,
        prompt_identity: dict | None = None,
    ) -> None:
        self.directory, self.package = directory, package
        self.ref = load_reference(reference)
        self.media = CaseMediaChannel(media_dir)
        self.binary, self.auth = binary, auth
        self.factory = episode_factory or episode_type(
            self.ref, cli_config or historical_cli_config(self.ref.bridge.isolation)
        )
        self.prompt = prompt if prompt is not None else historical_prompt(self.ref)
        self.prompt_identity = prompt_identity or {}
        # Serialize the historical mailbox/encoder across transient HTTP
        # request threads. EmbodiedEpisode separately owns the model worker.
        self.owner = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="bench2dex-policy"
        )
        self.task = None
        self.episode = None
        self.episode_id = None
        self.expected_step = 0
        self.remaining_chunk = 0
        self.robot = None
        self.closed = False
        self.closing = False
        self.close_lock = threading.Lock()
        self.ended = False
        self.policy_failed = False

    def reset(self, task: Task) -> None:
        """Verify the task and its immutable package before creating a session."""
        if self.task is not None or self.closing:
            raise ValueError("Agent owns one episode")
        validate_task(task)
        if self.package.is_file():
            self.directory.mkdir(parents=True, exist_ok=True)
            target = self.directory / "icl-package"
            extract_verified(self.package, target, package_entry(task.task_id))
            self.package = target
        verify_package(self.package, task)
        self.task = task

    def _raw(self, observation: Observation) -> dict:
        if (
            self.task is None
            or (observation.benchmark, observation.step_index)
            != (
                "bench2dex",
                self.expected_step,
            )
            or (
                self.episode_id is not None
                and observation.episode_id != self.episode_id
            )
        ):
            raise ValueError("observation identity or order mismatch")
        if set(observation.data) != {"sample", "robot"}:
            raise ValueError("unexpected public observation fields")
        sample, robot = observation.data["sample"], observation.data["robot"]
        if (
            set(sample)
            != {
                "instruction",
                "qpos",
                "policy_step",
                "physics_step",
                "remaining_policy_steps",
                "cameras",
                "camera_calibration",
            }
            or sample["instruction"] != self.task.instruction
        ):
            raise ValueError("unexpected sensor fields or instruction")
        if (
            sample["policy_step"],
            sample["physics_step"],
            sample["remaining_policy_steps"],
        ) != (
            self.expected_step,
            3 * self.expected_step,
            self.task.max_steps - self.expected_step,
        ):
            raise ValueError("policy clock or remaining budget mismatch")
        if set(robot) != ROBOT_KEYS | {
            "kinematics",
            "inactive_joint_positions",
            "resolved_joint_units",
        }:
            raise ValueError("unexpected robot fields")
        if (
            robot["robot_key"] != self.task.native["robot_key"]
            or len(robot["active_joint_names"]) != TASKS[self.task.task_id]["dof"]
            or (robot["physics_hz"], robot["policy_hz"]) != (60, 20)
            or self.robot is not None
            and self.robot != robot
        ):
            raise ValueError("robot contract changed")
        if len(sample["cameras"]) != 4 or len(observation.media) != 4:
            raise ValueError("expected four ordered RGB cameras")
        images = {}
        for name, camera, reference in zip(
            self.ref.core.CAMERAS[PROTOCOL["observation_profile"]],
            sample["cameras"],
            observation.media,
        ):
            if set(camera) != {"name", "path", "sha256", "rgb_sha256"} or (
                camera["name"],
                camera["path"],
                reference.mime_type,
            ) != (
                name,
                reference.ref,
                "image/png",
            ):
                raise ValueError("camera order/path/format mismatch")
            payload = self.media.read(
                reference, observation.episode_id, observation.step_index
            )
            if hashlib.sha256(payload).hexdigest() != camera["sha256"]:
                raise ValueError("camera PNG hash mismatch")
            with Image.open(io.BytesIO(payload)) as image:
                if image.mode != "RGB" or image.size != (640, 480):
                    raise ValueError("invalid camera image")
                rgb = np.array(image)
            if (
                hashlib.sha256(rgb.tobytes()).hexdigest() != camera["rgb_sha256"]
                or self.ref.throughput.encode_rgb(rgb) != payload
            ):
                raise ValueError("camera pixels or historical PNG encoding changed")
            images[name] = {"rgb": rgb}
        raw = {
            "language": sample["instruction"],
            "joint_action": {"vector": np.asarray(sample["qpos"], dtype=np.float32)},
            "observation": images,
            "camera_calibration": sample["camera_calibration"],
            "robot_metadata": {k: robot[k] for k in ROBOT_KEYS},
        }
        public, _ = self.ref.core.sanitize_observation(
            raw, PROTOCOL["observation_profile"], TASKS[self.task.task_id]["dof"]
        )
        calibration = self.ref.calibration.validate_calibration(
            raw["camera_calibration"],
            self.ref.core.CAMERAS[PROTOCOL["observation_profile"]],
        )
        if (
            public["qpos"] != sample["qpos"]
            or calibration != sample["camera_calibration"]
        ):
            raise ValueError("public numerical representation changed")
        self.episode_id = observation.episode_id
        self.robot = json.loads(json.dumps(robot))
        return raw

    def _create_episode(self) -> None:
        self.episode = self.factory(
            self.directory / "episode",
            dict(PROTOCOL),
            self.task.instruction,
            self.task.native["robot_key"],
            self.robot["kinematics"],
            self.task.max_steps,
            self.binary,
            self.auth,
            self.prompt,
        )
        self.ref.knowledge.attach_package(
            self.package,
            package_entry(self.task.task_id)["manifest_sha256"],
            int(self.task.task_id),
            PROTOCOL,
            self.episode.inputs,
            self.episode.private,
        )
        self.ref.core.write_json(
            self.directory / "provenance.json",
            {
                "setting": SETTING,
                "historical_setting": "task_icl_v3",
                "reference_sha256": REFERENCE_SHA256,
                "task_id": self.task.task_id,
                "icl_manifest_sha256": package_entry(self.task.task_id)[
                    "manifest_sha256"
                ],
                "model": MODEL,
                "reasoning_effort": "max",
                "backend": "codex_cli",
                "entry_point": "EmbodiedAgent.start_episode",
                "tool_protocol": "external",
                "effective_prompt_sha256": hashlib.sha256(
                    self.prompt.encode()
                ).hexdigest(),
                "historical_prompt_equal": self.prompt == historical_prompt(self.ref),
                "prompt_configuration": self.prompt_identity,
            },
        )

    def act(self, observation: Observation) -> Action | EpisodeStop:
        """Request inference only at the original full-chunk boundaries."""
        if self.closing or self.ended or self.policy_failed:
            raise ValueError("Agent episode has ended")
        raw = self._raw(observation)
        if self.episode is None:
            self.owner.submit(self._create_episode).result()
        if self.closing:
            raise RuntimeError("Agent was closed during startup")
        if self.remaining_chunk:
            self.owner.submit(self.episode.observe, raw).result()
            self.remaining_chunk -= 1
            data = {"continue": True}
        else:
            try:
                actions = self.owner.submit(self.episode.action, raw).result()
            except self.ref.bridge.PolicyError as exc:
                self.policy_failed = True
                self.ref.core.write_json(
                    self.episode.private / "policy_failure.json", {"error": str(exc)}
                )
                return EpisodeStop(
                    benchmark="bench2dex",
                    episode_id=self.episode_id,
                    step_index=self.expected_step,
                    reason="policy_failure",
                )
            self.remaining_chunk = len(actions) - 1
            data = {"actions": actions.tolist()}
        self.expected_step += 1
        return Action(
            benchmark="bench2dex",
            episode_id=self.episode_id,
            step_index=observation.step_index,
            data=data,
        )

    def observe_end(self, result: StepResult) -> None:
        """Let the original cleanup publish only generic episode_done."""
        obs = result.observation
        if (
            not (result.terminated or result.truncated)
            or (obs.benchmark, obs.episode_id, obs.step_index)
            != ("bench2dex", self.episode_id, self.expected_step)
            or obs.data != {"episode_done": True}
            or obs.media
            or result.reward is not None
            or result.native
            or result.termination_reason != "episode_ended"
        ):
            raise ValueError("terminal identity or public information mismatch")
        self.ended = True
        self.close()

    def close(self) -> None:
        """Wake any mailbox wait before queuing cleanup on its owning thread."""
        self.closing = True
        if self.episode is not None:
            self.episode.stopped.set()
        # Creation may still be queued or executing on the owner. Resolve the
        # episode there so cancellation during startup also receives cleanup.
        with self.close_lock:
            if self.closed:
                return

            def finish():
                if self.episode is not None:
                    self.episode.finish()

            try:
                self.owner.submit(finish).result()
            finally:
                if self.episode is None or self.episode.finished:
                    self.owner.shutdown(wait=True)
                    self.closed = True


def historical_prompt(reference) -> str:
    """Return the exact original initial prompt for baseline comparison."""
    return (
        (reference.root / "prompts/policy_icl.md").read_text()
        + "\n"
        + (reference.root / "prompts/calibrated_deployment.md").read_text()
    )


def configured_policy(config, reference):
    """Load actual prompt/Skill bytes and reject unsupported experiment settings."""
    options = dict(config.document.get("agent", {}))
    inline = options.pop("system_prompt", None)
    cli = CodexCliConfig.from_dict(options.pop("codex_cli", {}))
    if (
        options
        != {
            "planner": "codex_cli",
            "model": MODEL,
            "reasoning_effort": "max",
            "max_turns": None,
            "max_tokens": None,
            "planner_timeout_s": 0,
            "action_timeout_s": None,
        }
        or config.parameters
        or config.document.get("runtime")
        or cli != historical_cli_config(reference.bridge.isolation, cli.executable)
    ):
        raise ValueError(
            "Bench2Dex requires the pinned Astra/max CLI configuration; "
            "only prompt_file, system_prompt, skills and executable may vary"
        )
    prompt_path = config.document.get("prompt_file")
    if prompt_path is not None:
        prompt = (config.path.parent / prompt_path).read_text()
        source = "prompt_file"
    elif inline is not None:
        prompt, source = inline, "agent.system_prompt"
    else:
        raise ValueError("declare prompt_file or agent.system_prompt explicitly")
    identity = {
        "source": source,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "skills": [],
    }
    # A configured Skill is full initial text, in declared order. ICL remains
    # the original on-demand task_knowledge files, not a preloaded Skill.
    for item in config.document.get("skills", []):
        text = (config.path.parent / item).read_text()
        if not text.strip():
            raise ValueError("configured skills must contain nonempty text")
        prompt += "\n\n" + text
        identity["skills"].append(hashlib.sha256(text.encode()).hexdigest())
    return prompt, cli, identity


def make_agent(case_dir: Path, *, config) -> FrozenPolicyAgent:
    """Use the shared CLI planner with the original isolated file protocol."""
    config.verify()
    if set(config.resources) != {
        "reference",
        "reference_license",
        "icl_package",
        "protocol",
        "native_protocol",
        "task_catalog",
        "icl_catalog",
    }:
        raise ValueError("declare the complete frozen reference and ICL resources")
    reference = load_reference(config.resources["reference"])
    prompt, cli, identity = configured_policy(config, reference)
    from codex_cli_bin import bundled_codex_path

    selected = os.environ.get("BENCH2DEX_CODEX_BIN")
    if selected is None:
        selected = (
            bundled_codex_path()
            if cli.executable == "codex"
            else (shutil.which(cli.executable) or config.path.parent / cli.executable)
        )
    binary = Path(selected).resolve()
    if (
        file_sha256(binary) != CLI_SHA256
        or subprocess.check_output(
            [str(binary), "--version"],
            text=True,
        ).strip()
        != PROTOCOL["codex_version"]
    ):
        raise ValueError("install the pinned Linux x86_64 Codex CLI 0.154.0")
    if file_sha256(binary.with_name("codex-code-mode-host")) != (
        "0c57be435e73b70d9106c850d751cd259a7f04da958a453d7ef59090d82b70f1"
    ):
        raise ValueError("install the pinned companion codex-code-mode-host")
    auth = Path(
        os.environ.get("BENCH2DEX_CODEX_AUTH", str(Path.home() / ".codex/auth.json"))
    ).resolve()
    if not auth.is_file():
        raise ValueError(
            "authenticate locally and set BENCH2DEX_CODEX_AUTH to its auth file"
        )
    return FrozenPolicyAgent(
        case_dir / "bench2dex",
        reference=config.resources["reference"],
        package=config.resources["icl_package"],
        media_dir=Path(require_project_env("ROBORSI_MEDIA_DIR")),
        binary=binary,
        auth=auth,
        prompt=prompt,
        cli_config=cli,
        prompt_identity=identity,
    )
