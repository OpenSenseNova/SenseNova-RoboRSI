"""Official XPolicyLab boundary for the original RoboDojo hybrid Agent."""

from __future__ import annotations

import atexit
import copy
import fcntl
import json
import logging
import os
import queue
import re
import sys
import tempfile
import time
import uuid
import weakref
from pathlib import Path
from typing import Any

import numpy as np

from .model_template import ModelTemplate
from .checkpoint_resolver import resolve_checkpoint_root
from robots.robodojo.contracts import get_robot_action_dim_info

from .runtime_config import (
    POLICY_DIR, resolve_eef_execution_backend, load_agent_config, resolve_adaptive_steps,
    resolve_base_url, resolve_model, resolve_vla_config, submission,
)
from .tool_runtime import resolve_tool_python

_LIVE_MODELS = weakref.WeakSet()
ACTION_POLL_SECONDS = 30.0
logger = logging.getLogger(__name__)


class _SharedVLA:
    """Non-owning handle for a node-local shared pi0.5 websocket service."""

    def __init__(self, port: int):
        self.port = port

    def close(self) -> None:
        return None


def close_models() -> None:
    """Stop all episode and VLA owners before Python joins background threads."""
    for model in list(_LIVE_MODELS):
        model.shutdown()


class Model(ModelTemplate):
    """One original conversation per episode; one acknowledged joint step per call."""

    def __init__(self, model_cfg: dict[str, Any]):
        super().__init__()
        if not os.environ.get("CODEX_API_KEY", "").strip():
            raise ValueError(
                "CODEX_API_KEY is required; set it in the runtime environment"
            )
        if model_cfg.get("bench_name") != "RoboDojo":
            raise ValueError("SenseNova-RoboRSI supports only bench_name=RoboDojo")
        if model_cfg.get("env_cfg_type") != "arx_x5":
            raise ValueError("The original hybrid supports only env_cfg_type=arx_x5")
        if model_cfg.get("action_type") != "joint":
            raise ValueError("The original hybrid FK/DLS emits action_type=joint")
        if model_cfg.get("eval_batch", False):
            raise ValueError("SenseNova-RoboRSI requires eval_batch=false")
        resolve_base_url()
        resolve_model()
        self._adaptive_steps = resolve_adaptive_steps()
        # Validate the independent original tool payload before materializing
        # ICL or starting the expensive VLA service. Never substitute the host.
        self._tool_python = resolve_tool_python(POLICY_DIR)
        self.model_cfg = dict(model_cfg)
        from .agent_track import validate_mode
        environment_mode = os.environ.get("ROBORSI_SKILL_LIBRARY_MODE")
        if (environment_mode and "skill_library_mode" in self.model_cfg
                and self.model_cfg["skill_library_mode"] != environment_mode):
            raise ValueError("Environment and deployment skill library modes differ")
        self.model_cfg["skill_library_mode"] = validate_mode(
            self.model_cfg.get("skill_library_mode", environment_mode or "general_only"))
        if self.model_cfg["skill_library_mode"] == "general_only":
            resolve_eef_execution_backend(self.model_cfg, self.model_cfg.get("task_name", ""))
            for key, environment_key in [
                ("official_description_bundle", "ROBORSI_DESCRIPTION_BUNDLE"),
                ("official_description_manifest_sha256", "ROBORSI_DESCRIPTION_MANIFEST_SHA256"),
            ]:
                value = os.environ.get(environment_key)
                if value and self.model_cfg.get(key) not in (None, value):
                    raise ValueError("Environment and deployment description pins differ")
                if value: self.model_cfg[key] = value
        if self.model_cfg.get("agent_exit_policy", "protocol_error") not in {"native_finish", "protocol_error"}:
            raise ValueError("agent_exit_policy must be native_finish or protocol_error")
        vla = resolve_vla_config(self.model_cfg)
        self.model_cfg.update(vla_backend=vla["backend"], checkpoint_path=vla["checkpoint_path"])
        self.action_type = "joint"
        self.robot_action_dim_info = get_robot_action_dim_info("arx_x5")
        dimensions = self.robot_action_dim_info
        if dimensions["arm_dim"] != [6, 6] or dimensions["ee_dim"] != [1, 1]:
            raise ValueError(
                "The pinned ARX-X5 hybrid metadata requires six joints and one gripper per arm"
            )
        # Use the installed shared framework; the historical vendor is provenance.
        from robots.robodojo.eef_types import ActionSpace

        if self.model_cfg["vla_backend"] != "disabled":
            from .pi05_source import load_openpi_client

            load_openpi_client(POLICY_DIR)

        channels = []
        for index, arm in enumerate(("left", "right")):
            channels.extend(
                (
                    (f"{arm}_arm_joint_state", dimensions["arm_dim"][index]),
                    (f"{arm}_ee_joint_state", dimensions["ee_dim"][index]),
                )
            )
        self._action_space = ActionSpace(channels)
        self._agent_config = load_agent_config(self.model_cfg)
        self._temporary = self._connector = self._icl = self._obs = self._pending = None
        self._trajectory_directory = None
        self._trajectory_id = None
        self._vla = None
        self._simate_observer = None
        self._step = 0
        self._resume = None
        self._unfinished_native = None
        self._resume_request_path = None
        self._resume_start_lock = None
        self._deferred_controller = None
        self._cleanup_issue = None
        self._awaiting_feedback = self._ended = False
        # The server constructs Model before opening its websocket port. Load
        # VLA weights here so cold start does not consume a native action RPC.
        self._ensure_vla()
        _LIVE_MODELS.add(self)
        atexit.register(self.shutdown)

    def reset(self) -> None:
        """End the old conversation; retain only the owned immutable VLA service."""
        if getattr(self, "_deferred_controller", None) is not None:
            raise RuntimeError(
                "Previous episode cleanup is deferred; restart the policy server before reset"
            )
        self.close()
        if os.environ.get("ROBORSI_RESUME_MANIFEST"):
            raise ValueError(
                "Legacy restore-receipt manifests do not execute physical replay; "
                "prepare a verified ROBORSI_RESUME_REQUEST instead"
            )
        task = self.model_cfg.get("task_name")
        if not isinstance(task, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", task):
            raise ValueError("An exact RoboDojo task name is required")
        from .hybrid_resume import (
            declared_identity,
            load_resume,
            process_identity,
            write_json,
        )

        identity = declared_identity(
            self.model_cfg, os.environ.get("ROBODOJO_RUN_ID", "")
        )
        self._resume = None
        self._resume_request_path = None
        resume_path = os.environ.get("ROBORSI_RESUME_REQUEST")
        if resume_path:
            request_path = Path(resume_path).absolute()
            from .hybrid_resume import digest, read_json

            request_sha = os.environ.get("ROBORSI_RESUME_REQUEST_SHA256")
            if request_sha and digest(request_path) != request_sha:
                raise ValueError("Resume request changed after launcher validation")
            requested = read_json(request_path).get("identity", {})
            for key, expected in identity.items():
                if expected is not None and requested.get(key) != expected:
                    raise ValueError(f"Resume request differs from deployment: {key}")
            identity = requested
            self._resume = load_resume(request_path, identity)
            self._resume_request_path = request_path
            if request_path.with_suffix(request_path.suffix + ".claimed").exists():
                raise FileExistsError("Resume request already started a controller")
        # Each attempt keeps a separate output directory. The wire identity
        # belongs to the logical episode and survives process/session changes.
        self._episode_id = uuid.uuid4().hex
        self._trajectory_id = (
            self._resume["trajectory_id"] if self._resume else self._episode_id
        )
        root = self.model_cfg.get("trajectory_dir")
        root = Path(root).expanduser() if root else POLICY_DIR / "outputs/trajectories"
        if not root.is_absolute():
            root = POLICY_DIR / root
        seed = int(self.model_cfg.get("seed") or 0)
        self._trajectory_directory = root / task / f"seed-{seed}" / self._episode_id
        self._trajectory_directory.mkdir(parents=True, mode=0o700)
        write_json(self._trajectory_directory / "trajectory.json", {
            "schema_version": 1, "task": task, "seed": seed,
            "episode_id": self._episode_id, "trajectory_id": self._trajectory_id,
        })
        self._temporary = tempfile.TemporaryDirectory(
            prefix="runtime-", dir=self._trajectory_directory
        )
        # Preserve interrupted episodes even when close() cannot complete.
        self._temporary._finalizer.detach()
        self._directory = Path(self._temporary.name)
        self._agent_config = load_agent_config(
            self.model_cfg, episode_dir=self._directory, trajectory_id=self._trajectory_id
        )
        self._step = 0
        owner = {
            "format": "xpolicylab_episode_owner_v1",
            "identity": identity,
            "episode_id": self._episode_id,
            "trajectory_id": self._trajectory_id,
            "pid": os.getpid(),
            "process_start": process_identity(os.getpid()),
            "started_at": time.time(),
            "closed": False,
        }
        if self._resume:
            owner.update(
                source=self._resume["source"],
                recovery_state=self._resume["recovery_state"],
                recovery_count=self._resume["recovery_count"],
            )
        write_json(self._directory / "episode-owner.json", owner)
        self._obs = self._pending = None
        self._awaiting_feedback = self._ended = False

    def _reserve_resume(self) -> None:
        """Exclude other starters without consuming a request during initialization."""
        if self._resume is None or getattr(self, "_resume_start_lock", None) is not None:
            return
        request_path = self._resume_request_path
        # Keep this inode: unlinking a lock file permits competing inode locks.
        descriptor = os.open(
            request_path.with_suffix(request_path.suffix + ".startup.lock"),
            os.O_CREAT | os.O_RDWR | os.O_CLOEXEC,
            0o600,
        )
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise FileExistsError("Resume request has an active controller starter") from error
            if request_path.with_suffix(request_path.suffix + ".claimed").exists():
                raise FileExistsError("Resume request already started a controller")
        except BaseException:
            os.close(descriptor)
            raise
        self._resume_start_lock = descriptor

    def _claim_resume(self) -> None:
        """Consume only after controller initialization, just before execution."""
        if self._resume is None:
            return
        self._reserve_resume()
        from .hybrid_resume import digest, process_identity

        request_path = self._resume_request_path
        expected = os.environ.get("ROBORSI_RESUME_REQUEST_SHA256")
        if expected and digest(request_path) != expected:
            raise ValueError("Resume request changed before controller startup")
        with request_path.with_suffix(request_path.suffix + ".claimed").open("x") as claim:
            json.dump(
                {"pid": os.getpid(), "process_start": process_identity(os.getpid()),
                 "attempt": str(self._directory), "controller_start_requested": True},
                claim,
            )

    def update_obs(self, obs: dict[str, Any]) -> None:
        """Accept the standard server's decoded RGB; acknowledge only at get_action."""
        if self._temporary is None or self._ended:
            raise ValueError("Call reset before updating an episode observation")
        if not isinstance(obs, dict) or not isinstance(obs.get("vision"), dict):
            raise ValueError("Observation requires a vision mapping")
        from .agent_track import official_observation, validate_mode
        mode = validate_mode(self.model_cfg.get("skill_library_mode", "general_only"))
        if mode == "general_only":
            obs = official_observation(
                obs, task=self.model_cfg.get("task_name"),
                bundle=self.model_cfg.get("official_description_bundle"),
                manifest_sha256=self.model_cfg.get("official_description_manifest_sha256"))
        instruction = obs.get("instruction") or obs.get("instructions")
        if (
            isinstance(instruction, (list, tuple))
            and instruction
            and all(isinstance(value, str) and value.strip() for value in instruction)
        ):
            instruction = "\n".join(instruction)
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Observation requires instruction (or instructions)")
        if not isinstance(obs.get("state"), dict):
            raise ValueError("Observation requires robot state")
        for name in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
            color = obs["vision"].get(name, {}).get("color")
            if not isinstance(color, np.ndarray) or color.dtype != np.uint8:
                raise ValueError(
                    "The policy server must supply three decoded uint8 RGB colors"
                )
            if color.ndim != 3 or color.shape[-1] != 3:
                raise ValueError("Camera colors must have shape H,W,3")
        from .hybrid_resume import read_json, validate_native_identity, write_json

        owner_path = self._directory / "episode-owner.json"
        owner = read_json(owner_path)
        native = obs.get("roborsi_native_identity")
        if native is not None or self._resume or os.environ.get("ROBORSI_REQUIRE_NATIVE_IDENTITY") == "1":
            expected_step = self._step + int(
                self._pending is not None and getattr(self._pending, "kind", "joint") in ("joint", "bounded_curobo")
            )
            source_native = self._resume["native_identity"] if self._resume else None
            native = validate_native_identity(
                native, owner["identity"], step=expected_step,
                step_limit=(source_native or owner.get("native_identity", {})).get("step_limit"),
                scene_sha256=(source_native or owner.get("native_identity", {})).get("scene_sha256"),
            )
            unfinished = getattr(self, "_unfinished_native", None)
            if not self._resume and unfinished and all(
                native.get(key) == unfinished.get(key)
                for key in ("task", "eval_seed", "layout", "run_id", "scene_sha256")
            ):
                raise ValueError(
                    "Interrupted native episode requires a prepared resume request; "
                    "refusing a fresh workspace and renewed budget for the same scene"
                )
            self._unfinished_native = None
            if owner["identity"].get("layout") is None:
                owner["identity"]["layout"] = native["layout"]
            if owner["identity"].get("run_id") is None:
                owner["identity"]["run_id"] = native["run_id"]
            owner["native_identity"] = native
            write_json(owner_path, owner)
        if mode == "general_only":
            write_json(self._directory / "official-description-source.json",
                       obs["_roborsi_official_description_source"])
        self._obs = {**obs, "instruction": instruction}
        if mode == "general_only" and os.environ.get("ROBORSI_INITIALIZATION_READY_FILE"):
            from .agent_track import publish_initialization_ready
            publish_initialization_ready(obs)
        if self.model_cfg.get("vla_backend") == "simate_beta":
            from .simate_bridge import SimateObserver

            observed_step = self._step + int(
                self._pending is not None and getattr(self._pending, "kind", "joint") in ("joint", "bounded_curobo")
            )
            if self._simate_observer is None:
                self._simate_observer = SimateObserver(self._ensure_vla(), self._episode_id)
            self._simate_observer.observe(self._obs, observed_step)

    def _ensure_vla(self) -> int | None:
        if self.model_cfg.get("vla_backend") == "disabled":
            return None
        if self._vla is None:
            if os.environ.get("ROBORSI_VLA_POOL_URL"):
                if self.model_cfg.get("vla_backend", "pi05") != "pi05":
                    raise ValueError("Configured VLA pool requires the Pi0.5 backend")
                from .vla_pool import LABEL_PORT, install_client
                install_client(POLICY_DIR)
                self._vla = _SharedVLA(LABEL_PORT)
                return LABEL_PORT
            shared_port = os.environ.get("ROBORSI_SHARED_VLA_PORT")
            if shared_port:
                try:
                    port = int(shared_port)
                except ValueError as error:
                    raise ValueError(
                        "ROBORSI_SHARED_VLA_PORT must be an integer"
                    ) from error
                if not 1 <= port <= 65535:
                    raise ValueError(
                        "ROBORSI_SHARED_VLA_PORT is outside the TCP port range"
                    )
                from .vla_identity import check_shared_vla
                receipt = os.environ.get("ROBORSI_SHARED_VLA_RECEIPT")
                check_shared_vla(port, backend=self.model_cfg.get("vla_backend", "pi05"),
                                 receipt_path=Path(receipt) if receipt else None, policy_dir=POLICY_DIR)
                self._vla = _SharedVLA(port)
                return port
            checkpoint = resolve_checkpoint_root(
                self.model_cfg, POLICY_DIR / "checkpoints", policy_dir=POLICY_DIR
            )
            # Keep each owned service's diagnostics after failure or shutdown.
            # mkdtemp creates a private directory without a deleting finalizer.
            log_root = POLICY_DIR / "outputs/vla"
            log_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            output_dir = Path(tempfile.mkdtemp(prefix="run-", dir=log_root)).resolve()
            options = dict(
                checkpoint_dir=checkpoint, output_dir=output_dir,
                gpu_id=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                port=0, policy_dir=POLICY_DIR,
            )
            if self.model_cfg.get("vla_backend") == "simate_beta":
                from .simate_runtime import SimateServer

                server = SimateServer(seed=int(self.model_cfg.get("seed") or 0), **options)
            else:
                from .pi05_runtime import Pi05Server

                server = Pi05Server(data_home=POLICY_DIR / "runtime/pi05-data", **options)
            try:
                self._vla = server.start()
            except BaseException:
                server.close()
                raise
        return self._vla.port

    def _start(self) -> None:
        from .hybrid_agent import HybridRunContext
        from .skillpack_agent import SkillpackController
        from .execution_profile import extend_reference, load_skillpack_profile, prepare_skillpack_context
        from .hybrid_backend import HybridBackend, load_reference_source

        from .agent_track import validate_resume_mode
        validate_resume_mode(self.model_cfg.get("skill_library_mode", "general_only"),
                             self._resume["config"] if self._resume else None)
        pins = submission()
        profile = pins["hybrid"]
        reference = load_reference_source(
            POLICY_DIR / profile["source_path"],
            POLICY_DIR / profile["manifest_path"],
            profile["manifest_sha256"],
        )
        reference = extend_reference(reference)
        self._icl = prepare_skillpack_context(
            POLICY_DIR, self.model_cfg["task_name"], self._directory,
            vla_backend=self.model_cfg.get("vla_backend", "pi05"),
            native_identity=self._obs.get("roborsi_native_identity"),
            skill_library_mode=self.model_cfg.get("skill_library_mode", "general_only"),
        )
        base_task = self._icl.source_identity["mapping"].get("base_task") or self.model_cfg["task_name"]
        config_path = (
            POLICY_DIR
            / profile["profile_path"]
            / "resources/fresh-configs"
            / f"{base_task}-s0.json"
        )
        cfg = json.loads(config_path.read_text())
        cfg = copy.deepcopy(cfg)
        cfg["model"] = self._agent_config.document["agent"]["model"]
        cfg["reasoning_effort"] = self._agent_config.document["agent"]["reasoning_effort"]
        cfg["episode_seconds"] = self.model_cfg.get("episode_seconds")
        if cfg["episode_seconds"] is not None and (
            type(cfg["episode_seconds"]) is not int or cfg["episode_seconds"] <= 0
        ):
            raise ValueError("episode_seconds must be null or a positive integer")
        cfg.pop("continue_from_run", None)
        cfg["options"]["task"] = self.model_cfg["task_name"]
        cfg["options"]["skill_library_mode"] = self.model_cfg.get("skill_library_mode", "general_only")
        from .runtime_config import PI05_BASE_PROMPT_SHA256
        cfg["options"]["vla_backend"] = self.model_cfg.get("vla_backend", "pi05")
        cfg["options"]["vla_base_prompt_sha256"] = (PI05_BASE_PROMPT_SHA256
            if self.model_cfg.get("vla_backend", "pi05") == "pi05" else None)
        cfg["options"].update(self._adaptive_steps)
        default_backend = (
            self._resume["config"]["options"].get("eef_execution_backend", "legacy_dls")
            if self._resume else "bounded_curobo"
        )
        eef_backend = resolve_eef_execution_backend({"eef_execution_backend": default_backend, **self.model_cfg}, self.model_cfg["task_name"])
        if eef_backend not in {"legacy_dls", "bounded_curobo"}:
            raise ValueError("eef_execution_backend must be legacy_dls or bounded_curobo")
        cfg["options"]["eef_execution_backend"] = eef_backend
        native_limit = (self._obs.get("roborsi_native_identity") or {}).get("step_limit")
        requested_limit = self.model_cfg.get("native_step_limit", native_limit or cfg["options"]["review_window_steps"])
        if type(requested_limit) is not int or requested_limit <= 0:
            raise ValueError("native_step_limit must be a positive integer")
        if native_limit is not None and requested_limit != native_limit:
            raise ValueError("Configured native step limit differs from the actual episode")
        previous_exit_policy = (self._resume["config"]["options"].get("agent_exit_policy", "protocol_error")
                                if self._resume else "protocol_error")
        exit_policy = self.model_cfg.get("agent_exit_policy", previous_exit_policy)
        if self._resume and exit_policy != previous_exit_policy:
            raise ValueError("agent_exit_policy cannot change while resuming an episode")
        cfg["options"].update(codex_compact_tokens=200000, review_window_steps=requested_limit,
                              review_max_windows=1, agent_exit_policy=exit_policy,
                              stop_min_steps=requested_limit,
                              review_allow_reset=False, max_repeat=1)
        cfg["options"]["gpu"] = cfg["options"]["physical_gpu"] = 0
        cfg["options"]["vla_port"] = self._ensure_vla()
        episode = cfg["test_episodes"][0]
        episode.update(
            task=self.model_cfg["task_name"],
            eval_seed=int(self.model_cfg.get("seed") or 0),
        )
        native_identity = self._obs.get("roborsi_native_identity")
        if native_identity is not None:
            from .hybrid_resume import validate_native_identity

            validate_native_identity(
                native_identity,
                {"task": self.model_cfg["task_name"], "eval_seed": int(self.model_cfg.get("seed") or 0)},
                step=0, step_limit=cfg["options"]["review_window_steps"],
            )
            episode["layout_id"] = native_identity["layout"]
            episode["id"] = (
                f"{episode['task']}-s{episode['seed']}-eval{episode['eval_seed']}"
                f"-layout{episode['layout_id']}"
            )
        if self._resume:
            from .hybrid_resume import validate_policy

            validate_policy(self._resume["config"], cfg)
            cfg = copy.deepcopy(self._resume["config"])
            cfg["options"]["gpu"] = cfg["options"]["physical_gpu"] = 0
            cfg["options"]["vla_port"] = self._ensure_vla()
        # Internal records describe the requested case. The official evaluator's
        # result, outside this policy, is authoritative for the actual layout.
        cfg["episode_identity_scope"] = (
            "verified official reset and saved-scene sha256" if native_identity
            else "unverified debug observation; not eligible for native resume"
        )
        case = self._directory / "agent"
        case.mkdir(exist_ok=True)
        self._agent_config = load_agent_config(
            self.model_cfg, episode_dir=case, trajectory_id=self._trajectory_id
        )
        if self._agent_config.document["agent"]["model"] != cfg["model"]:
            raise ValueError("ROBORSI_CODEX_MODEL changed while preparing the episode; reset before retrying")
        (case / "icl-source.json").write_text(
            json.dumps(self._icl.source_identity, indent=2) + "\n"
        )
        self._reserve_resume()
        backend = HybridBackend(
            reference,
            task_name=self.model_cfg["task_name"],
            episode_dir=self._trajectory_directory,
            initial_observation=self._obs,
            step_limit=cfg["options"]["review_window_steps"],
            metadata_path=POLICY_DIR / profile["robot_metadata_path"],
            expected_metadata_sha256=profile["robot_metadata_sha256"],
            resource_root=POLICY_DIR / "hybrid",
            vla_port=cfg["options"]["vla_port"],
            replay=cfg["options"].get("replay"),
            vla_session_id=self._episode_id
            if self.model_cfg.get("vla_backend") == "simate_beta" else None,
            eef_execution_backend=cfg["options"].get("eef_execution_backend", "legacy_dls"),
        )
        agent = self._agent_config.create_agent(case)
        context = HybridRunContext(
            reference=reference,
            backend=backend,
            cfg=cfg,
            episode_dir=case,
            prepared_workspace=self._icl.workspace,
            training_set=self._icl.workspace,
            public_instructions=POLICY_DIR
            / profile["profile_path"]
            / "resources/interfaces"
            / "BENCHMARK.md",
            agent_python=self._tool_python,
            bubblewrap=POLICY_DIR / "runtime/codex/codex-resources/bwrap",
            bubblewrap_sha256=self._agent_cli_bwrap_sha256(),
            resume=self._resume,
            trajectory_dir=self._trajectory_directory,
            before_execution=self._claim_resume,
            execution_profile=load_skillpack_profile(
                POLICY_DIR, self.model_cfg["task_name"], case,
                vla_backend=self.model_cfg.get("vla_backend", "pi05"),
                skill_library_mode=self.model_cfg.get("skill_library_mode", "general_only")),
            model_cfg=dict(self.model_cfg),
        )
        self._connector = SkillpackController(agent, context).start()

    @staticmethod
    def _agent_cli_bwrap_sha256():
        from .custom_cli import selected_cli
        return selected_cli(POLICY_DIR, submission()["cli"])["companion_sha256"]["codex-resources/bwrap"]

    def get_action(self) -> list[dict[str, np.ndarray]]:
        """Yield one original joint update; wait for its actual executed observation."""
        if self._obs is None or self._ended:
            raise ValueError("A fresh update_obs is required before get_action")
        try:
            if self._connector is None:
                self._start()
            elif self._pending is not None:
                self._connector.acknowledge(self._pending.request_id, self._obs)
                self._step += int(getattr(self._pending, "kind", "joint") in ("joint", "bounded_curobo"))
                self._pending = None
                self._awaiting_feedback = False
            try:
                pending = self._connector.next_action(timeout=ACTION_POLL_SECONDS)
            except queue.Empty:
                # An empty standard chunk means the original Agent is still
                # thinking. Keep the same observation/session; the deployment
                # loop retries without a synthetic action or simulator step.
                return []
            if pending is None:
                raise RuntimeError(
                    "Hybrid Agent ended before the official environment terminated"
                )
            if getattr(pending, "kind", "joint") == "finish":
                self._pending = pending
                self._obs = None
                self._awaiting_feedback = True
                return [{"_roborsi_control": "finish"}]
            if getattr(pending, "kind", "joint") == "bounded_curobo":
                from .native_curobo import envelope
                self._pending = pending
                self._obs = None
                self._awaiting_feedback = True
                return [envelope(pending.request_id, pending.step_index, pending.action)]
            values = [
                number
                for name in self._action_space.names
                for number in pending.action[name]
            ]
            decoded = dict(self._action_space.decode(values).data)
            self._pending = pending
            self._obs = None
            self._awaiting_feedback = True
            return [decoded]
        except BaseException:
            self.close()
            raise

    def end_episode(self, obs: dict[str, Any]) -> None:
        """Deliver native terminal observation without reading or inventing its score."""
        if self._connector is None or self._pending is None:
            return
        controller = self._connector
        terminal_accepted = False
        try:
            self.update_obs(obs)
            self._connector.acknowledge(
                self._pending.request_id, self._obs, terminated=True
            )
            self._step += int(getattr(self._pending, "kind", "joint") in ("joint", "bounded_curobo"))
            self._pending = None
            self._awaiting_feedback = False
            self._ended = True
            deadline = time.monotonic() + 16
            while not self._connector.done and time.monotonic() < deadline:
                time.sleep(0.05)
            if self._resume:
                from .hybrid_resume import read_json

                validation_path = (
                    self._directory / "agent/controller/resume-validation.json"
                )
                if (
                    not validation_path.is_file()
                    or read_json(validation_path).get("matched") is not True
                    or self._connector.error is not None
                ):
                    raise RuntimeError(
                        "Resumed episode terminated before validated continuation"
                    )
            terminal_accepted = True
        finally:
            execution_failed = sys.exc_info()[0] is not None
            try:
                self.close()
            except Exception as error:
                # Native run_eval persists its result only after this RPC returns.
                # A cleanup failure must not discard an accepted terminal ACK, or
                # hide an earlier observation/replay validation failure.
                self._deferred_controller = controller
                self._cleanup_issue = {
                    "format": "roborsi_native_cleanup_issue_v1",
                    "error_type": type(error).__name__,
                    "reason": "controller_cleanup_failed",
                    "native_terminal_accepted": terminal_accepted,
                    "acknowledged_steps": self._step,
                    "supervisor_cleanup_required": True,
                }
                try:
                    from .hybrid_resume import write_json

                    write_json(self._directory / "cleanup-error.json", self._cleanup_issue)
                except OSError:
                    logger.warning("Could not persist native episode cleanup receipt")
                logger.warning(
                    "Native episode controller cleanup deferred to supervisor (%s)",
                    type(error).__name__,
                )
                if not terminal_accepted and not execution_failed:
                    raise
        return copy.deepcopy(getattr(self, "_cleanup_issue", None))

    def update_obs_batch(self, obs_list) -> None:
        if len(obs_list) != 1:
            raise ValueError("SenseNova-RoboRSI supports one environment per server")
        self.update_obs(obs_list[0])

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is not None and len(env_idx_list) != 1:
            raise ValueError("SenseNova-RoboRSI supports one environment per server")
        return [self.get_action()]

    def close(self) -> None:
        """Release episode state before cleanup; VLA weights stay owned by this Model."""
        if getattr(self, "_deferred_controller", None) is not None:
            # Keep the uncancelled owner quarantined until the case supervisor
            # reaps this server. reset() must never reuse it for another trial.
            return copy.deepcopy(self._cleanup_issue)
        try:
            if self._connector is not None:
                self._connector.close()
            observer = getattr(self, "_simate_observer", None)
            if observer is not None:
                observer.close()
                self._simate_observer = None
            if self._temporary is not None:
                from .hybrid_resume import read_json, write_json

                owner_path = self._directory / "episode-owner.json"
                if owner_path.is_file():
                    owner = read_json(owner_path)
                    if self._ended:
                        self._unfinished_native = None
                    elif self._connector is not None and owner.get("native_identity"):
                        self._unfinished_native = owner["native_identity"]
                    owner.update(
                        closed=True,
                        closed_at=time.time(),
                        acknowledged_steps=self._step,
                    )
                    write_json(owner_path, owner)
                # Large copy/hash publication belongs to the case supervisor,
                # after the native result RPC returns. Preserve source records.
                write_json(self._directory / "archive.pending.json", {
                    "format": "roborsi_skillpack_archive_pending_v1",
                    "task": str(self.model_cfg.get("task_name", "unknown")),
                    "episode_id": self._episode_id,
                    "source": str(self._directory),
                    "trajectory_directory": str(self._trajectory_directory),
                    "native_terminal_accepted": self._ended,
                })
        except BaseException:
            # A failed close must never permit a later reset to create another
            # episode while its old controller may still own writers/processes.
            if self._connector is not None:
                self._deferred_controller = self._connector
            raise
        finally:
            # A failed controller cleanup keeps its reservation: its thread may
            # still execute. Otherwise close (or process exit) releases startup
            # failures without consuming the request or changing trajectory ID.
            if getattr(self, "_deferred_controller", None) is None:
                descriptor = getattr(self, "_resume_start_lock", None)
                self._resume_start_lock = None
                if descriptor is not None:
                    os.close(descriptor)
            self._connector = self._icl = self._obs = self._pending = None
            self._awaiting_feedback = False
            if self._temporary is not None:
                # Keep the source episode directory for post-mortem and full
                # trajectory recovery and asynchronous supervisor publication.
                # Closing the RPC never deletes or copies this source tree.
                finalizer = getattr(self._temporary, "_finalizer", None)
                if finalizer is not None:
                    finalizer.detach()
                self._temporary = None

    def shutdown(self) -> None:
        """Stop the owned CLI and persistent VLA service at server shutdown."""
        try:
            self.close()
        finally:
            if self._vla is not None:
                self._vla.close()
                self._vla = None
