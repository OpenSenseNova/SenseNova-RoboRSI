"""Execute standard observations and one-action chunks through the policy server."""

from __future__ import annotations

import logging
import math
import sys
from copy import deepcopy

logger = logging.getLogger(__name__)


def _report_cleanup_issue(receipt):
    if isinstance(receipt, dict) and receipt.get("format") == "roborsi_native_cleanup_issue_v1":
        logger.warning("RoboRSI controller cleanup requires case supervisor; native scoring continues")


def _finish_native_episode(env):
    """Use the original NativeAdapter.finish final check without a robot step.

    The official run_eval remains responsible for persisting the resulting score.
    Reject other environment interfaces rather than converting a finish request
    into an idle action or an invented terminal reward.
    """
    flags = getattr(env, "end_flag", None)
    success = getattr(env, "success", None)
    reward_manager = getattr(env, "reward_manager", None)
    get_reward = getattr(reward_manager, "get_reward", None)
    if (
        getattr(env, "num_envs", None) != 1
        or not isinstance(flags, list)
        or len(flags) != 1
        or not isinstance(success, list)
        or len(success) != 1
        or not callable(get_reward)
    ):
        raise RuntimeError("Native finish requires the official single-env reward interface")
    if flags[0]:
        return
    rewards = get_reward(final_check=True)
    if len(rewards) != 1:
        raise RuntimeError("Native finish requires exactly one official reward")
    reward = float(rewards[0])
    if not math.isfinite(reward):
        raise RuntimeError("Native finish received a non-finite official reward")
    success[0] = bool(reward > 1 - 1e-3)
    flags[0] = True


def eval_one_episode(TASK_ENV, model_client):
    """Preserve each action's feedback and close the owned agent after a trial."""
    try:
        model_client.call(func_name="reset")
        cached_observation = None
        execution_ack = None
        while not TASK_ENV.is_episode_end():
            obs = TASK_ENV.get_obs() if cached_observation is None else cached_observation
            cached_observation = None
            if execution_ack is not None:
                obs = {**obs, "roborsi_execution": deepcopy(execution_ack)}
            model_client.call(func_name="update_obs", obs=obs)
            actions = model_client.call(func_name="get_action")
            while actions == []:
                # Retry only inference: another observation would append video
                # frames, and an idle joint command would spend native steps.
                actions = model_client.call(func_name="get_action")
            if not isinstance(actions, list) or len(actions) != 1:
                raise ValueError(
                    "SenseNova-RoboRSI requires exactly one action per chunk"
                )
            action = actions[0]
            if isinstance(action, dict) and "_roborsi_control" in action:
                if action == {"_roborsi_control": "finish"}:
                    _finish_native_episode(TASK_ENV)
                    break
                if action.get("_roborsi_control") != "bounded_curobo":
                    raise ValueError("Unsupported RoboRSI native control request")
                import numpy as np
                from .native_curobo import execute_control
                def original_joint_step(command):
                    nonlocal cached_observation
                    TASK_ENV.take_action(command)
                    cached_observation = TASK_ENV.get_obs()
                    return {"observation": {"state": {k: np.asarray(v).tolist()
                            for k, v in cached_observation["state"].items()}},
                            "done": TASK_ENV.is_episode_end()}
                result = execute_control(TASK_ENV, action, step_joint=original_joint_step,
                                         initial_observation=obs)
                execution_ack = result["roborsi_execution"]
            else:
                TASK_ENV.take_action(action)
                execution_ack = None

        # The final action also needs feedback, even though no next action is due.
        terminal = TASK_ENV.get_obs() if cached_observation is None else cached_observation
        if execution_ack is not None:
            terminal = {**terminal, "roborsi_execution": deepcopy(execution_ack)}
        receipt = model_client.call(func_name="end_episode", obs=terminal)
        _report_cleanup_issue(receipt)
    finally:
        if sys.exc_info()[0] is None:
            _report_cleanup_issue(model_client.call(func_name="close"))
        else:
            try:
                model_client.call(func_name="close")
            except Exception:
                # Keep the original environment or inference failure visible.
                logger.warning("Could not close the RoboRSI agent after a trial error")


def eval_one_episode_batch(TASK_ENV, model_client):
    """Reject unsupported parallel episodes before starting any agent."""
    raise NotImplementedError("SenseNova-RoboRSI requires eval_batch=false")
