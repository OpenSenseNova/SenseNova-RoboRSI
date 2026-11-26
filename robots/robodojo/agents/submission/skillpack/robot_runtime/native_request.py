"""Enable model-chosen EEF targets behind the registered robot tools.

Install only in a new episode's bridge. Public targets have already been
translated to native poses by RobotTools. Existing runtime/scoring is reused.
"""

from copy import deepcopy
import json
import math

from robot_tools import NATIVE_DUAL


def make_request(hybrid_request, native_request, *, invalid_action_type,
                 allow_direct_targets: bool, tracked_execution: bool = True):
    """Wrap the existing host while keeping its native per-step checks."""
    def request(self, req):
        if req.get("op") == "observe":
            result = deepcopy(hybrid_request(self, req))
            result["manual_tcp_targets_enabled"] = allow_direct_targets
            result["execution_policy"] = "original_vla_joints_or_closed_loop_tcp" if tracked_execution else "legacy_single_step"
            return result
        if tracked_execution and req.get("op") == "vla_review" and req.get("decision") == "accept":
            from execution import execute_vla
            with self.lock:
                return execute_vla(self, req, native_request, invalid_action_type=invalid_action_type)
        if req.get("op") != "act_eef":
            return hybrid_request(self, req)
        if not allow_direct_targets:
            raise invalid_action_type("Direct TCP fallback is disabled")
        from jsonschema import Draft202012Validator, ValidationError

        schema = dict(type="object", properties={
            "op": dict(const="act_eef"),
            "targets": dict(type="array", items=NATIVE_DUAL, minItems=1, maxItems=50),
            "reason": dict(type="string", minLength=1),
            "max_native_steps": dict(type="integer", minimum=1, maximum=160),
            "assessment": dict(type="object", properties={
                "current_subgoal": dict(type="string", minLength=1)},
                required=["current_subgoal"], additionalProperties=False),
        }, required=["op", "targets", "reason", "assessment"], additionalProperties=False)
        try:
            json.dumps(req, allow_nan=False)
            Draft202012Validator(schema).validate(req)
            for target in req["targets"]:
                for arm in target.values():
                    if abs(math.sqrt(sum(x*x for x in arm["quaternion_wxyz"])) - 1) > 1e-4:
                        raise ValueError("A unit quaternion is required")
        except (ValueError, TypeError, ValidationError) as error:
            raise invalid_action_type(str(error)) from error
        with self.lock:
            # Reject the old prediction using its actual current ID internally.
            # The model supplies only its selected EEF targets.
            pending = getattr(self, "vla_pending", None)
            if pending is not None:
                hybrid_request(self, dict(op="vla_review", proposal_id=pending["proposal_id"],
                                         decision="reject", targets=None,
                                         reason=req["reason"], assessment=req["assessment"]))
            if tracked_execution:
                from execution import execute_tcp
                return execute_tcp(self, req, native_request, invalid_action_type=invalid_action_type)
            before = self.latest["observation"]["control_step"]
            for target in req["targets"]:
                native_request(self, dict(op="act", repeat=1, action=dict(
                    mode="eef", target=target, reason=req["reason"], assessment=req["assessment"])))
                if self.latest["done"] or self.public["observation"].get("review_due"):
                    break
            moved = self.latest["observation"]["control_step"] - before
            return dict(status="executed", action_source="model_tcp", executed_steps=moved,
                        **deepcopy(self.public))
    return request


def make_experiment_class(reference, *, allow_direct_targets: bool):
    """Construct the bridge class for a new robot-tool episode owner."""
    class Experiment(reference.runner.Experiment):
        request = make_request(reference.hybrid_host.request, reference.hybrid_host.ORIGINAL,
                               invalid_action_type=reference.runner.InvalidAction,
                               allow_direct_targets=allow_direct_targets)
    return Experiment
