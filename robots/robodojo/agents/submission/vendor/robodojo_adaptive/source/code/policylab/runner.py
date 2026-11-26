from __future__ import annotations

import json
from pathlib import Path
import socketserver
import threading
import time
import traceback

from . import codex
from .api import InvalidAction
from .common import event, snapshot, stop_process, write_json



def public_observation(value, folder, public_prefix):
    if isinstance(value, dict):
        return {k: public_observation(v, folder, public_prefix) for k, v in value.items()}
    if isinstance(value, list):
        return [public_observation(v, folder, public_prefix) for v in value]
    if isinstance(value, str) and value.startswith("obs://"):
        relative = Path(value[6:])
        target = (folder / relative).resolve()
        if relative.is_absolute() or not target.is_relative_to(folder.resolve()) or not target.is_file():
            raise ValueError("Adapter observation file must exist inside its observation directory")
        return public_prefix + "/" + relative.as_posix()
    return value


class Experiment:
    def __init__(self, cfg, run, backend):
        self.cfg, self.run, self.backend = cfg, run, backend
        self.lock = threading.RLock()
        self.phase, self.active, self.latest = "idle", None, None
        self.count = 0
        self.request_sequence = 0
        self.segment_done = False
        self.fault = None
        self.deadline = float("inf")
        self.results = []
        options = cfg.get("options", {})
        self.review_window_steps = options.get("review_window_steps")
        self.review_max_windows = options.get("review_max_windows", 10)
        self.review_allow_reset = options.get("review_allow_reset", False)
        if self.review_window_steps is not None and (type(self.review_window_steps) is not int or self.review_window_steps <= 0
                                                     or type(self.review_max_windows) is not int or self.review_max_windows <= 0):
            raise ValueError("Review windows must be positive integers")
        if type(self.review_allow_reset) is not bool or (self.review_allow_reset and self.review_window_steps is None):
            raise ValueError("review_allow_reset requires review windows and a boolean value")
        self.reviewed_windows = 0
        self.used_steps_before_attempt = 0
        self.last_review_step = None

    def total_step(self, reply=None):
        reply = reply or self.latest
        return self.used_steps_before_attempt + int(reply["observation"]["control_step"])

    def review_status(self, reply=None):
        if self.review_window_steps is None or self.phase != "evaluation" or reply is None:
            return {}
        step = self.total_step(reply) if self.review_allow_reset else int(reply["observation"]["control_step"])
        limit = self.review_window_steps * self.review_max_windows if self.review_allow_reset else int(reply["observation"]["step_limit"])
        window = self.reviewed_windows + 1
        boundary = window * self.review_window_steps if window <= self.review_max_windows else None
        success = reply["done"] and reply["metrics"].get("success") is True
        return {"base_step_limit": self.review_window_steps, "max_step_limit": limit,
                "review_window": min(window, self.review_max_windows),
                "next_review_step": boundary, "steps_to_review": max(0, boundary - step) if boundary else 0,
                "review_due": boundary is not None and step >= boundary and not success
                and (not reply["done"] or (self.review_allow_reset and step < limit))}

    def checkpoint(self, name):
        destination = self.run / "checkpoints" / name
        files = snapshot(self.run / "workspace", destination / "workspace")
        write_json(destination / "files.json", files)
        return str(destination.relative_to(self.run))

    def finalize(self, reason):
        if self.active is None or self.active.get("recorded"):
            return
        terminal = self.latest is not None and self.latest["done"]
        result = {"episode": self.active["episode"], "phase": "evaluation_attempt" if reason == "agent_reset" else self.phase,
                  "record": self.active["name"], "reason": reason,
                  "native_done": terminal,
                  "metrics": self.latest["metrics"] if self.latest else {},
                  "success": self.latest["metrics"]["success"] if terminal else None}
        if reason == "reset_requested":
            result["native_success"] = result["success"]
            result["success"] = False
            result["failure_type"] = "reset_requested"
        result["checkpoint"] = self.checkpoint(self.active["name"])
        write_json(self.active["record_dir"] / "result.json", result)
        event(self.run / "episodes.jsonl", **result)
        self.results.append(result)
        self.active["recorded"] = True

    def accept(self, reply, terminal_reason="native_terminal"):
        self.latest = reply
        obs = public_observation(reply["observation"], self.active["observation_dir"],
                                 "/observations/" + self.active["name"])
        if self.review_allow_reset and self.phase == "evaluation":
            obs["attempt_control_step"] = obs["control_step"]
            obs["control_step"] = self.total_step(reply)
            obs["step_limit"] = self.review_window_steps * self.review_max_windows
            obs["steps_remaining"] = max(0, obs["step_limit"] - obs["control_step"])
        obs.update(self.review_status(reply))
        result = {"observation": obs, "done": reply["done"], "phase": self.phase}
        if reply["done"] and self.cfg["terminal_feedback"]:
            result["success"] = reply["metrics"]["success"]
        event(self.active["record_dir"] / "transitions.jsonl", **reply)
        write_json(self.run / "observations" / "latest.json", result)
        self.public = result
        if reply["done"]:
            self.finalize(terminal_reason)
            if self.phase == "evaluation" and not obs.get("review_due"):
                self.segment_done = True
        return result

    def end_episode(self, reason):
        if self.active is not None and not self.active.get("recorded"):
            self.accept(self.backend.request(op="finish"), reason)

    def reset_episode(self, episode, preserve_review=False):
        self.end_episode("practice_reset")
        self.count += 1
        name = f"{self.phase}-{self.count:04d}"
        record_dir = self.run / "episodes" / name
        observation_dir = self.run / "observations" / name
        record_dir.mkdir(parents=True)
        observation_dir.mkdir(parents=True)
        self.active = {"episode": episode, "name": name, "record_dir": record_dir,
                       "observation_dir": observation_dir, "recorded": False}
        self.latest = None
        if not preserve_review:
            self.reviewed_windows = 0
            self.used_steps_before_attempt = 0
        self.last_review_step = None
        write_json(record_dir / "episode.json", episode)
        reply = self.backend.request(op="reset", episode=episode,
                                     record_dir=str(record_dir / "native"), observation_dir=str(observation_dir))
        if self.review_window_steps is not None and self.phase == "evaluation":
            expected = self.review_window_steps * self.review_max_windows
            if reply["observation"].get("step_limit") != expected:
                raise ValueError(f"Review experiment requires a {expected}-step adapter horizon")
            step = int(reply["observation"]["control_step"])
            if not preserve_review:
                self.reviewed_windows = min((max(step, 1) - 1) // self.review_window_steps,
                                            self.review_max_windows - 1)
        return self.accept(reply)

    def request(self, request):
        with self.lock:
            self.request_sequence += 1
            event(self.run / "requests.jsonl", sequence=self.request_sequence, request=request)
            if not isinstance(request, dict):
                raise InvalidAction("Request must be a JSON object")
            if time.monotonic() >= self.deadline or self.segment_done:
                raise InvalidAction("This segment has ended; save your notes and exit")
            if self.fault:
                raise InvalidAction("Environment unavailable; exit and retain your notes")
            op = request.get("op")
            allowed = {"observe": {"op"}, "act": {"op", "action", "repeat"},
                       "reset": {"op"}, "ready": {"op"},
                       "review": {"op", "analysis"}, "stop": {"op", "reason"}}
            if op not in allowed or set(request) != allowed[op]:
                raise InvalidAction("Use observe, act(action, repeat), review(analysis), stop(reason), reset or ready")
            if self.phase == "exploration" and op != "ready":
                raise InvalidAction("Exploration is training-data study only; no live simulator is available. Save your research and call ready().")
            if op == "ready" and self.phase != "exploration":
                raise InvalidAction(f"{op} is only available during exploration")
            if op == "reset" and self.phase != "exploration" and self.review_allow_reset and not (
                self.phase == "evaluation" and self.review_allow_reset and self.active is not None
                and not self.latest["done"] and self.last_review_step == self.total_step()
                and int(self.latest["observation"]["control_step"]) > 0
                and self.total_step() < self.review_window_steps * self.review_max_windows
            ):
                raise InvalidAction("Evaluation reset requires a review, actions in this attempt, and remaining steps")
            try:
                if op == "reset" and self.phase == "evaluation" and not self.review_allow_reset:
                    if self.active is None or self.latest["done"]:
                        raise InvalidAction("Evaluation has no active attempt to stop")
                    event(self.active["record_dir"] / "stop.jsonl", control_step=self.total_step(),
                          reason="Evaluation reset requested; terminated under the no-reset policy.")
                    result = self.accept(self.backend.request(op="finish"), "reset_requested")
                    self.segment_done = True
                    return result
                if op == "reset":
                    if self.phase == "evaluation":
                        step = self.total_step()
                        reply = self.backend.request(op="finish")
                        result = self.accept(reply, "native_terminal" if reply["metrics"].get("success") else "agent_reset")
                        if reply["metrics"].get("success"):
                            return result
                        self.used_steps_before_attempt = step
                        self.segment_done = False
                        return self.reset_episode(self.active["episode"], preserve_review=True)
                if op == "ready":
                    self.end_episode(op)
                    self.segment_done = True
                    return {"status": op, "message": "End your response now."}
                if self.active is None:
                    raise InvalidAction("No active evaluation episode")
                if op == "stop":
                    if self.phase != "evaluation" or self.latest["done"] or self.last_review_step != self.total_step():
                        raise InvalidAction("Stop requires an evaluation review at the current step")
                    minimum = self.cfg.get("options", {}).get("stop_min_steps", 0)
                    if type(minimum) is not int or minimum < 0:
                        raise InvalidAction("stop_min_steps must be a nonnegative integer")
                    if self.total_step() < minimum:
                        raise InvalidAction(f"Stop is unavailable before step {minimum}; continue and review the demonstrated ending requirements")
                    reason = request["reason"]
                    if not isinstance(reason, str) or not reason.strip() or len(reason) > 2000:
                        raise InvalidAction("Stop reason must be a nonempty string of at most 2000 characters")
                    event(self.active["record_dir"] / "stop.jsonl", control_step=self.total_step(), reason=reason)
                    result = self.accept(self.backend.request(op="finish"), "agent_stop")
                    self.segment_done = True
                    return result
                if op == "observe":
                    return self.public
                if op == "review":
                    if self.phase != "evaluation" or self.review_window_steps is None or (self.latest["done"] and not self.review_status(self.latest).get("review_due")):
                        raise InvalidAction("No evaluation review is available")
                    analysis = request["analysis"]
                    fields = {"observed_state", "demo_comparison", "why_not_done", "next_plan", "ending_check"}
                    if not isinstance(analysis, dict) or set(analysis) != fields or any(
                        not isinstance(analysis[key], str) or not analysis[key].strip() for key in fields
                    ):
                        raise InvalidAction("Review needs observed_state, demo_comparison, why_not_done, next_plan and ending_check")
                    scheduled = self.review_status(self.latest).get("review_due", False)
                    if scheduled:
                        self.reviewed_windows += 1
                    self.last_review_step = self.total_step()
                    event(self.active["record_dir"] / "reviews.jsonl", window=self.reviewed_windows,
                          kind="checkpoint" if scheduled else "early",
                          control_step=self.total_step() if self.review_allow_reset else self.latest["observation"]["control_step"], analysis=analysis)
                    self.public["observation"].update(self.review_status(self.latest))
                    write_json(self.run / "observations" / "latest.json", self.public)
                    if self.latest["done"]:
                        self.segment_done = True
                    return self.public
                if self.latest["done"]:
                    raise InvalidAction("Episode ended; end your response")
                if self.review_status(self.latest).get("review_due"):
                    raise InvalidAction("Checkpoint review due; inspect observations and call review(analysis)")
                repeat = request["repeat"]
                maximum = self.cfg.get("options", {}).get("max_repeat", 10)
                if type(repeat) is not int or not 1 <= repeat <= maximum:
                    raise InvalidAction(f"repeat must be an integer from 1 to {maximum}")
                self.last_review_step = None
                for step_index in range(repeat):
                    if time.monotonic() >= self.deadline:
                        break
                    record = {"request_sequence": self.request_sequence, "step_index": step_index,
                              "requested_repeat": repeat, "action": request["action"]}
                    event(self.active["record_dir"] / "actions.jsonl", status="requested", **record)
                    reply = self.backend.request(op="step", action=request["action"])
                    result = self.accept(reply)
                    if self.review_allow_reset and self.phase == "evaluation" and not reply["done"] and self.total_step() >= self.review_window_steps * self.review_max_windows:
                        result = self.accept(self.backend.request(op="finish"))
                    event(self.active["record_dir"] / "actions.jsonl", status="completed",
                          applied_action=reply.get("applied_action"),
                          control_step=reply["observation"].get("control_step"), **record)
                    if result["done"] or result["observation"].get("review_due"):
                        break
                return self.public
            except InvalidAction:
                raise
            except Exception:
                self.fault = traceback.format_exc()
                (self.run / "adapter_error.txt").write_text(self.fault)
                self.finalize("adapter_error")
                raise


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        self.connection.settimeout(30)
        sequence = None
        try:
            line = self.rfile.readline(1024 * 1024 + 1)
            if len(line) > 1024 * 1024 or not line.endswith(b"\n"):
                raise InvalidAction("Request too large or incomplete")
            request = json.loads(line)
            sequence = self.server.experiment.request_sequence + 1
            result = self.server.experiment.request(request)
        except (InvalidAction, json.JSONDecodeError) as exc:
            result = {"error": str(exc)}
        except Exception:
            result = {"error": "Environment error; do not repeat the action. Exit and retain your notes."}
        event(self.server.experiment.run / "responses.jsonl", sequence=sequence, response=result)
        try:
            self.wfile.write((json.dumps(result, allow_nan=False) + "\n").encode())
        except (BrokenPipeError, ConnectionResetError):
            pass


def run_segment(experiment, cfg, run, private_home, name, prompt, seconds, session, launch=codex.launch):
    segment = run / "segments" / name
    segment.mkdir(parents=True)
    with experiment.lock:
        experiment.segment_done = (experiment.phase == "evaluation" and experiment.latest is not None
                                   and experiment.latest["done"] and not experiment.review_status(experiment.latest).get("review_due"))
        experiment.deadline = time.monotonic() + seconds
    process = launch(cfg, run, private_home, segment, prompt, session)
    started, stop_requested = time.monotonic(), None
    reason = "agent_exited"
    current_segment, refreshes = segment, []
    try:
        while True:
            if process.poll() is not None:
                if (cfg.get("options", {}).get("live_context_recovery", False)
                        and experiment.phase == "evaluation" and not experiment.segment_done
                        and not experiment.fault and time.monotonic() < experiment.deadline
                        and len(refreshes) < 64):
                    messages = []
                    log = current_segment / "model.jsonl"
                    if log.exists():
                        for line in log.read_text().splitlines():
                            try:
                                item = json.loads(line)
                            except ValueError:
                                continue
                            if item.get("type") == "turn.failed":
                                messages.append(item.get("error", {}).get("message", ""))
                    failed = messages[-1] if messages else ""
                    image_limit = "maximum number of images" in failed
                    rate_limit = "429" in failed
                    if image_limit or rate_limit:
                        found, _ = codex.events_summary(log)
                        entry = {"source": str(current_segment.relative_to(run)), "error": failed,
                                 "new_session": image_limit, "control_step": experiment.latest["observation"].get("control_step", experiment.latest["observation"].get("step")),
                                 "native_reset": False}
                        refreshes.append(entry)
                        event(segment / "live_context_recovery.jsonl", **entry)
                        if rate_limit:
                            until = min(experiment.deadline, time.monotonic() + 60)
                            while time.monotonic() < until:
                                time.sleep(.2)
                        current_segment = segment / "rounds" / f"refresh-{len(refreshes):04d}"
                        current_segment.mkdir(parents=True)
                        recovery_prompt = prompt + "\nContinue this SAME live episode from its current measured state; the simulator has NOT reset and the native action budget is unchanged. Model context was refreshed only for gateway capacity limits. Do not explore again. Read your own saved notes, call observe() and inspect current images. If observe returns pending_vla_proposal, review that same unmoved proposal before requesting another."
                        process = launch(cfg, run, private_home, current_segment, recovery_prompt,
                                         None if image_limit else found)
                        continue
                break
            if experiment.fault:
                reason = "adapter_error"
                break
            now = time.monotonic()
            if now >= experiment.deadline:
                reason = "budget_exhausted"
                break
            if experiment.segment_done:
                stop_requested = stop_requested or now
                if now - stop_requested >= 15:
                    reason = "submitted"
                    break
            time.sleep(.05)
    finally:
        stop_process(process)
    with experiment.lock:
        if not experiment.fault:
            if experiment.phase == "evaluation":
                experiment.finalize(reason)
            else:
                experiment.end_episode(reason)
        if experiment.segment_done:
            reason = "ready" if experiment.phase == "exploration" else "finished"
        experiment.deadline = 0
        checkpoint = experiment.checkpoint("segment-" + name)
    found_session, usage = codex.events_summary(current_segment / "model.jsonl")
    summary = {"status": reason, "returncode": process.returncode, "wall_seconds": time.monotonic() - started,
               "session_id": found_session or session, "usage": usage, "checkpoint": checkpoint,
               "live_context_recoveries": refreshes, "last_model_log": str(current_segment.relative_to(run))}
    write_json(segment / "result.json", summary)
    return summary
