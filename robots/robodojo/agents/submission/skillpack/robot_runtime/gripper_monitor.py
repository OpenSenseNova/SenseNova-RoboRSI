"""Observe gripper feedback after each acknowledged step; never move the robot."""
import math


OPENING_CHANGE_THRESHOLD = .02
COMMAND_CHANGE_TOLERANCE = .005
SETTLE_RANGE = .003
SETTLE_SAMPLES = 3
ARMS = ("left", "right")


class GripperMonitor:
    """Retain a settled reference across act calls with an unchanged command.

    Opening/closing commands first wait for stable measured feedback. Once
    armed, compare with the fixed reference, so gradual drift also interrupts.
    This detects changed contact, not confirmed possession or object loss.
    """

    def __init__(self):
        self._arms = {}
        self._step = None

    def prepare(self, observation, targets):
        """Register this step's actual commands before issuing any motion."""
        step = observation["control_step"]
        if self._step != step:
            # Do not reuse a reference across unobserved external motion.
            self._arms.clear()
        self._step = step
        for arm in ARMS:
            command = float(targets[arm]["gripper_opening"])
            current = self._arms.get(arm)
            if current is None or abs(command - current["command_reference"]) > COMMAND_CHANGE_TOLERANCE:
                self._arms[arm] = dict(command=command, command_reference=command,
                                       reference=None, samples=[])
            else:
                current["command"] = command

    def sample(self, observation):
        """Return a stop event on the first threshold crossing in either arm."""
        step = observation["control_step"]
        if step <= self._step:
            return None  # Re-reading the same observation is not a settled step.
        self._step = step
        changes = []
        for arm in ARMS:
            current = self._arms[arm]
            opening = float(observation["state"][arm + "_ee_joint_state"][0])
            if not math.isfinite(opening):
                raise ValueError("Non-finite measured gripper opening: " + arm)
            reference = current["reference"]
            if reference is not None:
                delta = opening - reference
                if abs(delta) >= OPENING_CHANGE_THRESHOLD - 1e-12:
                    changes.append(dict(arm=arm, commanded_opening=current["command"],
                                        reference_opening=reference, measured_opening=opening,
                                        opening_delta=delta,
                                        direction="opening" if delta > 0 else "closing"))
                    # A new model decision may continue from the changed state;
                    # never replay the cancelled tail or retain a stale alarm.
                    current.update(reference=None, samples=[opening])
            else:
                samples = current["samples"]
                samples.append(opening)
                del samples[:-SETTLE_SAMPLES]
                if len(samples) == SETTLE_SAMPLES and max(samples) - min(samples) <= SETTLE_RANGE:
                    current["reference"] = opening
        if changes:
            return dict(kind="gripper_opening_changed", control_step=step,
                        threshold=OPENING_CHANGE_THRESHOLD, changes=changes,
                        interpretation="Contact may have changed; inspect RGB to verify possession",
                        remaining_motion_cancelled=True)
        return None


def monitor_for(owner):
    """The existing episode owner retains state; there is no background worker."""
    monitor = getattr(owner, "_act_gripper_monitor", None)
    if monitor is None:
        monitor = owner._act_gripper_monitor = GripperMonitor()
    return monitor
