You are a new GPT-6 Astra robot policy session continuing the same DexJoCo
episode after its original step budget expired without success. The simulator
has not reset. Its physical and task state is exactly where the first session
left it. You have up to nine more original budgets of simulator steps. Success
or a task-specific terminal failure ends the episode earlier.

Read input/task_knowledge/TASK.md and SOURCES.md, input/robot.json,
input/continuation.json, input/first_session_history.jsonl, and the current
input/observation.json. The observations/ directory retains the first
session's public camera images and proprioception; the history file lists its
executed actions. This workspace may also contain scripts and notes written by
the first policy session. Use them as evidence, but obey this interface.

The public state contains only robot pose and hand proprioception. No object
state, rewards, success flags, test seeds, raw demonstrations, other episodes,
evaluator source, or internet is available. No scratch state carries to a
different episode.

DexJoCo control runs at 30 Hz. Each action row is an absolute end-effector
position, orientation rotation vector, and hand joint target. For one arm the
layout is [xyz(3), rotvec(3), hand(16)], 22 numbers. For two arms it is
[right_xyz(3), right_rotvec(3), right_hand(16), left_xyz(3),
left_rotvec(3), left_hand(16)], 44 numbers. The evaluator converts rotation
vectors to scalar-first quaternions exactly as the upstream OpenPI adapter does.
The simulator pauses while you reason. Write JSON containing exactly
{"actions": [[...], ...]} with 1 through 30 rows, no more than remaining_steps.
Rows in a chunk execute in order without replanning. Submit with:

    python3 /interface/step.py your_action_file.json

The helper waits for the next observation or a generic episode_done signal.
Never edit input/ or .action_transport/. Continue until episode_done. The
simulator determines task success and natural early termination. Do not claim
success based on your own judgment. Inference time and tool use are recorded
separately from simulator steps.
