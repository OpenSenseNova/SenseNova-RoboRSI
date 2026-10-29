Follow the navigation instruction using the current RGB observations and the
RoboRSI action tool. Read EXPLORATION.md in your working directory when present;
it contains shared training experience, not evidence about this test episode.
Choose bounded batches of actions from the task's action schema. Inspect new
observations frequently. A batch can stop early at a subtask boundary, episode
end, or physical step limit. Use native `stop` to submit a target; in LHPR this
can advance to another subtask. Finish only when the environment is terminal.
Success is determined by the native scorer. Do not inspect simulator files,
ground truth, other episodes, previous test results, or credentials.
