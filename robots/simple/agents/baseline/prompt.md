Control the Unitree G1 with continuous Dex3 hand targets for the supplied task.
Use the delivered head-left RGB and 32-value robot state to decide your actions.
Submit `actions`: an array of 36-value action rows through the native action tool.

Columns 0:14 are hand joint angles in the native Psi0 ordering, 14:28 arm joint
angles, 28:31 torso roll/pitch/yaw, 31 base height in metres, 32:34 forward/lateral
velocity in metres per second, 34 turning flag, and 35 target yaw in radians.
Angles and absolute joint targets use radians. Each row executes for 0.02 seconds.
Respect the remaining native action-row budget in the observation. Use shorter
chunks when contact, balance or object motion requires fresh feedback.

State 0:28 contains measured hands and arms. For decoupled WBC, state 28:31 is
measured waist roll/pitch/yaw; for AMO it is the previous commanded torso target.
State 31 is the previous commanded base height, not a measured object distance.
Infer geometry from the images; do not assume unobserved object positions.
The simulator waits during reasoning. Continue until the environment ends the
episode. Model-declared completion does not establish physical task success.
