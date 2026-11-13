CALIBRATED DEPLOYMENT, PROTOCOL V2

You have ordinary robot setup information, not object-state ground truth.
input/robot.urdf is a geometry-only robot description. input/robot.json also
provides runtime joint ordering, position limits and resolved units. Inactive
non-mimic joints are modeled at their documented zero command, not measured.
The kinematic model is nominal and may differ from realized motion under load.

Each observation includes camera_calibration for exactly its permitted cameras.
T_robot_base_from_camera_optical maps homogeneous camera optical coordinates
to the URDF robot base: p_base = T @ p_optical. Optical axes are x right, y down,
z forward; lengths are metres. K is calibrated for the delivered 640x480 image.
Pinhole cameras have no distortion; fisheye cameras use OpenCV's equidistant
model with four coefficients k1,k2,k3,k4. Do not treat fisheye pixels as pinhole.
These are exact simulated calibration values after camera randomization, an
idealization of deployment calibration. They supply no object depths or poses.

A generic robot-only FK/IK helper is available read-only at
/interface/kinematics_tool.py. For example:

    import sys
    sys.path.insert(0, '/interface')
    from kinematics_tool import RobotKinematics
    robot = RobotKinematics.from_file()
    poses = robot.fk(qpos)  # T_robot_base_from_link for each link
    result = robot.ik(qpos, link_name, target_4x4, selected_active_joint_names)

IK returns a candidate, residuals, and success flag. It is local bounded
numerical IK, not a collision checker, motion planner, grasp selector, or
guarantee of physical tracking. It never executes actions. Check success and
choose your own interpolation before submitting through the existing step tool.
All chosen arm/finger targets and all executed intermediate steps count normally.
You remain free to implement your own tools using the same permitted inputs.
