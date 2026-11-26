"""Simulator-owned RGB calibration; no depth, task geometry or scorer access.

Projection conventions follow uni-robot-agent's RoboDojo camera_projection.py.
Capture uses separate camera products while physics remains frozen.
"""
import numpy as np


def intrinsic(params, shape):
    height, width = shape
    if params.get("cameraModel") != "pinhole" or tuple(params.get("renderProductResolution", ())) != (width, height):
        raise ValueError("Pinhole projection must match the captured RGB grid")
    matrix = np.asarray(params["cameraProjection"], dtype=float).reshape(4, 4)
    if (not np.isfinite(matrix).all() or not np.isclose(matrix[2, 3], -1)
            or min(matrix[0, 0], matrix[1, 1]) <= 0
            or not np.allclose(matrix[[0,0,0,1,1,1,3,3,3],[1,2,3,0,2,3,0,1,3]], 0)):
        raise ValueError("Unsupported camera projection convention")
    return [[width*matrix[0,0]/2, 0, width*(1-matrix[2,0])/2-.5],
            [0, height*matrix[1,1]/2, height*(1+matrix[2,1])/2-.5], [0,0,1]]


def array(value):
    if hasattr(value, "detach"):value=value.detach()
    if hasattr(value, "cpu"):value=value.cpu()
    if hasattr(value, "numpy"):value=value.numpy()
    return np.asarray(value)


def optical_pose(env, camera, slot):
    # Query the actual optical camera, including all URDF and mount transforms.
    position, q = camera.get_world_pose(camera_axes="ros")
    w,x,y,z = array(q).astype(float).reshape(4)
    if not np.isclose(np.linalg.norm([w,x,y,z]),1,atol=1e-4):
        raise ValueError("Camera quaternion must be normalized wxyz")
    transform=np.eye(4)
    transform[:3,:3]=[[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                        [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                        [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]]
    transform[:3,3]=array(position).reshape(3)
    if not hasattr(env,"room"):
        scene=getattr(env,"scene",None)
        if scene is None:scene=env.scene_manager
        transform[:3,3]-=array(scene.env_origins[slot])
    return transform.tolist()


def calibrated_observation(env, obs, slot=0):
    """Attach exact-grid calibration and replace RGB with its paired capture.

    Only simulator-owner code calls this function. Separate products never
    attach readers to the live tiled graph. Extra renders advance no physics.
    """
    import omni.replicator.core as rep
    manager=env.camera_manager
    names=list(manager.camera_names[slot])
    required=("cam_head","cam_left_wrist","cam_right_wrist")
    before=int(env.take_action_cnt[slot])
    physics_before=getattr(getattr(env,"sim",None),"_sim_step_counter",None)
    readers=[]
    captured={}
    try:
        for name in required:
            camera=manager.cameras[slot][names.index(name)]
            if getattr(camera,"_roborsi_calibration_initialized",False):camera.resume()
            else:
                camera.initialize()
                camera._roborsi_calibration_initialized=True
            reader=rep.annotators.get("CameraParams")
            readers.append((name,camera,reader))
            reader.attach(camera.get_render_product_path())
        for _ in range(8):
            env.render()
            for name,camera,reader in readers:
                rgb=camera.get_rgb()
                if rgb is None:continue
                rgb=array(rgb)[...,:3]
                if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.size==0:continue
                try:k=intrinsic(reader.get_data(),rgb.shape[:2])
                except (ValueError,KeyError):continue
                captured[name]=(rgb.copy(),k,optical_pose(env,camera,slot))
            if len(captured)==len(required):break
        if len(captured)!=len(required):
            raise RuntimeError("Camera calibration not ready; host capture failed before model inference")
        physics_after=getattr(getattr(env,"sim",None),"_sim_step_counter",None)
        if int(env.take_action_cnt[slot])!=before or physics_before!=physics_after:
            raise RuntimeError("Physics advanced during calibrated RGB capture")
        cameras={}
        for name,(rgb,k,t) in captured.items():
            obs["vision"][name]={"color":rgb}
            cameras[name]={"K":k,"T_base_camera_cv":t,"image_size":list(rgb.shape[:2]),"distortion":[]}
        obs.setdefault("additional_info",{})["spatial_calibration"]={
            "schema_version":1,"frame":"environment_origin","units":"meters",
            "control_step":before,"source":"render_projection_and_actual_optical_pose",
            "pixel_order":"u,v","pixel_coordinates":"integer_pixel_centers","cameras":cameras}
        return obs
    finally:
        for _,camera,reader in readers:
            try:reader.detach()
            finally:camera.pause()
