"""Calibrated RGB ray geometry. No environment interfaces."""
import itertools
import cv2,numpy as np
from scipy.optimize import least_squares

def skew(v):
    x,y,z=v;return np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])

def camera_arrays(cam):
    K=np.asarray(cam['K'],float);C=np.asarray(cam['T_base_camera_cv'],float)
    if K.shape!=(3,3) or C.shape!=(4,4) or not np.isfinite(K).all() or not np.isfinite(C).all():raise ValueError('Invalid calibration matrices')
    if not np.allclose(C[3],[0,0,0,1]) or not np.allclose(C[:3,:3].T@C[:3,:3],np.eye(3),atol=1e-5) or np.linalg.det(C[:3,:3])<.999:raise ValueError('Camera transform is not a right-handed SE(3) matrix')
    if K[0,0]<=0 or K[1,1]<=0:raise ValueError('Focal lengths must be positive')
    return K,C

def undistort(pixel,cam):
    K,_=camera_arrays(cam);return cv2.undistortPoints(np.array(pixel,dtype=float).reshape(-1,1,2),K,np.asarray(cam.get('distortion',[]),float)).reshape(-1,2)

def ray(pixel,cam):
    _,C=camera_arrays(cam);uv=undistort(pixel,cam)[0];d=C[:3,:3]@np.r_[uv,1.];return C[:3,3],d/np.linalg.norm(d)

def project(point,cam):
    K,C=camera_arrays(cam);v=C[:3,:3].T@(np.asarray(point)-C[:3,3]);rvec=np.zeros(3);tvec=np.zeros(3)
    pix=cv2.projectPoints(v.reshape(1,3),rvec,tvec,K,np.asarray(cam.get('distortion',[]),float))[0].reshape(2)
    return pix,float(v[2])

def fundamental(c1,c2):
    K1,C1=camera_arrays(c1);K2,C2=camera_arrays(c2);T=np.linalg.inv(C2)@C1
    return np.linalg.inv(K2).T@skew(T[:3,3])@T[:3,:3]@np.linalg.inv(K1)

def ideal_pixel(p,cam):
    K,_=camera_arrays(cam);xy=undistort(p,cam)[0];v=K@np.r_[xy,1.];return v/v[2]

def epipolar_error(p1,p2,c1,c2):
    F=fundamental(c1,c2);a=ideal_pixel(p1,c1);b=ideal_pixel(p2,c2);l=F@a;r=F.T@b;num=abs(b@l)
    return float(max(num/max(np.linalg.norm(l[:2]),1e-12),num/max(np.linalg.norm(r[:2]),1e-12)))

def triangulate(observations,cameras,pixel_sigma=1.,extrinsic_sigma_m=0.,min_angle_deg=2.,max_reprojection_px=2.,max_uncertainty_m=.01):
    if len(observations)<2:return {'accepted':False,'reason':'fewer_than_two_views'}
    names=list(observations);rays=[ray(observations[n],cameras[n]) for n in names]
    A=sum(np.eye(3)-np.outer(d,d) for _,d in rays);b=sum((np.eye(3)-np.outer(d,d))@o for o,d in rays)
    if np.linalg.cond(A)>1e8:return {'accepted':False,'reason':'degenerate_baseline'}
    x=np.linalg.solve(A,b)
    def residual(X):return np.concatenate([project(X,cameras[n])[0]-observations[n] for n in names])
    fit=least_squares(residual,x,loss='huber',f_scale=max(pixel_sigma,0.5),max_nfev=100)
    x=fit.x;errors={n:float(np.linalg.norm(project(x,cameras[n])[0]-observations[n])) for n in names};depths={n:project(x,cameras[n])[1] for n in names}
    angles=[float(np.degrees(np.arccos(np.clip(np.dot(rays[i][1],rays[j][1]),-1,1)))) for i,j in itertools.combinations(range(len(names)),2)]
    covariance=np.linalg.pinv(fit.jac.T@fit.jac)*pixel_sigma**2+np.eye(3)*extrinsic_sigma_m**2
    sigma=float(np.sqrt(np.linalg.eigvalsh(covariance).max()));reason='ok'
    if min(depths.values())<=0:reason='behind_camera'
    elif max(angles)<min_angle_deg:reason='small_triangulation_angle'
    elif max(errors.values())>max_reprojection_px:reason='large_reprojection_error'
    elif sigma>max_uncertainty_m:reason='large_position_uncertainty'
    return dict(accepted=reason=='ok',reason=reason,position_base_m=x.tolist(),reprojection_errors_px=errors,depth_camera_z_m=depths,triangulation_angles_deg=angles,covariance_base_m2=covariance.tolist(),largest_axis_sigma_m=sigma,uncertainty_note='Linearized pixel-noise uncertainty plus configured isotropic extrinsic floor; semantic correspondence error is not captured.')
