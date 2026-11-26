"""Fit a generic curved strip to RGB silhouettes; no asset geometry/dimensions.

Parameters (pose, length, width, crown rise, thickness, corner roundness) are
estimated from calibrated image contours. The semantic crown is a model point,
not an image centroid. This prior must be declared; it is not general freeform MVS.
"""
import cv2,numpy as np
from scipy.optimize import least_squares
import rgb_geometry as G

def resample(q,n=64):
 q=np.asarray(q,float).reshape(-1,2);edges=np.roll(q,-1,axis=0)-q;length=np.linalg.norm(edges,axis=1);s=np.r_[0,np.cumsum(length)];u=np.linspace(0,s[-1],n,endpoint=False);idx=np.minimum(np.searchsorted(s,u,side='right')-1,len(q)-1);return q[idx]+edges[idx]*((u-s[idx])/np.maximum(length[idx],1e-9))[:,None]

def shape(parameters):
 center=np.asarray(parameters[:3]);yaw,L,W,H,T,r=parameters[3:];a=np.linspace(-L/2,L/2,35);b=np.linspace(-W/2,W/2,19);uu,vv=np.meshgrid(a,b);rad=r*W
 if rad>1e-5:
  dx=np.maximum(np.abs(uu)-(L/2-rad),0);dy=np.maximum(np.abs(vv)-(W/2-rad),0);keep=dx*dx+dy*dy<=rad*rad+1e-12
 else:keep=np.ones_like(uu,bool)
 u,v=uu[keep],vv[keep];z=H*np.maximum(0,1-(2*v/W)**2)
 local=np.concatenate([np.c_[u,v,z],np.c_[u,v,np.full(len(u),-T)]],axis=0);c,s=np.cos(yaw),np.sin(yaw);R=np.array([[c,-s,0],[s,c,0],[0,0,1]])
 return local@R.T+center

def projected_hull(parameters,cam):
 K,C=G.camera_arrays(cam);xyz=(shape(parameters)-C[:3,3])@C[:3,:3];uv=xyz@K.T;uv=uv[:,:2]/uv[:,2,None];return cv2.convexHull(uv.astype(np.float32)).reshape(-1,2)

def distances(points,polygon):
 polygon=np.asarray(polygon,np.float32);return np.asarray([cv2.pointPolygonTest(polygon,(float(p[0]),float(p[1])),True) for p in points])

def fit(contours,cameras):
 if len(contours)<3:return dict(accepted=False,reason='fewer_than_three_surface_silhouettes')
 names=list(contours);rectangles={n:cv2.minAreaRect(np.asarray(q,np.float32)) for n,q in contours.items()};centers={n:rectangles[n][0] for n in names};coarse=G.triangulate(centers,cameras,pixel_sigma=20,max_reprojection_px=1000,max_uncertainty_m=1000)
 if 'position_base_m' not in coarse:return dict(accepted=False,reason='no_coarse_silhouette_volume')
 center=np.asarray(coarse['position_base_m']);yaws=[];lengths=[];widths=[]
 for n in names:
  K,C=G.camera_arrays(cameras[n]);r=cv2.boxPoints(rectangles[n]);edges=np.roll(r,-1,axis=0)-r;v=edges[np.argmax(np.linalg.norm(edges,axis=1))];v=v/np.linalg.norm(v);uv=G.project(center,cameras[n])[0];J=np.column_stack([(G.project(center+np.eye(3)[k]*.001,cameras[n])[0]-uv)/.001 for k in [0,1]])
  try:
   d=np.linalg.solve(J,v);d/=np.linalg.norm(d);yaw=np.arctan2(d[1],d[0]);yaws.append(yaw);long=max(rectangles[n][1])/np.linalg.norm(J@d);perp=np.array([-d[1],d[0]]);short=min(rectangles[n][1])/np.linalg.norm(J@perp);lengths.append(long);widths.append(short)
  except np.linalg.LinAlgError:pass
 if not yaws:return dict(accepted=False,reason='no_horizontal_axis_initialization')
 yaw=np.arctan2(np.mean(np.sin(2*np.asarray(yaws))),np.mean(np.cos(2*np.asarray(yaws))))/2;L=float(np.median(lengths));W=float(np.median(widths));observed={n:resample(cv2.convexHull(np.asarray(contours[n],np.float32))) for n in names}
 lo=np.r_[center-.06,yaw-.7,L*.5,W*.4,0,0,0];hi=np.r_[center+.06,yaw+.7,L*1.7,W*1.8,W*.6,W*.6,.45]
 def residual(p,use):
  out=[]
  for n in use:
   hull=projected_hull(p,cameras[n]);out.extend(distances(resample(hull),cv2.convexHull(np.asarray(contours[n],np.float32))));out.extend(distances(observed[n],hull))
  return np.asarray(out)
 fits=[]
 for rise in [.04,.15,.30]:
  p=np.r_[center,yaw,L,W,W*rise,W*.10,.12];p=np.clip(p,lo+1e-8,hi-1e-8)
  result=least_squares(lambda p:residual(p,names),p,bounds=(lo,hi),x_scale=np.r_[.03,.03,.03,.3,L,W,W*.2,W*.2,.2],diff_step=.001,loss='soft_l1',f_scale=1,max_nfev=140)
  errors=np.abs(residual(result.x,names));fits.append((float(np.median(errors)+.2*np.quantile(errors,.9)),result,errors))
 score,result,errors=min(fits,key=lambda x:x[0]);parameters=result.x;position=parameters[:3]+[0,0,parameters[6]]
 per_view={n:dict(median_px=float(np.median(np.abs(residual(parameters,[n])))),p90_px=float(np.quantile(np.abs(residual(parameters,[n])),.9))) for n in names}
 # Raw silhouettes provide validation independent of the initial semantic pixel.
 accepted=np.median(errors)<=1.5 and np.quantile(errors,.9)<=3
 return dict(accepted=bool(accepted),reason='ok' if accepted else 'raw_RGB_silhouette_rejects_curved_strip',position_base_m=position.tolist(),parameters=parameters.tolist(),parameter_names=['x','y','edge_z','yaw','length','width','crown_rise','thickness','corner_roundness_fraction'],views=names,raw_boundary_errors=per_view,median_boundary_error_px=float(np.median(errors)),p90_boundary_error_px=float(np.quantile(errors,.9)),
  model='Generic horizontal strip with parabolic convex upper cross-section and rounded rectangular footprint. All dimensions inferred from RGB. No asset/CAD lookup.',alternative_fit_positions=[(f.x[:3]+[0,0,f.x[6]]).tolist() for _,f,_ in fits],largest_axis_sigma_m=None,
  uncertainty_note='Silhouette fit residual does not certify freeform surface shape or crown height. Generic curved-strip prior is explicit; no conditional covariance claimed.')
