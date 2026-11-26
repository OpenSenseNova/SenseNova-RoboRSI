"""Generic symmetric thin curved sheet + visible vertical support axis."""
import cv2,numpy as np
from scipy.optimize import least_squares
import rgb_geometry as G

def resample(q,n=60):
 q=np.asarray(q,float).reshape(-1,2);edge=np.roll(q,-1,axis=0)-q;length=np.linalg.norm(edge,axis=1);s=np.r_[0,np.cumsum(length)];u=np.linspace(0,s[-1],n,endpoint=False);ii=np.minimum(np.searchsorted(s,u,side='right')-1,len(q)-1);return q[ii]+edge[ii]*((u-s[ii])/np.maximum(length[ii],1e-9))[:,None]
def shape(p):
 center=p[:3];yaw,L,W,Hx,Hy,T=p[3:];u,v=np.meshgrid(np.linspace(-L/2,L/2,35),np.linspace(-W/2,W/2,25));z=Hx*(1-(2*u/L)**2)+Hy*(1-(2*v/W)**2);a=np.c_[u.ravel(),v.ravel(),z.ravel()];b=a.copy();b[:,2]-=T;local=np.r_[a,b];c,s=np.cos(yaw),np.sin(yaw);R=np.array([[c,-s,0],[s,c,0],[0,0,1]]);return local@R.T+center
def hull(p,cam):
 K,C=G.camera_arrays(cam);q=(shape(p)-C[:3,3])@C[:3,:3];uv=q@K.T;return cv2.convexHull((uv[:,:2]/uv[:,2,None]).astype(np.float32)).reshape(-1,2)
def distance(q,poly):return np.array([cv2.pointPolygonTest(np.asarray(poly,np.float32),(float(x),float(y)),True) for x,y in q])
def surface_z(p,xy):
 c,s=np.cos(p[3]),np.sin(p[3]);R=np.array([[c,-s],[s,c]]);uv=(np.asarray(xy)-p[:2])@R;return p[2]+p[6]*(1-(2*uv[:,0]/p[4])**2)+p[7]*(1-(2*uv[:,1]/p[5])**2)
def fit(contours,cameras,axes,supports):
 names=list(contours)
 if len(names)<3:return dict(accepted=False,reason='fewer_than_three_visible_sheet_boundaries')
 rectangles={n:cv2.minAreaRect(np.asarray(q,np.float32)) for n,q in contours.items()};coarse=G.triangulate({n:r[0] for n,r in rectangles.items()},cameras,pixel_sigma=20,max_reprojection_px=1000,max_uncertainty_m=1000)
 if 'position_base_m' not in coarse:return dict(accepted=False,reason='no_RGB_sheet_initialization')
 center=np.asarray(coarse['position_base_m']);yaws=[];lengths=[];widths=[]
 for n in names:
  rect=rectangles[n];r=cv2.boxPoints(rect);edges=np.roll(r,-1,axis=0)-r;v=edges[np.argmax(np.linalg.norm(edges,axis=1))];v/=np.linalg.norm(v);uv=G.project(center,cameras[n])[0];J=np.column_stack([(G.project(center+np.eye(3)[k]*.001,cameras[n])[0]-uv)/.001 for k in [0,1]])
  try:d=np.linalg.solve(J,v);d/=np.linalg.norm(d)
  except np.linalg.LinAlgError:continue
  yaws.append(np.arctan2(d[1],d[0]));lengths.append(max(rect[1])/np.linalg.norm(J@d));widths.append(min(rect[1])/np.linalg.norm(J@np.array([-d[1],d[0]])))
 yaw=np.arctan2(np.mean(np.sin(2*np.asarray(yaws))),np.mean(np.cos(2*np.asarray(yaws))))/2;L=float(np.median(lengths));W=float(np.median(widths));observed={n:resample(cv2.convexHull(np.asarray(q,np.float32))) for n,q in contours.items()};supports=np.asarray(supports).reshape(-1,3)
 low=np.r_[center-.07,yaw-.8,L*.5,W*.5,-W*.6,-W*.6,.0001];high=np.r_[center+.07,yaw+.8,L*1.8,W*1.8,W*.6,W*.6,W*.25]
 def residual(p,use,xyz):
  rr=[]
  for n in use:
   hh=hull(p,cameras[n]);rr.extend(distance(resample(hh),cv2.convexHull(np.asarray(contours[n],np.float32))));rr.extend(distance(observed[n],hh))
  for n,line in axes.items():
   a=G.project(p[:3],cameras[n])[0];b=G.project(p[:3]+[0,0,.1],cameras[n])[0];normal=np.array([b[1]-a[1],a[0]-b[0]]);normal/=np.linalg.norm(normal);rr.extend(2*((np.asarray(line)-a)@normal))
  if len(xyz):rr.extend((surface_z(p,xyz[:,:2])-xyz[:,2])/.0015)
  return np.asarray(rr)
 if len(supports)>=8:
  order=np.random.default_rng(148).permutation(len(supports));heldout=supports[order[:max(2,len(order)//4)]];train=supports[order[max(2,len(order)//4):]]
 else:heldout=np.empty((0,3));train=supports
 candidates=[]
 for hx,hy in [(0,0),(.12,0),(-.12,0),(0,.12),(0,-.12)]:
  initial=np.r_[center,yaw,L,W,W*hx,W*hy,W*.03];initial=np.clip(initial,low+1e-8,high-1e-8)
  result=least_squares(lambda p:residual(p,names,train),initial,bounds=(low,high),x_scale=np.r_[.03,.03,.03,.3,L,W,W*.2,W*.2,W*.05],diff_step=.001,loss='soft_l1',f_scale=1,max_nfev=160)
  error=np.abs(residual(result.x,names,np.empty((0,3))));held=np.abs(surface_z(result.x,heldout[:,:2])-heldout[:,2]) if len(heldout) else np.array([]);quality=float(np.median(error)+.2*np.quantile(error,.9)+(np.median(held)/.0015 if len(held) else 0));candidates.append((quality,result,error,held))
 _,selected,errors,held=min(candidates,key=lambda c:c[0]);p=selected.x;position=p[:3].copy();position[2]=surface_z(p,position[None,:2])[0]
 reason='ok'
 if np.median(errors)>2 or np.quantile(errors,.9)>5:reason='raw_RGB_boundary_or_axis_rejects_sheet'
 if len(held) and (np.median(held)>.0025 or np.max(held)>.005):reason='heldout_local_RGB_support_rejects_surface'
 if not axes:reason='no_visible_support_axis_constraint'
 return dict(accepted=reason=='ok',reason=reason,position_base_m=position.tolist(),parameters=p.tolist(),parameter_names=['x','y','edge_z','yaw','length','width','curvature_x','curvature_y','thickness'],views=names,axis_views=list(axes),raw_boundary_median_px=float(np.median(errors)),raw_boundary_p90_px=float(np.quantile(errors,.9)),local_RGB_support_count=len(supports),heldout_surface_errors_m=held.tolist(),alternative_fit_positions=[(f.x[:3]+[0,0,f.x[6]+f.x[7]]).tolist() for _,f,_,_ in candidates],
  model='Generic bilaterally symmetric curved thin sheet. Its center axis is constrained by a visible support post. Dimensions, curvature, pose fitted from RGB. Object CAD/native size never loaded.',uncertainty_note='Generic shape and symmetry priors are explicit. Boundary fit residual is not a complete position uncertainty estimate.')
