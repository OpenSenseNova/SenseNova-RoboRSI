"""RGB surface boundaries instead of texture keypoints for featureless centers."""
import cv2,numpy as np

def diagonal_center(q):
 q=np.asarray(q,float);a=q[2]-q[0];b=q[3]-q[1];t=np.linalg.solve(np.c_[a,-b],q[1]-q[0])[0];return q[0]+t*a

def ellipse(image,seed):
 seed=np.asarray(seed,float);lo=np.maximum([0,0],np.floor(seed-64).astype(int));hi=np.minimum([image.shape[1],image.shape[0]],np.ceil(seed+65).astype(int));x,y=lo;xx,yy=hi;patch=image[y:yy,x:xx];lab=cv2.cvtColor(patch,cv2.COLOR_BGR2LAB).astype(float)
 u,v=np.clip(np.round(seed-lo).astype(int),[1,1],[patch.shape[1]-2,patch.shape[0]-2]);proto=np.median(lab[v-1:v+2,u-1:u+2],axis=(0,1));dist=np.linalg.norm(lab[:,:,1:]-proto[1:],axis=2);candidates=[]
 for threshold in [8,12,18,24]:
  mask=(dist<threshold).astype('uint8')*255;mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8));cs,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
  for c in cs:
   q=c.reshape(-1,2)
   if len(q)<12 or np.any(q[:,0]<=0) or np.any(q[:,1]<=0) or np.any(q[:,0]>=patch.shape[1]-1) or np.any(q[:,1]>=patch.shape[0]-1):continue
   if cv2.pointPolygonTest(c,(float(u),float(v)),False)<0:continue
   center,diameters,angle=cv2.fitEllipse(q.astype(np.float32));a,b=np.asarray(diameters)/2;c0=np.asarray(center)+lo
   if min(a,b)<2 or max(a,b)>48 or np.linalg.norm(c0-seed)>12:continue
   theta=np.radians(angle);R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]);z=(q-np.asarray(center))@R/np.array([a,b]);residual=np.abs(np.linalg.norm(z,axis=1)-1)*min(a,b)
   if np.median(residual)>.9 or np.quantile(residual,.9)>2:continue
   candidates.append((float(np.median(residual))+.15*np.linalg.norm(c0-seed),c0,dict(method='adaptive_Lab_boundary_ellipse',threshold=threshold,diameters_px=list(diameters),median_residual_px=float(np.median(residual)),contour=(q+lo).tolist())))
 return min(candidates,key=lambda c:c[0])[1:] if candidates else None

def colored_surface_center(image,item,definition):
 colors={'green':(45,85),'blue':(95,135),'yellow':(22,38)};requested=[name for name in colors if name in definition.lower()]
 if len(requested)!=1:return None
 loh,hih=colors[requested[0]];hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);mask=((hsv[:,:,0]>=loh)&(hsv[:,:,0]<=hih)&(hsv[:,:,1]>90)&(hsv[:,:,2]>55)).astype('uint8')*255
 box=np.asarray(item['bbox_px']);center=(box[:2]+box[2:])/2;half=np.maximum((box[2:]-box[:2])*.75,24);lo=np.maximum([0,0],np.floor(center-half).astype(int));hi=np.minimum([image.shape[1],image.shape[0]],np.ceil(center+half).astype(int));roi=np.zeros_like(mask);roi[lo[1]:hi[1],lo[0]:hi[0]]=255;mask&=roi
 mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8));cs,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);candidates=[]
 for c in cs:
  area=cv2.contourArea(c)
  if area<30:continue
  rect=cv2.minAreaRect(c);sizes=np.asarray(rect[1]);aspect=max(sizes)/max(min(sizes),1)
  if aspect<2.2:continue
  hull=cv2.convexHull(c);perimeter=cv2.arcLength(hull,True);quads=[]
  for epsilon in [.01,.02,.03,.04,.05]:
   q=cv2.approxPolyDP(hull,epsilon*perimeter,True).reshape(-1,2)
   if len(q)==4 and cv2.isContourConvex(q):
    try:quads.append((diagonal_center(q),q))
    except np.linalg.LinAlgError:pass
  if not quads:continue
  position=np.asarray(rect[0]);candidates.append((area,position,dict(method='RGB_colored_surface_oriented_rectangle_center',color=requested[0],aspect=float(aspect),corners=quads[0][1].tolist(),contour=c.reshape(-1,2).tolist())))
 if not candidates:return None
 _,position,evidence=max(candidates,key=lambda c:c[0]);return position,evidence

def measure(image,item,target):
 result=dict(item);definition=target['physical_definition'];kind=item['kind'];points=np.asarray(item['points'],float);evidence=[]
 # Elongated surface centers are supported by the entire same-color boundary.
 # A model's misplaced seed does not override the requested semantic color.
 center_request=('center' in definition.lower() or 'crown' in definition.lower()) and len(points)==1
 surface=colored_surface_center(image,item,definition) if center_request else None
 if surface is not None:
  p,meta=surface;points=np.asarray([p]);evidence=[meta]
 elif len(points)==1 and (kind=='circular_face_center' or 'cap center' in definition.lower() or 'circular end face' in definition.lower()):
  measured=ellipse(image,points[0])
  if measured is not None:p,meta=measured;points=np.asarray([p]);evidence=[meta]
 result.update(points=points.tolist(),measurement_evidence=evidence or [{'method':'semantic_point_preserved'}]);return result
