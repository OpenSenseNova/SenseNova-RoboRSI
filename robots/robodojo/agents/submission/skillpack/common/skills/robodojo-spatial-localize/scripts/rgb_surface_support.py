"""RGB mask refinement and local SIFT surface supports."""
import cv2,numpy as np
import rgb_geometry as G
from local_rgb_features import sift,sift_match,calibrated_matches

def refine_mask(image,polygon):
 polygon=np.asarray(polygon,np.int32);mask=np.zeros(image.shape[:2],np.uint8);cv2.fillPoly(mask,[polygon],255);kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15));inside=cv2.erode(mask,kernel);outside=cv2.dilate(mask,kernel);labels=np.where(mask>0,cv2.GC_PR_FGD,cv2.GC_PR_BGD).astype('uint8');labels[outside==0]=cv2.GC_BGD;labels[inside>0]=cv2.GC_FGD
 if np.count_nonzero(inside)<10:raise ValueError('polygon_too_small_for_RGB_refinement')
 cv2.grabCut(image,labels,None,np.zeros((1,65)),np.zeros((1,65)),4,cv2.GC_INIT_WITH_MASK);refined=np.isin(labels,[cv2.GC_FGD,cv2.GC_PR_FGD]).astype('uint8')*255;cs,_=cv2.findContours(refined,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
 if not cs:raise ValueError('no_RGB_refined_sheet')
 contour=max(cs,key=cv2.contourArea);out=np.zeros_like(mask);cv2.drawContours(out,[contour],-1,255,-1)
 return out,contour.reshape(-1,2)

def triangulated_supports(images,masks,cameras,seed=None):
 reference=max(masks,key=lambda n:np.count_nonzero(masks[n]));mom=cv2.moments(masks[reference]);point=np.array([mom['m10']/mom['m00'],mom['m01']/mom['m00']]) if seed is None else np.asarray(seed)
 r0=masks[reference].copy();yy,xx=np.indices(r0.shape);r0[np.hypot(xx-point[0],yy-point[1])>48]=0;f0=sift(images[reference],r0,point);records=[];positions=[];source_pixels=[]
 for name in masks:
  if name==reference:continue
  f1=sift(images[name],masks[name]);indices=sift_match(f0,f1);aa,bb,meta=calibrated_matches(f0,f1,indices,cameras[reference],cameras[name]);meta.update(source=reference,target=name,source_features=f0['xy'].tolist(),source_matches=aa.tolist(),target_matches=bb.tolist());records.append(meta)
  for a,b in zip(aa,bb):
   result=G.triangulate({'a':a,'b':b},{'a':cameras[reference],'b':cameras[name]},pixel_sigma=.8,extrinsic_sigma_m=.001,max_reprojection_px=1.5,max_uncertainty_m=.015)
   if result['accepted']:positions.append(result['position_base_m']);source_pixels.append(a)
 return np.asarray(positions),dict(reference=reference,query_center_px=point.tolist(),records=records,source_pixels=np.asarray(source_pixels).tolist())
