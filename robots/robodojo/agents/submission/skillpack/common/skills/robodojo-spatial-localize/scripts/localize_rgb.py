#!/usr/bin/env python3
"""Recover named physical points from RGB, calibrated cameras and Agent evidence."""
from pathlib import Path
import argparse,hashlib,json,sys
import cv2,numpy as np
import rgb_geometry as G
import support_plane
from annotation_input import AnnotationError, normalize_targets
from rgb_consensus import solve,reject
from rgb_measure import ellipse,colored_surface_center
from curved_strip import fit as fit_strip,projected_hull
from curved_plate import fit as fit_plate,hull
from rgb_surface_support import refine_mask,triangulated_supports

cv2.setNumThreads(1)
KINDS={'support_plane','direct','midpoint','circle_center','curved_strip_crown','axis_surface_intersection'}
def load(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def coordinate(p,shape,field='pixel'):
 p=np.asarray(p,float)
 if p.shape!=(2,) or not np.isfinite(p).all():raise AnnotationError(field,'a finite [u,v] pair',p.tolist(),'invalid_pixel_shape')
 if np.any(p<0) or p[0]>=shape[1] or p[1]>=shape[0]:raise AnnotationError(field,f'0 <= u < {shape[1]}, 0 <= v < {shape[0]} on original RGB; do not clip or guess a scale',p.tolist(),'coordinate_outside_original_RGB_grid')
 return p.tolist()
def validate(case,targets,root):
 if set(case)-{'frame','units','cameras','support_planes'}:raise ValueError('unexpected_case_field')
 if not isinstance(case.get('frame'),str) or case.get('units')!='meters':raise ValueError('named_Base_frame_and_meters_required')
 if not case.get('cameras'):raise ValueError('no_calibrated_RGB_cameras')
 if not isinstance(case.get('support_planes',{}),dict):raise ValueError('support_planes_mapping_required')
 for plane in case.get('support_planes',{}).values():support_plane.plane_in_base(plane,case['frame'])
 images={};paths={}
 for name,cam in case['cameras'].items():
  if set(cam)-{'image','K','T_base_camera_cv','distortion'}:raise ValueError('unexpected_camera_field')
  G.camera_arrays(cam);p=(root/cam['image']).resolve()
  if p.suffix.lower() not in {'.png','.jpg','.jpeg','.bmp','.ppm'} or any(t in str(p).lower() for t in ['/truth/','evaluator_private','depth','mesh']):raise ValueError('ordinary_raw_RGB_image_required')
  image=cv2.imread(str(p),cv2.IMREAD_UNCHANGED)
  if image is None or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:raise ValueError('uint8_three_channel_RGB_required')
  images[name]=image;paths[name]=p
 normalized=normalize_targets(targets,case)
 targets.clear();targets.update(normalized)
 ids=[]
 for ti,t in enumerate(targets['targets']):
  if set(t)-{'id','object_id','description','kind','color','observations','regions','surface_method','support_plane','height_above_plane_m','target_height_sigma_m','plane_options'}:raise ValueError('unexpected_target_field')
  if not all(isinstance(t.get(k),str) and t[k] for k in ['id','object_id','description']) or t.get('kind') not in KINDS:raise ValueError('named_semantic_target_and_supported_kind_required')
  if any(c not in images for c in set(t.get('observations',{}))|set(t.get('regions',{}))):raise ValueError('unknown_observation_camera')
  for n,obs in t.get('observations',{}).items():
   if set(obs)-{'points_px','bbox_px'}:raise ValueError('unexpected_observation_field')
   for pi,p in enumerate(obs.get('points_px',[])):coordinate(p,images[n].shape,f'targets[{ti}].observations.{n}.points_px[{pi}]')
   if 'bbox_px' in obs:
    b=np.asarray(obs['bbox_px'],float)
    if b.shape!=(4,) or not np.isfinite(b).all() or np.any(b<0) or np.any(b[2:]<=b[:2]) or b[2]>images[n].shape[1] or b[3]>images[n].shape[0]:raise ValueError('invalid_local_RGB_region')
  for n,region in t.get('regions',{}).items():
   if set(region)-{'polygon_px','axis_px'}:raise ValueError('unexpected_surface_evidence_field')
   for field in ['polygon_px','axis_px']:
    for p in region.get(field,[]):coordinate(p,images[n].shape)
   if 'polygon_px' in region and len(region['polygon_px'])<4:raise ValueError('sheet_outline_needs_at_least_four_vertices')
   if 'axis_px' in region and (len(region['axis_px'])!=2 or np.linalg.norm(np.asarray(region['axis_px'][0])-region['axis_px'][1])<8):raise ValueError('visible_axis_requires_two_separated_points')
  if t['kind']=='support_plane':support_plane.validate_target(t,case.get('support_planes',{}),case['frame'])
  elif any(k in t for k in ['support_plane','height_above_plane_m','target_height_sigma_m','plane_options']):raise ValueError('plane_fields_require_support_plane_kind')
  ids.append((t['object_id'],t['id']))
 if len(set(ids))!=len(ids):raise ValueError('duplicate_target_id')
 return images,paths

def quadratic(plate,points):
 if not plate.get('accepted'):return reject('sheet_axis_construction_not_accepted')
 points=np.asarray(points,float).reshape(-1,3);xy=np.asarray(plate['position_base_m'])[:2]
 if len(points)<8:return reject('fewer_than_eight_local_surface_supports')
 if cv2.pointPolygonTest(cv2.convexHull(points[:,:2].astype(np.float32)),tuple(map(float,xy)),True)<0:return reject('semantic_axis_outside_matched_surface_support_hull')
 origin=np.mean(points[:,:2],axis=0);scale=np.std(points[:,:2],axis=0)
 if min(scale)<.002:return reject('collinear_local_surface_supports')
 def design(q):
  u,v=((np.asarray(q)-origin)/scale).T;return np.c_[u*u,u*v,v*v,u,v,np.ones(len(u))]
 A=design(points[:,:2]);order=np.random.default_rng(1411).permutation(len(points));cut=max(2,len(points)//4);test,train=order[:cut],order[cut:]
 if np.linalg.matrix_rank(A[train])<6:return reject('quadratic_surface_not_identifiable')
 beta=np.linalg.lstsq(A[train],points[train,2],rcond=None)[0];errors=np.abs(A[test]@beta-points[test,2])
 if np.median(errors)>.0025 or max(errors)>.005:return reject('heldout_local_surface_rejects_quadratic',heldout_errors_m=errors.tolist())
 beta=np.linalg.lstsq(A,points[:,2],rcond=None)[0];z=float(design(xy[None])[0]@beta)
 return dict(accepted=True,reason='ok',position_base_m=[float(xy[0]),float(xy[1]),z],construction=plate,heldout_errors_m=errors.tolist(),support_count=len(points),model='Observed axis XY intersecting a local RGB-triangulated quadratic surface.')

def axis_surface(target,images,cameras,out):
 masks={};contours={};axes={};failures={}
 for name,region in target.get('regions',{}).items():
  if 'axis_px' in region:axes[name]=region['axis_px']
  if 'polygon_px' not in region or cv2.contourArea(np.asarray(region['polygon_px'],np.float32))<1000:continue
  try:
   mask,contour=refine_mask(images[name],region['polygon_px']);masks[name]=mask;contours[name]=contour;cv2.imwrite(str(out/(name+'_mask.png')),mask)
  except Exception as exc:failures[name]=str(exc)
 points=np.empty((0,3));evidence={}
 if len(masks)>=2:points,evidence=triangulated_supports(images,masks,cameras)
 save(out/'local_supports.json',dict(points_base_m=points.tolist(),evidence=evidence,mask_failures=failures));plate=fit_plate(contours,cameras,axes,points);save(out/'parametric_surface.json',plate)
 result=quadratic(plate,points) if target.get('surface_method','parametric')=='local_quadratic' else plate
 if 'parameters' in plate:
  for name in contours:
   image=images[name].copy();cv2.polylines(image,[contours[name].astype(np.int32)],True,(0,255,0),1);q=hull(np.asarray(plate['parameters']),cameras[name]);cv2.polylines(image,[np.round(q).astype(np.int32)],True,(0,0,255),1);cv2.imwrite(str(out/(name+'_surface_fit.png')),image)
 return result

def localize(target,images,cameras,out,case=None):
 if target['kind']=='support_plane':
  if case is None:raise ValueError('support_plane_case_required')
  return support_plane.localize(target,cameras,case.get('support_planes',{}),case['frame'])
 if target['kind']=='axis_surface_intersection':return axis_surface(target,images,cameras,out)
 if target['kind']=='curved_strip_crown':
  contours={}
  for name,obs in target.get('observations',{}).items():
   if 'bbox_px' not in obs:continue
   proposal=colored_surface_center(images[name],obs,target.get('color','')+' center')
   if proposal is not None:contours[name]=proposal[1]['contour']
  result=fit_strip(contours,cameras)
  if 'parameters' in result:
   for name in contours:
    image=images[name].copy();cv2.polylines(image,[np.asarray(contours[name],np.int32)],True,(0,255,0),1);q=projected_hull(np.asarray(result['parameters']),cameras[name]);cv2.polylines(image,[np.round(q).astype(np.int32)],True,(0,0,255),1);cv2.imwrite(str(out/(name+'_surface_fit.png')),image)
  return result
 observations={};measurements={}
 for name,obs in target.get('observations',{}).items():
  if 'points_px' not in obs:continue
  points=obs['points_px'];meta=[]
  if target['kind']=='circle_center' and len(points)==1:
   measured=ellipse(images[name],points[0])
   if measured is not None:p,evidence=measured;points=[p.tolist()];meta=[evidence]
  observations[name]=points;measurements[name]=dict(points_px=points,RGB_boundary_evidence=meta)
 save(out/'RGB_measurements.json',measurements)
 return solve(observations,cameras,2 if target['kind']=='midpoint' else 1)

def install_input_audit():
 reads=[];violations=[]
 def audit(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes,Path)):return
  p=str(args[0]);mode=args[1] if len(args)>1 else None
  if isinstance(mode,str) and not any(c in mode for c in ['r','+']):return
  if any(t in p.lower() for t in ['/truth/','/evaluator_private/','scoring_truth','point_results','.usd','.urdf']):violations.append(p);raise PermissionError('RGB runtime denied privileged scene/evaluation data')
  if p.lower().endswith(('.json','.png','.jpg','.jpeg','.npy','.npz')):reads.append(p)
 sys.addaudithook(audit);return reads,violations

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case',type=Path,required=True);parser.add_argument('--targets',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();out=args.output.resolve()
 if out.exists() or out.is_relative_to(Path(__file__).resolve().parents[1]):parser.error('Output must be a new path outside the skill package')
 reads,violations=install_input_audit();case_path=args.case.resolve();targets_path=args.targets.resolve();case=load(case_path);targets=normalize_targets(load(targets_path),case);images,paths=validate(case,targets,case_path.parent);out.mkdir(parents=True);save(out/'input_hashes.json',{str(p):sha(p) for p in [case_path,targets_path]+list(paths.values())});rows=[]
 for index,target in enumerate(targets['targets']):
  folder=out/f'target_{index:03d}';folder.mkdir()
  try:estimate=localize(target,images,case['cameras'],folder,case=case)
  except Exception as exc:estimate=reject(type(exc).__name__+': '+str(exc))
  row={k:target[k] for k in ['id','object_id','description','kind']};row['estimate']=estimate;rows.append(row)
  if estimate.get('accepted'):
   for name,image in images.items():
    overlay=image.copy();p,z=G.project(estimate['position_base_m'],case['cameras'][name])
    if z>0 and np.isfinite(p).all():cv2.drawMarker(overlay,tuple(np.round(p).astype(int)),(0,0,255),cv2.MARKER_CROSS,12,1)
    cv2.imwrite(str(folder/(name+'_target.png')),overlay)
 save(out/'points3d.json',dict(schema_version=3,frame=case['frame'],units='meters',points=rows,environment_depth_used=False,native_object_state_used=False));save(out/'input_audit.json',dict(data_reads=reads,forbidden_reads=violations));save(out/'runtime_freeze.json',{str(p):sha(p) for p in Path(__file__).parent.glob('*.py')});print(json.dumps(dict(requested=len(rows),returned=sum(r['estimate']['accepted'] for r in rows),output=str(out/'points3d.json'))))
if __name__=='__main__':main()
