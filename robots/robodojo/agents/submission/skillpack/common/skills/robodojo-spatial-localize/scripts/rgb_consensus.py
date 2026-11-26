import itertools
import numpy as np
import rgb_geometry as G

def reject(reason,**kw):return dict(accepted=False,reason=reason,**kw)

def align(observations,cameras,reference):
 """Resolve unordered physical endpoints using only calibrated epipolar geometry."""
 out={reference:observations[reference]}
 for name,points in observations.items():
  if name==reference:continue
  permutations=list(itertools.permutations(points))
  out[name]=min(permutations,key=lambda ps:sum(G.epipolar_error(a,b,cameras[reference],cameras[name]) for a,b in zip(observations[reference],ps)))
 return out

def solve(observations,cameras,expected,consensus=True):
 observations={k:v for k,v in observations.items() if len(v)==expected}
 if len(observations)<2:return reject('fewer_than_two_semantically_valid_views')
 # Highest magnification is an image-calibration choice, independent of target GT.
 reference=max(observations,key=lambda n:float(np.asarray(cameras[n]['K'])[0,0]));observations=align(observations,cameras,reference)
 names=list(observations);candidates=[]
 subsets=[tuple(names)] if not consensus else [subset for size in range(len(names),1,-1) for subset in itertools.combinations(names,size)]
 for subset in subsets:
  parents=[G.triangulate({n:observations[n][j] for n in subset},cameras,pixel_sigma=1.5,extrinsic_sigma_m=.001,max_reprojection_px=2.,max_uncertainty_m=.010) for j in range(expected)]
  if not all(p['accepted'] for p in parents):continue
  errors={n:max(float(np.linalg.norm(G.project(p['position_base_m'],cameras[n])[0]-observations[n][j])) for j,p in enumerate(parents)) for n in names}
  support=[n for n in names if errors[n]<=2.]
  # Prefer explanations supported by more views, then independent held-out fit.
  heldout=[errors[n] for n in names if n not in subset]
  quality=(-len(support),-len(subset),float(np.median(heldout)) if heldout else max(errors.values()),max(p['largest_axis_sigma_m'] for p in parents))
  candidates.append((quality,parents,subset,errors,support))
 if not candidates:return reject('no_calibrated_consistent_subset',views=names)
 quality,parents,subset,errors,support=min(candidates,key=lambda c:c[0]);position=np.mean([p['position_base_m'] for p in parents],axis=0)
 return dict(accepted=True,reason='ok',position_base_m=position.tolist(),parents=parents,views=list(subset),support_views=support,reprojection_all_views_px=errors,
  largest_axis_sigma_m=max(p['largest_axis_sigma_m'] for p in parents),selection_quality=list(quality),candidate_count=len(candidates),
  uncertainty_note='Conditional pixel-noise covariance only; coherent semantic errors are not covered. Consensus of two views without a third supporting observation is weak evidence.')
