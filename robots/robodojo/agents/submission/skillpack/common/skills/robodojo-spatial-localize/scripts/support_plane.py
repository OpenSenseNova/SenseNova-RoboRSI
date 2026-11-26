"""RGB pixel rays intersect an explicitly declared support plane; no scene access."""
import numpy as np
import rgb_geometry as G


def scalar(value, name, minimum=0., strictly_positive=False):
    if isinstance(value, (bool, str)):
        raise ValueError('invalid_'+name)
    x = float(value)
    if not np.isfinite(x) or x < minimum or (strictly_positive and x == minimum):
        raise ValueError('invalid_'+name)
    return x


def plane_in_base(plane, frame):
    allowed = {'frame', 'normal', 'offset_m', 'T_base_plane_frame',
               'height_sigma_m', 'extrinsic_sigma_m', 'source'}
    if set(plane)-allowed:
        raise ValueError('unexpected_support_plane_field')
    if not isinstance(plane.get('frame'), str) or not plane['frame']:
        raise ValueError('support_plane_frame_required')
    if not isinstance(plane.get('source'), str) or not plane['source'].strip():
        raise ValueError('support_plane_source_required')
    n = np.asarray(plane['normal'], float)
    if n.shape != (3,) or not np.isfinite(n).all() or not np.isclose(np.linalg.norm(n), 1., atol=1e-6):
        raise ValueError('support_plane_requires_unit_normal')
    d = float(plane['offset_m'])
    if not np.isfinite(d):
        raise ValueError('invalid_support_plane_offset')
    sigma = scalar(plane['height_sigma_m'], 'height_sigma_m')
    extrinsic = scalar(plane.get('extrinsic_sigma_m', 0.), 'extrinsic_sigma_m')
    if plane['frame'] != frame:
        if 'T_base_plane_frame' not in plane:
            raise ValueError('support_plane_frame_transform_required')
        T = np.asarray(plane['T_base_plane_frame'], float)
        if (T.shape != (4, 4) or not np.isfinite(T).all()
                or not np.allclose(T[3], [0, 0, 0, 1])
                or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(T[:3, :3]), 1., atol=1e-6)):
            raise ValueError('invalid_support_plane_SE3_transform')
        n = T[:3, :3] @ n
        d += float(n @ T[:3, 3])
    elif 'T_base_plane_frame' in plane:
        raise ValueError('same_frame_plane_must_not_supply_transform')
    return n, d, sigma, extrinsic


def validate_target(target, planes, frame):
    name = target.get('support_plane')
    if not isinstance(name, str) or name not in planes:
        raise ValueError('named_support_plane_required')
    plane_in_base(planes[name], frame)
    if 'height_above_plane_m' not in target:
        raise ValueError('explicit_target_height_above_plane_required')
    scalar(target['height_above_plane_m'], 'height_above_plane_m')
    scalar(target.get('target_height_sigma_m', 0.), 'target_height_sigma_m')
    obs = target.get('observations', {})
    if not obs or any(len(o.get('points_px', [])) != 1 for o in obs.values()):
        raise ValueError('support_plane_needs_one_physical_point_per_view')
    options = target.get('plane_options', {})
    allowed = {'pixel_sigma_px', 'max_position_sigma_m', 'min_ray_plane_angle_deg',
               'max_view_disagreement_m', 'max_reprojection_px'}
    if not isinstance(options, dict) or set(options)-allowed:
        raise ValueError('unexpected_plane_option')
    for k, v in options.items():
        scalar(v, k, strictly_positive=True)
    if options.get('min_ray_plane_angle_deg', 3.) >= 90.:
        raise ValueError('invalid_min_ray_plane_angle_deg')


def intersect(pixel, camera, normal, offset):
    origin, direction = G.ray(pixel, camera)
    denominator = float(normal @ direction)
    if abs(denominator) < 1e-12:
        raise ValueError('ray_parallel_to_support_plane')
    distance = (offset-float(normal @ origin))/denominator
    return origin+distance*direction, distance, abs(denominator)


def localize(target, cameras, planes, frame):
    validate_target(target, planes, frame)
    plane = planes[target['support_plane']]
    n, d, height_sigma, extrinsic_sigma = plane_in_base(plane, frame)
    height = float(target['height_above_plane_m'])
    height_sigma = float(np.hypot(height_sigma,target.get('target_height_sigma_m',0.)))
    d += height
    options = target.get('plane_options', {})
    pixel_sigma = options.get('pixel_sigma_px', 1.)
    minimum_angle = options.get('min_ray_plane_angle_deg', 3.)
    max_sigma = options.get('max_position_sigma_m', .01)
    max_disagreement = options.get('max_view_disagreement_m', .01)
    max_reprojection = options.get('max_reprojection_px', 3.)
    prior = dict(support_plane=target['support_plane'],declared_plane=plane,
                 normal_base=n.tolist(),target_plane_offset_base_m=d,
                 height_above_plane_m=height,target_height_sigma_m=target.get('target_height_sigma_m',0.),
                 assumption='The annotated physical point lies at the declared height along the support normal; object contact alone does not establish surface-point height.')
    evidence = {}
    def reject(reason, **extra):
        return dict(accepted=False,reason=reason,method='support_plane_ray_intersection',
                    priors=prior,per_view=evidence,**extra)
    points, covariances, height_derivatives = [], [], []
    for name, obs in target['observations'].items():
        pixel = np.asarray(obs['points_px'][0],float)
        try:
            point, distance, sine = intersect(pixel,cameras[name],n,d)
        except ValueError as exc:
            return reject(str(exc))
        angle = float(np.degrees(np.arcsin(np.clip(sine,0.,1.))))
        evidence[name] = dict(position_base_m=point.tolist(),ray_distance_m=distance,
                              ray_plane_angle_deg=angle)
        if distance <= 0:
            return reject('support_plane_behind_camera')
        if angle < minimum_angle:
            return reject('grazing_support_plane_ray')
        eps = 1e-3
        jac = np.column_stack([
            (intersect(pixel+np.eye(2)[j]*eps,cameras[name],n,d)[0]
             -intersect(pixel-np.eye(2)[j]*eps,cameras[name],n,d)[0])/(2*eps)
            for j in range(2)])
        derivative = (intersect(pixel,cameras[name],n,d+eps)[0]
                      -intersect(pixel,cameras[name],n,d-eps)[0])/(2*eps)
        covariance = jac@jac.T*pixel_sigma**2
        points.append(point);covariances.append(covariance);height_derivatives.append(derivative)
    points = np.asarray(points)
    disagreement = max((float(np.linalg.norm(a-b)) for a in points for b in points),default=0.)
    if disagreement > max_disagreement:
        return reject('inconsistent_plane_intersections',max_view_disagreement_m=disagreement)
    weights = 1/np.maximum([np.trace(v) for v in covariances],1e-18)
    weights /= weights.sum()
    point = weights@points
    # Height error is shared by all views and must not shrink as independent noise.
    dh = weights@np.asarray(height_derivatives)
    covariance = sum(w*w*cov for w,cov in zip(weights,covariances))
    covariance += height_sigma**2*np.outer(dh,dh)+np.eye(3)*extrinsic_sigma**2
    sigma = float(np.sqrt(max(0.,np.linalg.eigvalsh(covariance).max())))
    projections = {name:G.project(point,cameras[name]) for name in evidence}
    errors = {name:float(np.linalg.norm(projections[name][0]-target['observations'][name]['points_px'][0])) for name in evidence}
    reason = 'ok'
    if any(z <= 0 for _,z in projections.values()):reason = 'support_plane_behind_camera'
    elif max(errors.values()) > max_reprojection:reason = 'large_plane_reprojection_error'
    elif sigma > max_sigma:reason = 'large_plane_position_uncertainty'
    return dict(accepted=reason=='ok',reason=reason,method='support_plane_ray_intersection',
                position_base_m=point.tolist(),covariance_base_m2=covariance.tolist(),
                largest_axis_sigma_m=sigma,reprojection_errors_px=errors,
                max_view_disagreement_m=disagreement,view_count=len(evidence),
                priors=prior,per_view=evidence,
                uncertainty_note='Conditional linearized pixel and shared plane-height uncertainty, plus the supplied isotropic extrinsic floor. Plane tilt, calibration correlations, semantic errors and wrong target height are not estimated. Single-view acceptance is not independent validation of the plane prior.')
