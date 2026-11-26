"""Lossless annotation normalization and actionable format diagnostics."""
import copy
import math
import json


class AnnotationError(ValueError):
    def __init__(self, field, expected, received, code='invalid_annotation'):
        try:
            json.dumps(received, allow_nan=False)
        except (TypeError, ValueError):
            received = repr(received)
        self.details = dict(code=code, field=field, expected=expected, received=received)
        super().__init__(f'{field}: expected {expected}; received {received!r}')


def require_fields(value, allowed, field):
    if not isinstance(value, dict):
        raise AnnotationError(field, 'a JSON object', type(value).__name__)
    extra = sorted(set(value) - set(allowed))
    if extra:
        raise AnnotationError(field, f'only these fields: {", ".join(sorted(allowed))}', extra,
                              'unexpected_field')


def normalize_targets(value, case):
    """Accept equivalent containers; never infer geometry, units or pixel scaling."""
    value = copy.deepcopy(value)
    if isinstance(value, list):
        value = {'targets': value}
    elif isinstance(value, dict) and 'object_id' in value and 'targets' not in value:
        value = {'targets': [value]}
    require_fields(value, {'targets', 'frame', 'units'}, 'annotations')
    for name in ('frame', 'units'):
        if name in value and value[name] != case[name]:
            raise AnnotationError(f'annotations.{name}', case[name], value[name], 'frame_mismatch')
    targets = value.get('targets')
    if not isinstance(targets, list) or not targets:
        raise AnnotationError('annotations.targets', 'a nonempty list, e.g. {"targets":[{...}]}',
                              targets, 'targets_list_required')
    allowed = {'id', 'point', 'object_id', 'description', 'kind', 'color', 'observations', 'regions',
               'surface_method', 'support_plane', 'height_above_plane_m', 'target_height_sigma_m',
               'plane_options'}
    seen = set()
    for index, target in enumerate(targets):
        field = f'targets[{index}]'
        require_fields(target, allowed, field)
        if 'point' in target:
            if 'id' in target and target['id'] != target['point']:
                raise AnnotationError(field + '.point', 'the same value as id, or omit point', target['point'])
            target['id'] = target.pop('point')
        for name in ('id', 'object_id', 'description'):
            if not isinstance(target.get(name), str) or not target[name].strip():
                raise AnnotationError(field + '.' + name, 'a nonempty semantic name / description', target.get(name))
        kinds={'support_plane','direct','midpoint','circle_center','curved_strip_crown','axis_surface_intersection'}
        if not isinstance(target.get('kind'),str) or target['kind'] not in kinds:
            raise AnnotationError(field+'.kind', f'one of {sorted(kinds)}; do not infer unknown target height', target.get('kind'))
        if 'support_plane' in target and (not isinstance(target['support_plane'],str) or not target['support_plane'].strip()):
            raise AnnotationError(field+'.support_plane','a nonempty plane name',target['support_plane'])
        for name in ('height_above_plane_m','target_height_sigma_m'):
            if name in target:
                number=target[name]
                if not isinstance(number,(int,float)) or isinstance(number,bool) or not math.isfinite(number) or number<0:
                    raise AnnotationError(field+'.'+name,'a finite nonnegative value in metres',number)
        if target['kind']!='support_plane' and any(k in target for k in ('support_plane','height_above_plane_m','target_height_sigma_m','plane_options')):
            raise AnnotationError(field,'plane fields only on kind: support_plane',target['kind'])
        identity = (target['object_id'], target['id'])
        if identity in seen:
            raise AnnotationError(field + '.id', 'one target per (object_id, id); combine views under observations',
                                  list(identity), 'duplicate_target_id')
        seen.add(identity)
        for section in ('observations', 'regions'):
            mapping = target.get(section, {})
            if not isinstance(mapping, dict):
                raise AnnotationError(field + '.' + section, 'an object keyed by camera name', mapping)
            for name, observation in mapping.items():
                obs_field = f'{field}.{section}.{name}'
                if name not in case['cameras']:
                    raise AnnotationError(obs_field, f'camera in {list(case["cameras"])}', name,
                                          'unknown_observation_camera')
                require_fields(observation, {'points_px', 'bbox_px'} if section == 'observations'
                               else {'polygon_px', 'axis_px'}, obs_field)
                for key in ('points_px', 'polygon_px', 'axis_px'):
                    if key not in observation:
                        continue
                    points = observation[key]
                    if key == 'points_px' and isinstance(points, list) and len(points) == 2 and all(
                            isinstance(v, (float, int)) and not isinstance(v, bool) for v in points):
                        points = observation[key] = [points]
                    if not isinstance(points, list):
                        raise AnnotationError(obs_field + '.' + key, '[[u,v], ...] on original RGB', points)
                    for pi, point in enumerate(points):
                        if not isinstance(point, list) or len(point) != 2 or not all(
                                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                                for v in point):
                            raise AnnotationError(f'{obs_field}.{key}[{pi}]', 'a finite [u,v] pixel pair', point,
                                                  'invalid_pixel_shape')
    return {'targets': targets}
