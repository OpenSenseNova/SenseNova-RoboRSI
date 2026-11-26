"""Run a packaged calibrated observation through geometry and task memory."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import uuid
from space_memory import load, import_result, save
from annotation_input import AnnotationError, normalize_targets


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',required=True,type=Path,help='Use observe.spatial_localization.case_path unchanged')
    parser.add_argument('--targets',type=Path,help='Annotation JSON: {targets:[...]}, a target list, or one target object')
    parser.add_argument('--object',dest='object_id')
    parser.add_argument('--point',default='contact')
    parser.add_argument('--description')
    parser.add_argument('--observation',action='append',default=[],metavar='CAMERA,U,V')
    parser.add_argument('--plane-z',type=float,help='Explicit known horizontal support height in environment metres')
    parser.add_argument('--height',type=float,help='Known physical point height above that plane; zero for contact')
    parser.add_argument('--memory',type=Path,default=Path('space.json'))
    parser.add_argument('--output',type=Path,help='Optional evidence directory; existing paths get a fresh sibling attempt')
    args=parser.parse_args()
    for flag,value in (('--plane-z',args.plane_z),('--height',args.height)):
        if value is not None and (not math.isfinite(value) or (flag=='--height' and value<0)):
            raise AnnotationError(flag,'a finite metre value (height must be nonnegative)',value)
    memory=load(args.memory)
    case_path=args.case.resolve(strict=True)
    case=json.loads(case_path.read_text())
    receipt=json.loads(case_path.with_suffix('.meta.json').read_text())
    latest_path=case_path.parent/'latest_spatial.json'
    # A fixed read-only helper reads the host's complete append-only pointer.
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[5]))
    from runtime_records import read as read_host_state
    latest=read_host_state(latest_path)
    if latest['case_file']!=case_path.name or latest['control_step']!=receipt['control_step']:
        parser.error('This observation is stale; call observe and use its current case_path')
    if hashlib.sha256(case_path.read_bytes()).hexdigest()!=receipt['case_sha256']:
        parser.error('Observation case checksum differs; request a new observe result')
    if case.get('frame')!=memory['frame'] or case.get('units')!=memory['units']:
        parser.error('Observation and task memory frames differ')
    for name,camera in case['cameras'].items():
        image=case_path.parent/camera['image']
        if hashlib.sha256(image.read_bytes()).hexdigest()!=receipt['images'][name]:
            parser.error('Observation image checksum differs; request a new observe result')
        # Make a portable working copy without copying images or calibration math.
        camera['image']=str(image.resolve(strict=True))
    if args.targets:
        if args.object_id or args.observation:parser.error('Use --targets OR object/pixel annotations')
        targets=json.loads(args.targets.read_text())
    else:
        if not args.object_id or not args.description or not args.observation:
            parser.error('Supply --object, --description and --observation, or --targets')
        observations={}
        for item in args.observation:
            try:name,u,v=item.split(',');pixel=[float(u),float(v)]
            except ValueError:parser.error('Each --observation is CAMERA,U,V on the original image grid')
            if name not in case['cameras'] or name in observations:parser.error('Use distinct camera names from observe')
            observations[name]={'points_px':[pixel]}
        kind='support_plane' if args.plane_z is not None else 'direct'
        target=dict(id=args.point,object_id=args.object_id,description=args.description,
                    kind=kind,observations=observations)
        targets={'targets':[target]}
        if args.plane_z is None and len(observations)<2:
            parser.error('Unknown-height points need two views; known-plane points need --plane-z and --height')
        if args.plane_z is not None:
            target['height_above_plane_m']=args.height
    targets=normalize_targets(targets,case)
    plane_targets=[t for t in targets['targets'] if t.get('kind')=='support_plane']
    if args.plane_z is not None:
        if not plane_targets:
            raise AnnotationError('--plane-z','at least one kind: support_plane target', 'no plane targets')
        names={t['support_plane'] for t in plane_targets if t.get('support_plane')}
        if len(names)>1:
            raise AnnotationError('--plane-z','one named plane per batch; split batches with different planes',sorted(names))
        name=next(iter(names),'declared_support')
        plane=dict(frame='environment_origin',normal=[0,0,1],offset_m=args.plane_z,
                   height_sigma_m=0,source='Agent-declared known scene support height')
        planes=case.setdefault('support_planes',{})
        if name in planes and any(planes[name].get(k)!=plane[k] for k in ('frame','normal','offset_m')):
            raise AnnotationError('--plane-z','a prior consistent with the existing named plane',args.plane_z)
        planes.setdefault(name,plane)
        for target in plane_targets:target['support_plane']=name
    if args.height is not None and not plane_targets:
        raise AnnotationError('--height','at least one support_plane target','no plane targets')
    for index,target in enumerate(targets['targets']):
        if target.get('kind')!='support_plane':continue
        if target.get('height_above_plane_m') is None and args.height is not None:
            target['height_above_plane_m']=args.height
        name=target.get('support_plane')
        if name not in case.get('support_planes',{}):
            raise AnnotationError(f'targets[{index}].support_plane',
                'a declared plane; pass --plane-z Z for this batch, or use multi-view direct for unknown height',name,
                'named_support_plane_required')
        if target.get('height_above_plane_m') is None:
            raise AnnotationError(f'targets[{index}].height_above_plane_m',
                'an explicitly known height in metres (0 only for contact), or --height H',None)
    requested=args.output or Path('localization')/('step-'+str(receipt['control_step'])+'-'+uuid.uuid4().hex[:8])
    root=requested.resolve()
    if root.is_relative_to(Path(__file__).resolve().parents[1]):
        raise AnnotationError('--output','an evidence directory outside the skill package',str(root))
    root.parent.mkdir(parents=True,exist_ok=True)
    while True:
        try:
            root.mkdir()
            break
        except FileExistsError:
            root=requested.resolve().with_name(requested.name+'-attempt-'+uuid.uuid4().hex[:8])
    (root/'case.json').write_text(json.dumps(case,allow_nan=False))
    (root/'targets.json').write_text(json.dumps(targets,allow_nan=False))
    geometry=root/'geometry'
    child=subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('localize_rgb.py')),
        '--case',str(root/'case.json'),'--targets',str(root/'targets.json'),'--output',str(geometry)],
        capture_output=True,text=True)
    if child.returncode:
        failure={'status':'input_rejected','detail':child.stderr.strip().splitlines()[-1][:2000] if child.stderr.strip() else 'Geometry process failed',
            'attempt_dir':str(root),'next_action':'Correct the named field in annotations and retry; evidence paths are allocated automatically. Keep the current case_path unless the robot moved.'}
        (root/'error.json').write_text(json.dumps(failure))
        print(json.dumps(failure))
        return 2
    result_path=geometry/'points3d.json';result=json.loads(result_path.read_text())
    if read_host_state(latest_path)!=latest:
        print(json.dumps({'status':'stale_observation','attempt_dir':str(root),
            'next_action':'The robot moved during localization; observe again before updating spatial memory.'}))
        return 2
    # Reload after geometry so unrelated current memory edits are preserved.
    memory=load(args.memory)
    import_result(memory,result,evidence=str(result_path),step=receipt['control_step'])
    save(args.memory,memory)
    print(json.dumps({'status':'localized' if any(r['estimate'].get('accepted') for r in result['points']) else 'rejected',
        'control_step':receipt['control_step'],'memory':str(args.memory),'result':str(result_path),
        'attempt_dir':str(root),
        'points':[{'object_id':r['object_id'],'point':r['id'],'accepted':r['estimate'].get('accepted'),
                   'position_m':r['estimate'].get('position_base_m'),'reason':r['estimate'].get('reason')}
                  for r in result['points']]},allow_nan=False))
    return 0


if __name__=='__main__':
    try:
        code=main()
    except AnnotationError as error:
        print(json.dumps({'status':'input_rejected',**error.details,
            'next_action':'Correct this field and retry. Use SKILL.md or inputs.md for the annotation example.'}))
        code=2
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({'status':'host_input_error','detail':str(error)[:1000],
            'path':str(error.filename) if isinstance(error,OSError) and error.filename else None,
            'next_action':'Use the exact case_path returned by observe and the result path returned by localization. Create annotation files before reading them; do not guess artifact paths.'}))
        code=2
    raise SystemExit(code)
