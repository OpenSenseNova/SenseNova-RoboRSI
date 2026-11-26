"""Read mixed text/RGB replies and validate memory/crop coordinates without motion."""
import argparse,json,math

def parse_reply(reply):
    # Code mode can serialize the MCP envelope; a direct local call can instead
    # return the public payload. Never regex JSON.stringify(envelope): quotes
    # inside content[].text are escaped and valid status/step fields disappear.
    if isinstance(reply,str):
        try:reply=json.loads(reply)
        except ValueError as error:raise ValueError('Expected a complete JSON reply') from error
    if isinstance(reply,dict) and 'content' not in reply:
        return {'json_values':[reply], 'text':[], 'RGB_blocks':[],
                'motion_executed':'not_determined_by_reply_parsing'}
    blocks=reply.get('content')if isinstance(reply,dict)else reply
    if not isinstance(blocks,list):raise ValueError('Expected a tool content array or public JSON payload')
    values=[];text=[];images=[]
    for block in blocks:
        if not isinstance(block,dict):raise ValueError('Tool content blocks must be objects')
        if block.get('type')in('image','image_url','image_paths'):images.append(block);continue
        value=block.get('text')
        if not isinstance(value,str):continue
        text.append(value)
        try:values.append(json.loads(value))
        except ValueError:pass
    return {'json_values':values,'text':text,'RGB_blocks':images,'motion_executed':'not_determined_by_reply_parsing'}


def normalize_angles(pitch_deg,roll_deg,yaw_deg):
    """Equivalent Rz(yaw) Ry(roll) Rx(pitch), within the public act ranges.

    This only re-expresses the same rotation; it never clamps a desired tilt,
    executes motion, or establishes that a proposed pose is collision-free.
    """
    values=(pitch_deg,roll_deg,yaw_deg)
    if any(type(v)not in(int,float)or not math.isfinite(v)for v in values):
        raise ValueError('Angles must be finite numbers')
    def wrap(v):return (v+180)%360-180
    pitch,roll,yaw=map(wrap,values)
    if roll>90:pitch,roll,yaw=pitch+180,180-roll,yaw+180
    elif roll< -90:pitch,roll,yaw=pitch+180,-180-roll,yaw+180
    return dict(pitch_deg=wrap(pitch),roll_deg=roll+0.,yaw_deg=wrap(yaw))


def require_points(points,*,minimum=1,dimensions=2):
    """Guard empty/invalid detections before indexing or geometric reduction."""
    if type(minimum)is not int or minimum<1 or type(dimensions)is not int or dimensions<1:
        raise ValueError('Positive integer minimum and dimensions required')
    if not isinstance(points,(list,tuple))or len(points)<minimum:
        raise ValueError('Insufficient detected points; reobserve or change view, do not form a motion target')
    if any(not isinstance(p,(list,tuple))or len(p)!=dimensions or
           any(type(v)not in(int,float)or not math.isfinite(v)for v in p)for p in points):
        raise ValueError('Detected points require finite coordinates of the declared dimension')
    return points


def localization_point(value,*,allow_estimated=False):
    """Require usable public localization before doing arithmetic with XYZ."""
    if type(allow_estimated)is not bool:raise ValueError('allow_estimated must be boolean')
    statuses=('success','estimated')if allow_estimated else ('success',)
    if (not isinstance(value,dict)or value.get('status')not in statuses or
            value.get('coordinate_frame')!='environment_origin'or value.get('units')!='meters'):
        raise ValueError('Localization was rejected or uncertain; inspect status and relocalize')
    point=value.get('position_m')
    require_points([point],dimensions=3)
    return point


def image_pixel(pixel,width,height):
    """Validate original-image pixel centres, including exclusive upper bounds."""
    if any(type(v)is not int or v<1 for v in (width,height)):
        raise ValueError('Positive integer original-image dimensions required')
    require_points([pixel],dimensions=2)
    if not 0<=pixel[0]<width or not 0<=pixel[1]<height:
        raise ValueError('Pixel outside original image: require 0 <= u < width and 0 <= v < height')
    return pixel

def memory_point(memory,object_id,point_id,*,task):
    if (type(memory.get('schema_version'))is not int or memory['schema_version']!=1
            or not isinstance(task,str)or not task or memory.get('task')!=task
            or memory.get('frame')!='environment_origin'or memory.get('units')!='meters'
            or not isinstance(memory.get('objects'),dict)):
        raise ValueError('Spatial memory schema/task/frame/units do not match the current episode')
    value=memory['objects'][object_id][point_id]
    if not isinstance(value,dict)or value.get('status')!='localized':raise ValueError('Point is stale or not localized; observe and relocalize')
    position=value.get('position_m')
    if not isinstance(position,list)or len(position)!=3 or any(type(v)not in(int,float)or not math.isfinite(v)for v in position):raise ValueError('Localized point requires three finite position_m coordinates')
    return value

def crop_bounds(box,width,height):
    if len(box)!=4 or any(type(v)not in(int,float)or not math.isfinite(v)for v in box):raise ValueError('Crop requires four finite numeric coordinates')
    x0,y0,x1,y1=box
    if not 0<=x0<x1<=width or not 0<=y0<y1<=height:raise ValueError('Crop is outside original RGB bounds; do not silently clip')
    return [x0,y0,x1,y1]

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('reply_json',nargs='?')
    p.add_argument('--angles',nargs=3,type=float,metavar=('PITCH','ROLL','YAW'))
    args=p.parse_args()
    if (args.reply_json is None)==(args.angles is None):p.error('Choose reply_json or --angles')
    if args.angles is not None:value=normalize_angles(*args.angles)
    else:
        with open(args.reply_json)as stream:value=parse_reply(json.load(stream))
    print(json.dumps(value,ensure_ascii=False,allow_nan=False))
