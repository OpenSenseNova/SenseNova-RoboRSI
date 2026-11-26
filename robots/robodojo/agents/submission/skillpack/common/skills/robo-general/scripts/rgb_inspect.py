"""Crop/scale an ordinary public RGB observation, with optional source-pixel grid.

This reads camera image files, not serialized trajectory image-bit containers.
It neither localizes objects nor executes robot motion. The JSON result records
the inverse crop/resize transform needed before using pixels with robot tools.
"""
import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw
from robot_reply import crop_bounds


def inspect_image(source, output, *, box=None, scale=1., grid=False, grid_step=20):
    if type(scale)not in(int,float)or not math.isfinite(scale)or scale<=0:
        raise ValueError('scale must be a positive finite number')
    if type(grid)is not bool or type(grid_step)is not int or grid_step<1:
        raise ValueError('grid must be boolean and grid_step a positive integer')
    source,output=Path(source).resolve(strict=True),Path(output).resolve()
    if source==output:raise ValueError('Output must not overwrite the original observation')
    with Image.open(source)as image:
        width,height=image.size
        bounds=crop_bounds(box if box is not None else [0,0,width,height],width,height)
        if any(int(v)!=v for v in bounds):raise ValueError('Crop edges must be integer original-image coordinates')
        x0,y0,x1,y1=map(int,bounds)
        size=(max(1,round((x1-x0)*scale)),max(1,round((y1-y0)*scale)))
        if size[0]*size[1]>32_000_000:raise ValueError('Inspection image exceeds 32 million pixels')
        view=image.crop((x0,y0,x1,y1)).resize(size,Image.Resampling.NEAREST)
        if grid:
            draw=ImageDraw.Draw(view)
            for x in range(x0,x1,grid_step):
                u=(x-x0)*size[0]/(x1-x0);draw.line((u,0,u,size[1]-1),fill='white');draw.text((u+1,1),str(x),fill='white')
            for y in range(y0,y1,grid_step):
                v=(y-y0)*size[1]/(y1-y0);draw.line((0,v,size[0]-1,v),fill='white');draw.text((1,v+1),str(y),fill='white')
        output.parent.mkdir(parents=True,exist_ok=True)
        view.save(output)
    sx,sy=(x1-x0)/size[0],(y1-y0)/size[1]
    return dict(status='image_written',image=str(output),original_image=str(source),
                original_size=[width,height],crop_box=[x0,y0,x1,y1],output_size=list(size),
                # Pixel centres: original = (displayed + .5) * scale - .5 + crop origin.
                pixel_to_original=dict(scale=[sx,sy],offset=[x0+(sx-1)/2,y0+(sy-1)/2]),
                motion_executed=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--crop',nargs=4,type=int,metavar=('X0','Y0','X1','Y1'))
    parser.add_argument('--scale',type=float,default=1.)
    parser.add_argument('--grid',action='store_true');parser.add_argument('--grid-step',type=int,default=20)
    args=parser.parse_args()
    try:
        value=inspect_image(args.input,args.output,box=args.crop,scale=args.scale,grid=args.grid,grid_step=args.grid_step)
    except (OSError,ValueError)as error:
        print(json.dumps(dict(status='input_rejected',detail=str(error),motion_executed=False)))
        raise SystemExit(2)
    print(json.dumps(value,allow_nan=False))
