"""Select <=20 local source SIFT queries within48px; <=512 target candidates."""
import cv2,numpy as np
import rgb_geometry as G

def extraction_patch(image,mask):
    yy,xx=np.nonzero(mask)
    if len(xx)==0:raise ValueError('empty_feature_mask')
    lo=np.maximum([0,0],[int(xx.min())-12,int(yy.min())-12]);hi=np.minimum([image.shape[1],image.shape[0]],[int(xx.max())+13,int(yy.max())+13])
    x,y=lo;xx,yy=hi;patch=cv2.resize(image[y:yy,x:xx],None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
    return patch,np.asarray(lo,float)

def select_features(xy,descriptors,scores,sizes,mask,anchor=None,limit=512):
    if len(xy)==0:return {'xy':np.empty((0,2)),'desc':np.empty((0,descriptors.shape[-1] if descriptors.ndim==2 else 0),np.float32),'score':np.empty(0)}
    xy=np.asarray(xy,float);pixels=np.round(xy).astype(int);inside=(pixels[:,0]>=0)&(pixels[:,0]<mask.shape[1])&(pixels[:,1]>=0)&(pixels[:,1]<mask.shape[0])
    pixels=np.clip(pixels,[0,0],[mask.shape[1]-1,mask.shape[0]-1]);distance=cv2.distanceTransform(mask,cv2.DIST_L2,3)
    valid=inside&(distance[pixels[:,1],pixels[:,0]]>=np.maximum(2.,np.asarray(sizes)*.35))
    if anchor is not None:valid &= np.linalg.norm(xy-np.asarray(anchor),axis=1)<=48
    candidates=np.where(valid)[0].tolist();candidates.sort(key=lambda i:float(scores[i]),reverse=True)
    chosen=[]
    if anchor is not None:
        bins=[[] for _ in range(8)]
        for i in candidates:
            d=xy[i]-anchor;sector=int(np.floor((np.arctan2(d[1],d[0])+np.pi)/(2*np.pi)*8))%8;bins[sector].append(i)
        candidates=[]
        for j in range(max(map(len,bins),default=0)):
            for bucket in bins:
                if j<len(bucket):candidates.append(bucket[j])
    for i in candidates:
        if all(np.linalg.norm(xy[i]-xy[j])>=2 for j in chosen):chosen.append(i)
        if len(chosen)>=limit:break
    return {'xy':xy[chosen], 'desc':np.asarray(descriptors[chosen],np.float32),'score':np.asarray(scores)[chosen]}

def sift(image,mask,anchor=None):
    patch,lo=extraction_patch(image,mask);gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
    keys,descriptors=cv2.SIFT_create(nfeatures=1000,contrastThreshold=.006,edgeThreshold=15,nOctaveLayers=4).detectAndCompute(gray,None)
    if descriptors is None:return {'xy':np.empty((0,2)),'desc':np.empty((0,128),np.float32),'score':np.empty(0)}
    xy=(np.asarray([k.pt for k in keys])+.5)/3-.5+lo;scores=np.asarray([k.response for k in keys]);sizes=np.asarray([k.size/3 for k in keys])
    return select_features(xy,descriptors,scores,sizes,mask,anchor,20 if anchor is not None else 512)

def sift_match(a,b):
    if min(len(a['desc']),len(b['desc']))<2:return np.empty((0,2),int)
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    def good(x,y):return {m.queryIdx:m for pair in matcher.knnMatch(x,y,k=2) if len(pair)==2 for m,n in [pair] if m.distance<.8*n.distance}
    forward=good(a['desc'],b['desc']);reverse=good(b['desc'],a['desc']);pairs=[]
    for m in forward.values():
        if m.trainIdx in reverse and reverse[m.trainIdx].trainIdx==m.queryIdx:pairs.append([m.queryIdx,m.trainIdx])
    return np.asarray(pairs,int).reshape(-1,2)

def calibrated_matches(a,b,indices,c0,c1):
    raw_count=len(indices);kept=[]
    for i,j in indices:
        u,v=a['xy'][i],b['xy'][j]
        if G.epipolar_error(u,v,c0,c1)<=2:kept.append((int(i),int(j)))
    ii=np.asarray(kept,int).reshape(-1,2)
    return a['xy'][ii[:,0]],b['xy'][ii[:,1]],{'source_queries':len(a['xy']),'target_candidates':len(b['xy']),'descriptor_matches':raw_count,'epipolar_matches':len(ii),'indices':ii.tolist()}
