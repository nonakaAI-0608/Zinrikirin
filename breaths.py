"""Heuristic breath candidates for isolated vocals; not a breath classifier."""
from dataclasses import dataclass
import math
import numpy as np
from scipy.signal import resample_poly
from scipy.ndimage import maximum_filter1d, binary_closing
from core import Clip, runs

@dataclass
class BreathSettings:
    sensitivity: float = 50.
    minimum: float = .10
    spacing: float = .50

@dataclass
class BreathCandidate:
    start: float
    end: float
    split: float
    score: float


BREATH_LEAD = .03  # seconds kept before the detected breath onset


def detect_breaths(audio, rate, settings, position='start'):
    """position='start': split just before the breath so the breath stays at the head
    of the following clip. position='middle': legacy behaviour (split in the middle)."""
    if not (math.isfinite(settings.sensitivity) and 0<=settings.sensitivity<=100 and
            math.isfinite(settings.minimum) and .04<=settings.minimum<=1 and
            math.isfinite(settings.spacing) and .1<=settings.spacing<=10):
        raise ValueError('感度は0～100、最短ブレスは0.04～1秒、最短区間は0.1～10秒にしてください。')
    if not len(audio):return []
    a=np.asarray(audio)
    if a.ndim==2:
        channel=int(np.argmax(np.mean(a.astype(np.float64)**2,axis=0)))
        a=a[:,channel]
    divisor=math.gcd(int(rate),16000)
    a=resample_poly(a,16000//divisor,int(rate)//divisor).astype(np.float32)
    hop=160;frame=512
    if len(a)<frame:return []
    frames=np.lib.stride_tricks.sliding_window_view(a,frame)[::hop]
    db=[];flat=[];harm=[]
    # Bounded FFT batches keep memory modest for full songs.
    window=np.hanning(frame)
    for start in range(0,len(frames),2048):
        block=frames[start:start+2048].astype(np.float64)
        energy=np.mean(block**2,axis=1)
        db.extend(10*np.log10(np.maximum(energy,1e-12)))
        spectrum=np.abs(np.fft.rfft(block*window,axis=1))**2
        band=spectrum[:,3:240]+1e-12
        flat.extend(np.exp(np.mean(np.log(band),axis=1))/np.mean(band,axis=1))
        correlation=np.fft.irfft(np.abs(np.fft.rfft(block*window,n=1024,axis=1))**2,n=1024,axis=1)
        harm.extend(np.max(correlation[:,16:229],axis=1)/np.maximum(correlation[:,0],1e-12))
    db=np.asarray(db);flat=np.asarray(flat);harm=np.asarray(harm)
    if db.max() < -65:return []
    local=maximum_filter1d(db,size=121,mode='nearest')
    sensitivity=settings.sensitivity/100
    drop=local-db
    # Noise-like sound AND an energy dip; duration and voiced flanks reject many consonants.
    noisy=(flat>(.16-.12*sensitivity)) & (harm<(.36+.25*sensitivity))
    mask=noisy & (drop>(10-7*sensitivity)) & (db>max(-75,float(np.max(db))-50))
    mask=binary_closing(mask,structure=np.ones(3,dtype=bool))
    voiced=(harm>.50) & (db>local-14) & (db>-60)
    candidates=[]
    for left,right in runs(mask):
        length=(right-left)*.01
        if not settings.minimum<=length<=.9:continue
        # Require singing both before and after a candidate, not a trailing noise bed.
        before=voiced[max(0,left-65):left]
        after=voiced[right:min(len(voiced),right+65)]
        if np.count_nonzero(before)<5 or np.count_nonzero(after)<5:continue
        first=(left*hop+frame/2)/16000
        last=(right*hop+frame/2)/16000
        middle=(first+last)/2
        point=max(0.,first-BREATH_LEAD) if position=='start' else middle
        if point<settings.spacing or len(audio)/rate-point<settings.spacing:continue
        score=float(np.mean(flat[left:right]*(1-harm[left:right])*np.minimum(drop[left:right]/12,1)))
        candidates.append(BreathCandidate(first,last,round(point*rate)/rate,score))
    # Retain strongest candidates first if they are too close to one another.
    kept=[]
    for c in sorted(candidates,key=lambda c:c.score,reverse=True):
        if all(abs(c.split-other.split)>=settings.spacing for other in kept):kept.append(c)
    return sorted(kept,key=lambda c:c.split)


def split_at_candidates(clips, candidates, minimum=.5):
    """Split without gaps/overlaps; breath audio stays in the adjacent clips."""
    result=[]
    for clip in clips:
        points=sorted({c.split for c in candidates if clip.start+minimum<=c.split<=clip.end-minimum})
        start=clip.start
        for point in points:
            if point-start < minimum:continue
            result.append(Clip(start,point,excluded=clip.excluded));start=point
        result.append(Clip(start,clip.end,excluded=clip.excluded))
    return result
