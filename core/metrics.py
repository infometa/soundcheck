
import numpy as np

from utils.config import load_config


cfg=load_config()
fs=float(cfg.get("fs", 48000))
eps=float(cfg.get("min_energy", 1e-9))
if eps <= 0:
    eps = 1e-9

def RT60(ir):
    e=ir**2; e[e<eps]=eps
    sch=np.flip(np.cumsum(np.flip(e)))
    peak=np.max(sch)
    if peak<=0:
        return float("nan")
    sch/=peak
    db=10*np.log10(sch)
    t=np.arange(len(db))/fs
    m=(db>-35)&(db<-5)
    if np.sum(m)<10: return float("nan")
    p=np.polyfit(t[m],db[m],1)
    if p[0]==0: return float("nan")
    return -60/p[0]

def C50(ir):
    t0=np.argmax(np.abs(ir))
    i50=min(len(ir), t0+int(0.05*fs))
    num=np.sum(ir[t0:i50]**2)
    den=np.sum(ir[i50:]**2)
    if den<eps: return float("nan")
    return 10*np.log10(num/den)
