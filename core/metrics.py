
import numpy as np

from utils.config import load_config


cfg=load_config()
fs=float(cfg.get("fs", 48000))
eps=float(cfg.get("min_energy", 1e-9))
if eps <= 0:
    eps = 1e-9

def RT60(ir):
    """Calculate RT60 (Reverberation Time) - time for sound to decay by 60dB."""
    try:
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
        if p[0]==0 or abs(p[0])<1e-10: return float("nan")
        result = -60/p[0]
        # Sanity check: RT60 should be positive and reasonable (< 30 seconds for typical rooms)
        if result < 0 or result > 30:
            return float("nan")
        return result
    except Exception as e:
        print(f"⚠️ RT60 calculation error: {e}")
        return float("nan")

def C50(ir):
    """Calculate C50 (Clarity) - ratio of early (0-50ms) to late energy after direct sound."""
    t0=np.argmax(np.abs(ir))
    i50_samples=int(0.05*fs)

    # Ensure we have enough samples
    if t0 + i50_samples >= len(ir):
        return float("nan")

    i50_idx = t0 + i50_samples

    # Early energy: from direct sound to 50ms after
    num=np.sum(ir[t0:i50_idx]**2)
    # Late energy: after 50ms
    den=np.sum(ir[i50_idx:]**2)

    if den<eps or num<eps:
        return float("nan")
    return 10*np.log10(num/den)
