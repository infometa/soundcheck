
import os

import numpy as np
import soundfile as sf

from utils.config import load_config


cfg = load_config()
FS = int(float(cfg.get("fs", 48000)))
SWEEP_DURATION = float(cfg.get("sweep_duration", 8.0))
SILENCE_PRE = float(cfg.get("silence_pre", 0.0))
SILENCE_POST = float(cfg.get("silence_post", 0.0))


def generate_sweep():
    t=np.arange(0, SWEEP_DURATION, 1/FS)
    f1,f2=20.0,20000.0
    sweep=np.sin(2*np.pi*f1*(SWEEP_DURATION/np.log(f2/f1))*(np.exp(t*np.log(f2/f1)/SWEEP_DURATION)-1))
    w=2*np.pi*f1*np.exp(t*(np.log(f2/f1)/SWEEP_DURATION))
    inv=sweep[::-1]*w
    inv_max=np.max(np.abs(inv))
    if inv_max>0:
        inv/=inv_max
    sig=np.concatenate([
        np.zeros(int(SILENCE_PRE*FS)),
        sweep,
        np.zeros(int(SILENCE_POST*FS)),
    ])
    os.makedirs("data/raw",exist_ok=True)
    sf.write("data/raw/sweep.wav",sig,FS); np.save("data/raw/inv.npy",inv)
    return sig,inv
