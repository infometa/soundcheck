
import os

import numpy as np
import scipy.signal as sig
import soundfile as sf

from utils.config import load_config


fs = int(float(load_config().get("fs", 48000)))

def extract_ir(rec,inv):
    ir=sig.fftconvolve(rec,inv,mode="full")
    peak=np.max(np.abs(ir))
    if peak>0:
        ir/=peak
    os.makedirs("data/processed",exist_ok=True)
    sf.write("data/processed/ir.wav",ir,fs)
    return ir
