import os

import numpy as np
import sounddevice as sd
import soundfile as sf

from utils.config import load_config


cfg = load_config()
FS = int(float(cfg.get("fs", 48000)))
RECORD_TAIL = float(cfg.get("record_tail", 0.0))


def play_and_record(sig):
    tail_samples=int(RECORD_TAIL*FS)
    sd.wait()
    playback=np.concatenate([sig, np.zeros(tail_samples, dtype=sig.dtype)])
    rec=sd.playrec(playback,FS,channels=1)
    sd.wait()
    rec=rec.squeeze()
    os.makedirs("data/raw",exist_ok=True)
    sf.write("data/raw/rec.wav",rec,FS)
    return rec
