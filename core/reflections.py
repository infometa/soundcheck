
import numpy as np
from scipy.signal import find_peaks

from utils.config import load_config


p=load_config()
fs=float(p.get("fs", 48000))
db=float(p.get("min_peak_db", -25))
dist=float(p.get("min_peak_distance_ms", 1.0))

def reflections(ir):
    """Detect reflection peaks in impulse response."""
    th=np.max(np.abs(ir))*10**(db/20)
    min_dist=max(1,int(fs*dist/1000))
    peaks,_=find_peaks(np.abs(ir),height=th,distance=min_dist)
    print(f"🔍 检测到 {len(peaks)} 个反射峰值 (阈值: {db} dB)")
    return peaks/fs
