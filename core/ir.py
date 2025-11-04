
import os

import numpy as np
import scipy.signal as sig
import soundfile as sf

from utils.config import load_config


fs = int(float(load_config().get("fs", 48000)))

def extract_ir(rec,inv):
    """Extract impulse response using deconvolution with inverse filter."""
    print("🔄 提取脉冲响应中...")
    ir=sig.fftconvolve(rec,inv,mode="full")
    peak=np.max(np.abs(ir))
    if peak>0:
        ir/=peak
    else:
        print("⚠️ 警告：IR峰值为0")

    os.makedirs("data/processed",exist_ok=True)
    sf.write("data/processed/ir.wav",ir,fs)
    print(f"✅ IR提取完成，长度: {len(ir)/fs:.2f}秒")
    return ir
