import os

import numpy as np
import sounddevice as sd
import soundfile as sf

from utils.config import load_config


cfg = load_config()
FS = int(float(cfg.get("fs", 48000)))
RECORD_TAIL = float(cfg.get("record_tail", 0.0))


def play_and_record(sig):
    """Play sweep signal and simultaneously record response."""
    try:
        tail_samples=int(RECORD_TAIL*FS)
        sd.wait()
        playback=np.concatenate([sig, np.zeros(tail_samples, dtype=sig.dtype)])

        print(f"🎵 播放并录制中... ({len(playback)/FS:.1f}秒)")
        rec=sd.playrec(playback,FS,channels=1)
        sd.wait()

        if rec is None or len(rec) == 0:
            raise ValueError("录制失败：没有录制到音频数据")

        rec=rec.squeeze()

        # Check if recording is too quiet (potential hardware issue)
        max_level = np.max(np.abs(rec))
        if max_level < 1e-6:
            print("⚠️ 警告：录制音量过低，可能存在硬件问题")

        os.makedirs("data/raw",exist_ok=True)
        sf.write("data/raw/rec.wav",rec,FS)
        print(f"✅ 录制完成，峰值: {20*np.log10(max_level):.1f} dB")
        return rec
    except Exception as e:
        print(f"❌ 录制错误: {e}")
        raise
