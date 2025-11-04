
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.config import load_config

cfg = load_config()
EARLY_REFL_TIME = float(cfg.get("early_reflection_time", 0.08))  # 80ms default

def plot_ir(ir, fs, ref=None, path="data/plots/ir.png"):
    """Plot impulse response with IR waveform and ETC (Energy Time Curve)."""
    t = np.arange(len(ir)) / fs
    t_ms = t * 1000  # Convert to milliseconds for better readability

    # Find direct sound peak
    direct_idx = np.argmax(np.abs(ir))
    direct_time = t[direct_idx]

    # Calculate early reflection boundary (80ms after direct sound by default)
    early_end_time = direct_time + EARLY_REFL_TIME
    early_end_idx = int(early_end_time * fs)

    # Calculate ETC (Energy Time Curve) - squared IR in dB
    energy = ir ** 2
    energy_smooth = np.convolve(energy, np.ones(int(fs*0.001))/int(fs*0.001), mode='same')
    eps = 1e-12
    etc_db = 10 * np.log10(energy_smooth + eps)
    etc_db = etc_db - np.max(etc_db)  # Normalize to 0 dB

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # === Plot 1: IR Waveform ===
    ax1.plot(t_ms, ir, color='black', linewidth=0.5, alpha=0.7, label='IR Waveform')

    # Mark direct sound (RED)
    ax1.axvline(direct_time*1000, color='red', linewidth=2, label='直达声 (Direct)', zorder=10)

    # Shade early reflections region (BLUE)
    if early_end_idx < len(ir):
        ax1.axvspan(direct_time*1000, early_end_time*1000, alpha=0.15, color='blue', label='早反射 (Early)')

    # Shade reverb tail region (GRAY)
    if early_end_idx < len(ir):
        ax1.axvspan(early_end_time*1000, t_ms[-1], alpha=0.15, color='gray', label='混响尾声 (Late)')

    # Mark detected reflections
    if ref is not None and len(ref) > 0:
        for i, tt in enumerate(ref):
            if i == 0:
                ax1.axvline(tt*1000, color='orange', linestyle=':', alpha=0.6, linewidth=1, label='检测到的反射')
            else:
                ax1.axvline(tt*1000, color='orange', linestyle=':', alpha=0.6, linewidth=1)

    ax1.set_xlabel('时间 (ms)', fontsize=12)
    ax1.set_ylabel('振幅', fontsize=12)
    ax1.set_title('脉冲响应 (Impulse Response)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_xlim(0, min(500, t_ms[-1]))  # Show first 500ms

    # === Plot 2: ETC (Energy Time Curve) ===
    # Direct sound (RED)
    mask_direct = (t_ms <= direct_time*1000 + 5)  # 5ms window
    ax2.plot(t_ms[mask_direct], etc_db[mask_direct], color='red', linewidth=2, label='直达声')

    # Early reflections (BLUE)
    mask_early = (t_ms > direct_time*1000 + 5) & (t_ms <= early_end_time*1000)
    if np.any(mask_early):
        ax2.plot(t_ms[mask_early], etc_db[mask_early], color='blue', linewidth=2, label='早反射')

    # Reverb tail (GRAY)
    mask_late = (t_ms > early_end_time*1000)
    if np.any(mask_late):
        ax2.plot(t_ms[mask_late], etc_db[mask_late], color='gray', linewidth=2, label='混响尾声')

    # Mark time boundaries
    ax2.axvline(direct_time*1000, color='red', linewidth=1, linestyle='--', alpha=0.5)
    ax2.axvline(early_end_time*1000, color='blue', linewidth=1, linestyle='--', alpha=0.5)

    ax2.set_xlabel('时间 (ms)', fontsize=12)
    ax2.set_ylabel('能量 (dB)', fontsize=12)
    ax2.set_title('能量时间曲线 (ETC - Energy Time Curve)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_xlim(0, min(500, t_ms[-1]))  # Show first 500ms
    ax2.set_ylim(-80, 5)

    plt.tight_layout()
    os.makedirs("data/plots", exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 图表已保存: {path}")
