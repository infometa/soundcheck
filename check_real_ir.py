#!/usr/bin/env python3
"""
检查实际测量的IR文件
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from debug_rt60 import debug_RT60

# 读取实际测量的IR
ir_path = "data/processed/ir.wav"

try:
    ir, fs = sf.read(ir_path)
    print(f"读取IR: {ir_path}")
    print(f"采样率: {fs} Hz")
    print(f"长度: {len(ir)} 采样点 ({len(ir)/fs:.2f}秒)")
    print(f"峰值: {np.max(np.abs(ir)):.3f}")
    print(f"RMS: {np.sqrt(np.mean(ir**2)):.3e}")

    # 调试RT60计算
    rt60 = debug_RT60(ir, plot=True)

    print(f"\n最终RT60: {rt60:.3f}秒")

    # 额外检查：绘制IR波形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    t = np.arange(len(ir)) / fs

    # 上图：完整IR波形
    ax1.plot(t, ir, 'b-', linewidth=0.5, alpha=0.7)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Complete IR Waveform')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='k', linewidth=0.5)

    # 下图：能量包络（dB）
    window_size = int(fs * 0.01)  # 10ms窗口
    ir_abs = np.abs(ir)
    envelope = np.convolve(ir_abs, np.ones(window_size)/window_size, mode='same')
    envelope_db = 20 * np.log10(envelope + 1e-9)

    ax2.plot(t, envelope_db, 'r-', linewidth=1, label='Energy Envelope')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Energy (dB)')
    ax2.set_title('IR Energy Envelope')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(-100, 10)

    plt.tight_layout()
    plt.savefig('data/plots/ir_analysis.png', dpi=150)
    print(f"\n📊 IR分析图已保存: data/plots/ir_analysis.png")
    plt.close()

except FileNotFoundError:
    print(f"❌ 错误：找不到文件 {ir_path}")
    print("请先运行 python3 run.py 进行测量")
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
