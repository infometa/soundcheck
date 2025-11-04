#!/usr/bin/env python3
"""
分析IR问题
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# 读取IR
ir, fs = sf.read('data/processed/ir.wav')

print("="*70)
print("详细IR分析")
print("="*70)

# 基本信息
print(f"\n[1] 基本信息:")
print(f"   长度: {len(ir)} 采样点 ({len(ir)/fs:.2f}秒)")
print(f"   采样率: {fs} Hz")
print(f"   峰值: {np.max(np.abs(ir)):.6f}")
print(f"   位置: {np.argmax(np.abs(ir))} 采样点 ({np.argmax(np.abs(ir))/fs:.3f}秒)")

# 找到最大值位置
peak_idx = np.argmax(np.abs(ir))
peak_time = peak_idx / fs

print(f"\n[2] 峰值分析:")
print(f"   峰值在第 {peak_time:.3f} 秒")
print(f"   峰值在整个IR的 {peak_idx/len(ir)*100:.1f}% 位置")

# 前100ms的情况
first_100ms = int(0.1 * fs)
early_peak = np.max(np.abs(ir[:first_100ms]))
early_peak_idx = np.argmax(np.abs(ir[:first_100ms]))

print(f"\n[3] 前100ms分析:")
print(f"   前100ms峰值: {early_peak:.6f}")
print(f"   前100ms峰值位置: {early_peak_idx/fs*1000:.1f} ms")
print(f"   前100ms峰值 / 全局峰值: {early_peak/np.max(np.abs(ir))*100:.1f}%")

if early_peak < np.max(np.abs(ir)) * 0.5:
    print(f"   ⚠️ 前100ms峰值明显小于全局峰值，IR可能有问题！")

# 能量分布
total_energy = np.sum(ir**2)
energy_before_peak = np.sum(ir[:peak_idx]**2)
energy_after_peak = np.sum(ir[peak_idx:]**2)

print(f"\n[4] 能量分布:")
print(f"   峰值前能量: {energy_before_peak/total_energy*100:.1f}%")
print(f"   峰值后能量: {energy_after_peak/total_energy*100:.1f}%")

if energy_before_peak > total_energy * 0.3:
    print(f"   ⚠️ 峰值前有大量能量 ({energy_before_peak/total_energy*100:.1f}%)")
    print(f"      这说明IR同步可能有问题，或者有大量前置噪声")

# 检查IR的前几个峰值
print(f"\n[5] 前10个最大峰值位置:")
abs_ir = np.abs(ir)
top10_indices = np.argsort(abs_ir)[-10:][::-1]
for i, idx in enumerate(top10_indices):
    time_s = idx / fs
    value = ir[idx]
    print(f"   {i+1}. 位置: {time_s:.3f}秒, 幅值: {value:.6f}")

# 绘制详细图
fig, axes = plt.subplots(4, 1, figsize=(16, 12))

t = np.arange(len(ir)) / fs
t_ms = t * 1000

# 图1：完整IR
axes[0].plot(t, ir, 'b-', linewidth=0.5, alpha=0.7)
axes[0].axvline(peak_time, color='r', linewidth=2, label=f'Peak at {peak_time:.3f}s')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Complete IR (Full Length)')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# 图2：前1秒
first_1s_samples = int(1 * fs)
axes[1].plot(t[:first_1s_samples], ir[:first_1s_samples], 'b-', linewidth=0.8)
if early_peak_idx < first_1s_samples:
    axes[1].axvline(early_peak_idx/fs, color='g', linewidth=2, label=f'Early peak at {early_peak_idx/fs:.3f}s')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Amplitude')
axes[1].set_title('First 1 Second (Should contain direct sound)')
axes[1].grid(True, alpha=0.3)
axes[1].legend()
axes[1].set_xlim(0, 1)

# 图3：峰值附近
window = 0.5  # 峰值前后0.5秒
start_idx = max(0, peak_idx - int(window * fs))
end_idx = min(len(ir), peak_idx + int(window * fs))
t_window = (np.arange(end_idx - start_idx) + start_idx) / fs

axes[2].plot(t_window, ir[start_idx:end_idx], 'b-', linewidth=0.8)
axes[2].axvline(peak_time, color='r', linewidth=2, label='Peak')
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Amplitude')
axes[2].set_title(f'Around Peak ({peak_time-window:.2f}s to {peak_time+window:.2f}s)')
axes[2].grid(True, alpha=0.3)
axes[2].legend()

# 图4：能量包络(对数)
window_size = int(fs * 0.01)
envelope = np.convolve(np.abs(ir), np.ones(window_size)/window_size, mode='same')
envelope_db = 20 * np.log10(envelope + 1e-9)

axes[3].plot(t, envelope_db, 'g-', linewidth=1)
axes[3].axvline(peak_time, color='r', linewidth=2, label='Peak')
axes[3].axhline(-60, color='orange', linestyle='--', alpha=0.5, label='-60dB')
axes[3].set_xlabel('Time (s)')
axes[3].set_ylabel('Energy (dB)')
axes[3].set_title('Energy Envelope')
axes[3].grid(True, alpha=0.3)
axes[3].legend()
axes[3].set_ylim(-100, 10)

plt.tight_layout()
plt.savefig('data/plots/ir_problem_analysis.png', dpi=150)
print(f"\n[6] 图表已保存: data/plots/ir_problem_analysis.png")

print("\n" + "="*70)
print("诊断结果:")
print("="*70)

if peak_time > 1.0:
    print("❌ 严重问题：直达声位置在 {:.2f} 秒！".format(peak_time))
    print("   正常应该在前100ms内")
    print("\n可能原因：")
    print("   1. IR同步失败 - sync_and_trim函数没有正确对齐")
    print("   2. 扫频和录音之间有延迟")
    print("   3. 原始录音质量差")
    print("\n建议：")
    print("   1. 重新运行测量，确保录音音量足够")
    print("   2. 检查 data/raw/rec.wav - 应该能听到明显的扫频声")
    print("   3. 检查同步函数的输出")
elif early_peak < np.max(np.abs(ir)) * 0.5:
    print("⚠️ 问题：前100ms没有明显的直达声")
    print("   前100ms峰值只有全局峰值的 {:.1f}%".format(early_peak/np.max(np.abs(ir))*100))
    print("\n可能原因：同步偏移或者录音起点不对")
else:
    print("✅ IR看起来正常")
    print("   直达声在前100ms内")

plt.close()
