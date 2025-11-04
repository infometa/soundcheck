#!/usr/bin/env python3
"""
修正IR - 将直达声移到开头
"""

import numpy as np
import soundfile as sf
from core.metrics import RT60, C50
from core.separate import separate_ir_components, export_ir_comparison
from utils.plot import plot_ir
from core.reflections import reflections

print("="*70)
print("IR修正工具")
print("="*70)

# 读取IR
ir, fs = sf.read('data/processed/ir.wav')

print(f"\n[1] 原始IR信息:")
print(f"   长度: {len(ir)/fs:.2f}秒")

# 找到直达声（峰值）
peak_idx = np.argmax(np.abs(ir))
peak_time = peak_idx / fs

print(f"   直达声位置: {peak_time:.3f}秒")

if peak_time > 0.1:
    print(f"   ⚠️ 直达声位置异常（应该在0.1秒内）")

    # 裁剪IR：从直达声前50ms开始
    pre_padding = int(0.05 * fs)  # 50ms前置
    start_idx = max(0, peak_idx - pre_padding)

    ir_fixed = ir[start_idx:]

    print(f"\n[2] 修正IR:")
    print(f"   裁剪起点: {start_idx/fs:.3f}秒")
    print(f"   新长度: {len(ir_fixed)/fs:.2f}秒")
    print(f"   新的直达声位置: {(peak_idx-start_idx)/fs:.3f}秒")

    # 保存修正后的IR
    sf.write('data/processed/ir_fixed.wav', ir_fixed, fs)
    print(f"   ✅ 已保存: data/processed/ir_fixed.wav")

    # 使用修正后的IR计算指标
    ir = ir_fixed
    print(f"\n[3] 使用修正后的IR计算...")

else:
    print(f"   ✅ 直达声位置正常")
    ir_fixed = ir

# 计算声学指标
print(f"\n[4] 计算声学指标:")
rt60 = RT60(ir)
c50 = C50(ir)

print(f"   RT60: {rt60:.3f}秒")
print(f"   C50: {c50:.2f} dB")

# 分离IR
print(f"\n[5] 分离IR成分:")
paths = separate_ir_components(ir, output_dir='data/separated_fixed')
export_ir_comparison(ir, output_path='data/separated_fixed/comparison.wav')

# 绘制图表
print(f"\n[6] 绘制图表:")
ref = reflections(ir)
plot_ir(ir, fs, ref, path='data/plots/ir_fixed.png')
print(f"   ✅ 已保存: data/plots/ir_fixed.png")

print("\n" + "="*70)
print("修正完成！")
print("="*70)
print(f"\n修正后的文件:")
print(f"  IR: data/processed/ir_fixed.wav")
print(f"  直达声: {paths['direct']}")
print(f"  早反射: {paths['early']}")
print(f"  混响尾声: {paths['late']}")
print(f"  对比文件: data/separated_fixed/comparison.wav")
print(f"  图表: data/plots/ir_fixed.png")
print(f"\n声学指标:")
print(f"  RT60: {rt60:.3f}秒")
print(f"  C50: {c50:.2f} dB")
print("="*70)
