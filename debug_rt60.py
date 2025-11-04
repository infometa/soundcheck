#!/usr/bin/env python3
"""
调试RT60计算问题
"""

import numpy as np
import matplotlib.pyplot as plt
from core.ir import extract_ir
from core.sync import sync_and_trim
from core.sweep import generate_sweep
from utils.config import load_config

def debug_RT60(ir, plot=True):
    """调试版RT60计算，显示中间步骤"""
    cfg = load_config()
    fs = float(cfg.get("fs", 48000))
    eps = 1e-9

    print("\n" + "="*60)
    print("🔍 RT60 计算调试信息")
    print("="*60)

    # 1. 计算能量
    e = ir**2
    e[e<eps] = eps
    print(f"1. 能量计算:")
    print(f"   IR长度: {len(ir)} 采样点 ({len(ir)/fs:.2f}秒)")
    print(f"   能量最大值: {np.max(e):.2e}")
    print(f"   能量最小值: {np.min(e):.2e}")

    # 2. Schroeder积分（反向累积能量）
    sch = np.flip(np.cumsum(np.flip(e)))
    peak = np.max(sch)
    print(f"\n2. Schroeder积分:")
    print(f"   最大值: {peak:.2e}")

    if peak <= 0:
        print("   ❌ 错误：峰值<=0")
        return float("nan")

    sch = sch / peak

    # 3. 转换为dB
    db = 10 * np.log10(sch)
    t = np.arange(len(db)) / fs

    print(f"\n3. dB转换:")
    print(f"   dB范围: {np.min(db):.1f} 到 {np.max(db):.1f} dB")
    print(f"   时间范围: 0 到 {t[-1]:.2f} 秒")

    # 4. 选择-5dB到-35dB的区间
    m = (db > -35) & (db < -5)
    n_points = np.sum(m)

    print(f"\n4. 拟合区间 (-5dB 到 -35dB):")
    print(f"   数据点数: {n_points}")

    if n_points < 10:
        print("   ❌ 错误：数据点太少")
        return float("nan")

    # 找到区间的时间范围
    t_fit = t[m]
    db_fit = db[m]
    print(f"   时间范围: {t_fit[0]:.3f} 到 {t_fit[-1]:.3f} 秒")
    print(f"   dB范围: {db_fit[0]:.1f} 到 {db_fit[-1]:.1f} dB")

    # 5. 线性拟合
    p = np.polyfit(t_fit, db_fit, 1)
    slope = p[0]  # dB/秒
    intercept = p[1]

    print(f"\n5. 线性拟合:")
    print(f"   斜率: {slope:.2f} dB/秒")
    print(f"   截距: {intercept:.2f} dB")

    if abs(slope) < 1e-10:
        print("   ❌ 错误：斜率太小（接近0）")
        return float("nan")

    # 6. 计算RT60
    # RT60 = 从0dB衰减到-60dB所需的时间
    # 如果斜率是负数（正常情况），-60/slope会给出正数
    result = -60 / slope

    print(f"\n6. RT60计算:")
    print(f"   公式: RT60 = -60 / slope")
    print(f"   RT60 = -60 / {slope:.2f}")
    print(f"   RT60 = {result:.3f} 秒")

    # 检查合理性
    if result < 0:
        print(f"   ❌ 错误：RT60为负数（斜率符号错误）")
        return float("nan")
    elif result > 30:
        print(f"   ⚠️ 警告：RT60过大 (>{30}秒)")
        # 不返回nan，而是输出更多信息

    if plot:
        # 绘制调试图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # 上图：完整的衰减曲线
        ax1.plot(t, db, 'b-', linewidth=0.5, alpha=0.7, label='Schroeder Curve')
        ax1.axhline(-5, color='g', linestyle='--', alpha=0.5, label='-5 dB')
        ax1.axhline(-35, color='r', linestyle='--', alpha=0.5, label='-35 dB')
        ax1.axhline(-60, color='k', linestyle='--', alpha=0.5, label='-60 dB')

        # 标记拟合区间
        ax1.plot(t_fit, db_fit, 'ro', markersize=2, alpha=0.3, label='Fit Region')

        # 绘制拟合直线（延长到-60dB）
        t_extended = np.linspace(0, result * 1.2, 100)
        db_fit_line = slope * t_extended + intercept
        ax1.plot(t_extended, db_fit_line, 'g--', linewidth=2, label=f'Fit Line (slope={slope:.1f} dB/s)')

        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Energy (dB)')
        ax1.set_title(f'RT60 Calculation Debug (RT60={result:.2f}s)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlim(0, min(5, t[-1]))
        ax1.set_ylim(-80, 5)

        # 下图：放大拟合区域
        ax2.plot(t_fit, db_fit, 'ro', markersize=3, label='Data Points')
        t_fit_line = np.linspace(t_fit[0], t_fit[-1], 100)
        db_fit_line2 = slope * t_fit_line + intercept
        ax2.plot(t_fit_line, db_fit_line2, 'g-', linewidth=2, label='Linear Fit')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Energy (dB)')
        ax2.set_title('Fit Region Detail')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.savefig('data/plots/rt60_debug.png', dpi=150)
        print(f"\n📊 调试图已保存: data/plots/rt60_debug.png")
        plt.close()

    return result

if __name__ == "__main__":
    print("生成测试IR...")

    # 使用之前测试中的合成IR
    fs = 48000
    duration = 2.0
    t = np.arange(int(duration * fs)) / fs

    # 创建指数衰减IR (RT60应该约0.5秒)
    # 使用EDT公式: e^(-6.91*t/RT60)
    target_rt60 = 0.5
    decay_rate = -6.91 / target_rt60
    ir = np.exp(decay_rate * t) * np.random.randn(len(t)) * 0.1

    # 添加强直达声
    ir[1000] = 1.0

    # 调试计算
    calculated_rt60 = debug_RT60(ir, plot=True)

    print(f"\n{'='*60}")
    print(f"目标 RT60: {target_rt60:.3f}秒")
    print(f"计算 RT60: {calculated_rt60:.3f}秒")
    print(f"误差: {abs(calculated_rt60 - target_rt60):.3f}秒")
    print(f"{'='*60}")
