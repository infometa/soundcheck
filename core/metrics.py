
import numpy as np

from utils.config import load_config


cfg=load_config()
fs=float(cfg.get("fs", 48000))
eps=float(cfg.get("min_energy", 1e-9))
if eps <= 0:
    eps = 1e-9

def RT60(ir, debug=False):
    """Calculate RT60 (Reverberation Time) - time for sound to decay by 60dB.

    Args:
        ir: Impulse response array
        debug: If True, print detailed calculation steps

    Returns:
        RT60 in seconds, or nan if calculation fails
    """
    try:
        # 1. 计算能量
        e=ir**2
        e[e<eps]=eps

        # 2. Schroeder积分（反向累积能量）
        sch=np.flip(np.cumsum(np.flip(e)))
        peak=np.max(sch)

        if peak<=0:
            if debug:
                print("   ⚠️ RT60: Schroeder积分峰值<=0")
            return float("nan")

        sch=sch/peak

        # 3. 转换为dB
        db=10*np.log10(sch)
        t=np.arange(len(db))/fs

        # 4. 选择-5dB到-35dB的拟合区间
        m=(db>-35)&(db<-5)
        n_points = np.sum(m)

        if n_points<10:
            if debug:
                print(f"   ⚠️ RT60: 拟合区间数据点太少 ({n_points})")
            return float("nan")

        # 找到拟合区间的时间和dB值
        t_fit = t[m]
        db_fit = db[m]

        # 5. 线性拟合
        p=np.polyfit(t_fit,db_fit,1)
        slope = p[0]  # dB/秒

        if debug:
            print(f"\n   🔍 RT60调试信息:")
            print(f"      IR长度: {len(ir)/fs:.2f}秒")
            print(f"      拟合区间: {t_fit[0]:.3f}-{t_fit[-1]:.3f}秒 ({n_points}点)")
            print(f"      dB范围: {db_fit[0]:.1f} 到 {db_fit[-1]:.1f} dB")
            print(f"      斜率: {slope:.2f} dB/秒")

        if abs(slope)<1e-10:
            if debug:
                print(f"      ⚠️ 斜率太小")
            return float("nan")

        # 6. 计算RT60
        result = -60/slope

        if debug:
            print(f"      RT60 = -60 / {slope:.2f} = {result:.3f}秒")

        # 合理性检查
        if result < 0:
            if debug:
                print(f"      ⚠️ RT60为负数")
            return float("nan")

        # 对于过长的RT60，打印警告但仍返回值（不返回nan）
        if result > 5:
            print(f"\n   ⚠️ RT60值较大 ({result:.2f}秒)，可能的原因:")
            print(f"      1. 录音音量过低 → 背景噪声抬高了能量地板")
            print(f"      2. IR提取质量不佳 → 检查同步和对齐")
            print(f"      3. 录音时间太短 → 混响未完全衰减")
            print(f"      拟合区间: {t_fit[0]:.3f}-{t_fit[-1]:.3f}秒")
            print(f"      衰减速度: {slope:.2f} dB/秒")

            # 分析能量分布
            direct_idx = np.argmax(np.abs(ir))
            if direct_idx + int(fs) < len(ir):
                energy_after_1s = np.sum(ir[direct_idx + int(fs):]**2)
                total_energy = np.sum(ir[direct_idx:]**2)
                print(f"      直达声后1秒能量占比: {energy_after_1s/total_energy*100:.1f}%")

        return result

    except Exception as e:
        print(f"   ❌ RT60计算错误: {e}")
        import traceback
        traceback.print_exc()
        return float("nan")

def C50(ir):
    """Calculate C50 (Clarity) - ratio of early (0-50ms) to late energy after direct sound."""
    t0=np.argmax(np.abs(ir))
    i50_samples=int(0.05*fs)

    # Ensure we have enough samples
    if t0 + i50_samples >= len(ir):
        return float("nan")

    i50_idx = t0 + i50_samples

    # Early energy: from direct sound to 50ms after
    num=np.sum(ir[t0:i50_idx]**2)
    # Late energy: after 50ms
    den=np.sum(ir[i50_idx:]**2)

    if den<eps or num<eps:
        return float("nan")
    return 10*np.log10(num/den)
