#!/usr/bin/env python3
"""
声学测量诊断工具
当RT60值异常时运行此脚本诊断问题
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from core.metrics import RT60, C50
from utils.config import load_config

def diagnose_measurement():
    """诊断测量质量"""

    print("="*70)
    print("🔍 声学测量诊断工具")
    print("="*70)

    cfg = load_config()
    fs = int(float(cfg.get("fs", 48000)))

    # 检查所有相关文件
    files = {
        '扫频信号': 'data/raw/sweep.wav',
        '原始录音': 'data/raw/rec.wav',
        '脉冲响应': 'data/processed/ir.wav',
    }

    print("\n[1] 检查文件...")
    for name, path in files.items():
        try:
            data, sr = sf.read(path)
            duration = len(data) / sr
            peak = np.max(np.abs(data))
            rms = np.sqrt(np.mean(data**2))
            print(f"   ✅ {name}: {duration:.2f}秒, 峰值={peak:.3f}, RMS={rms:.2e}")
        except FileNotFoundError:
            print(f"   ❌ {name}: 文件不存在")
        except Exception as e:
            print(f"   ❌ {name}: 读取错误 - {e}")

    # 详细分析IR
    print("\n[2] 分析脉冲响应...")
    try:
        ir, sr = sf.read('data/processed/ir.wav')

        # 基本信息
        direct_idx = np.argmax(np.abs(ir))
        direct_time = direct_idx / sr
        peak_value = ir[direct_idx]

        print(f"   直达声位置: {direct_time:.3f}秒 (样本{direct_idx})")
        print(f"   直达声幅值: {peak_value:.3f}")

        # 能量分析
        total_energy = np.sum(ir**2)
        energy_before_direct = np.sum(ir[:direct_idx]**2)
        energy_after_direct = np.sum(ir[direct_idx:]**2)

        print(f"\n   能量分析:")
        print(f"   - 直达声前: {energy_before_direct/total_energy*100:.2f}%")
        print(f"   - 直达声后: {energy_after_direct/total_energy*100:.2f}%")

        # 时间衰减分析
        if direct_idx + int(sr) < len(ir):
            energy_1s = np.sum(ir[direct_idx:direct_idx+int(sr)]**2)
            energy_2s = np.sum(ir[direct_idx:direct_idx+int(2*sr)]**2) if direct_idx + int(2*sr) < len(ir) else 0
            print(f"   - 前1秒能量: {energy_1s/total_energy*100:.2f}%")
            if energy_2s > 0:
                print(f"   - 前2秒能量: {energy_2s/total_energy*100:.2f}%")

        # 噪声地板估计
        # 使用直达声前的能量估计噪声
        noise_floor = np.sqrt(np.mean(ir[:max(1000, direct_idx//2)]**2))
        signal_to_noise = 20 * np.log10(abs(peak_value) / noise_floor) if noise_floor > 0 else float('inf')

        print(f"\n   噪声估计:")
        print(f"   - 噪声地板RMS: {noise_floor:.2e}")
        print(f"   - 信噪比: {signal_to_noise:.1f} dB")

        if signal_to_noise < 40:
            print(f"   ⚠️ 信噪比较低 (<40dB)，可能影响RT60精度")

        # 检查能量衰减
        print(f"\n   能量衰减检查:")
        # 计算每秒的能量衰减
        section_duration = 0.5  # 0.5秒一段
        sections = int((len(ir) - direct_idx) / (sr * section_duration))

        if sections >= 2:
            section_energies = []
            for i in range(min(sections, 10)):  # 最多检查10段
                start = direct_idx + int(i * section_duration * sr)
                end = direct_idx + int((i+1) * section_duration * sr)
                if end <= len(ir):
                    section_energy = np.sum(ir[start:end]**2)
                    section_energies.append(section_energy)

            # 检查能量是否递减
            is_decreasing = all(section_energies[i] >= section_energies[i+1] for i in range(len(section_energies)-1))

            for i, energy in enumerate(section_energies[:5]):
                time_start = i * section_duration
                print(f"   - {time_start:.1f}-{time_start+section_duration:.1f}秒: {energy:.2e}")

            if not is_decreasing:
                print(f"   ⚠️ 能量未持续衰减，可能有噪声干扰")

    except Exception as e:
        print(f"   ❌ IR分析错误: {e}")

    # 计算声学指标（带调试）
    print("\n[3] 计算声学指标...")
    try:
        rt60 = RT60(ir, debug=True)
        c50 = C50(ir)

        print(f"\n   最终结果:")
        print(f"   - RT60: {rt60:.3f}秒")
        print(f"   - C50: {c50:.2f} dB")

        # 评估RT60合理性
        print(f"\n   RT60评估:")
        if rt60 < 0.2:
            print(f"   ⚠️ RT60过小 - 可能是计算错误或消音室")
        elif rt60 < 0.5:
            print(f"   ✅ RT60正常 - 适合小房间或经过声学处理的房间")
        elif rt60 < 1.5:
            print(f"   ✅ RT60正常 - 适合中型房间/会议室")
        elif rt60 < 3.0:
            print(f"   ⚠️ RT60偏大 - 房间混响较重，建议增加吸音")
        else:
            print(f"   ❌ RT60过大 - 可能测量有问题，请检查:")
            print(f"      1. 录音增益是否太低？")
            print(f"      2. 是否有持续的背景噪声？")
            print(f"      3. 扬声器和麦克风摆放是否合理？")

    except Exception as e:
        print(f"   ❌ 指标计算错误: {e}")

    # 生成诊断图
    print("\n[4] 生成诊断图...")
    try:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        t = np.arange(len(ir)) / sr

        # 图1：完整波形
        axes[0].plot(t, ir, 'b-', linewidth=0.5, alpha=0.7)
        axes[0].axvline(direct_time, color='r', linestyle='--', label='Direct Sound')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('Complete IR Waveform')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # 图2：能量包络(dB)
        window_size = int(sr * 0.01)  # 10ms
        envelope = np.convolve(np.abs(ir), np.ones(window_size)/window_size, mode='same')
        envelope_db = 20 * np.log10(envelope + 1e-9)

        axes[1].plot(t, envelope_db, 'g-', linewidth=1)
        axes[1].axhline(-60, color='r', linestyle='--', alpha=0.5, label='-60dB')
        axes[1].axvline(direct_time, color='r', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Energy (dB)')
        axes[1].set_title('Energy Envelope')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        axes[1].set_ylim(-100, 10)

        # 图3：Schroeder积分
        e = ir**2
        e[e<1e-9] = 1e-9
        sch = np.flip(np.cumsum(np.flip(e)))
        sch = sch / np.max(sch)
        sch_db = 10 * np.log10(sch)

        axes[2].plot(t, sch_db, 'b-', linewidth=1, label='Schroeder Curve')
        axes[2].axhline(-5, color='g', linestyle='--', alpha=0.5, label='-5dB')
        axes[2].axhline(-35, color='orange', linestyle='--', alpha=0.5, label='-35dB')
        axes[2].axhline(-60, color='r', linestyle='--', alpha=0.5, label='-60dB')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Energy (dB)')
        axes[2].set_title('Schroeder Decay Curve (for RT60 calculation)')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        axes[2].set_ylim(-80, 5)

        plt.tight_layout()
        plt.savefig('data/plots/diagnosis.png', dpi=150)
        print(f"   ✅ 诊断图已保存: data/plots/diagnosis.png")
        plt.close()

    except Exception as e:
        print(f"   ❌ 生成诊断图错误: {e}")

    # 建议
    print("\n" + "="*70)
    print("💡 改善测量质量的建议:")
    print("="*70)
    print("1. 增加录音增益")
    print("   - 确保录音峰值在-6dB左右")
    print("   - 避免过低导致噪声影响")
    print()
    print("2. 降低背景噪声")
    print("   - 关闭空调、风扇等噪声源")
    print("   - 选择安静的时段测量")
    print()
    print("3. 优化设备摆放")
    print("   - 扬声器和麦克风距离1-3米")
    print("   - 避免正对墙壁")
    print("   - 麦克风高度1.2米左右")
    print()
    print("4. 延长录音时间")
    print("   - 确保录制到混响完全衰减")
    print("   - 建议record_tail设置为5秒以上")
    print()
    print("5. 检查设备")
    print("   - 确保麦克风和扬声器工作正常")
    print("   - 检查音频接口设置")
    print("="*70)

if __name__ == "__main__":
    diagnose_measurement()
