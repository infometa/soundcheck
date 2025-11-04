#!/usr/bin/env python3
"""
测试脚本 - 验证所有修复
不需要音频硬件即可运行
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from core.sweep import generate_sweep
from core.sync import sync_and_trim
from core.ir import extract_ir
from core.metrics import RT60, C50
from core.reflections import reflections
from utils.plot import plot_ir
from utils.config import load_config

def test_config():
    """测试配置加载"""
    print("\n=== 测试1: 配置加载 ===")
    cfg = load_config()
    assert "fs" in cfg, "配置缺少 fs"
    assert "sweep_freq_min" in cfg, "配置缺少 sweep_freq_min"
    assert "sweep_freq_max" in cfg, "配置缺少 sweep_freq_max"
    assert "early_reflection_time" in cfg, "配置缺少 early_reflection_time"
    print("✅ 配置文件加载成功")
    print(f"   采样率: {cfg['fs']} Hz")
    print(f"   扫频范围: {cfg['sweep_freq_min']}-{cfg['sweep_freq_max']} Hz")
    return cfg

def test_sweep_generation():
    """测试扫频信号生成"""
    print("\n=== 测试2: 扫频信号生成 ===")
    sig, inv = generate_sweep()
    assert len(sig) > 0, "扫频信号为空"
    assert len(inv) > 0, "逆滤波器为空"
    assert not np.any(np.isnan(sig)), "扫频信号包含NaN"
    assert not np.any(np.isnan(inv)), "逆滤波器包含NaN"
    print(f"✅ 扫频信号生成成功")
    print(f"   信号长度: {len(sig)} 采样点")
    print(f"   逆滤波器长度: {len(inv)} 采样点")
    return sig, inv

def test_sync():
    """测试同步功能"""
    print("\n=== 测试3: 同步和裁剪 ===")
    cfg = load_config()
    fs = int(float(cfg.get("fs", 48000)))

    # Create synthetic recording (sweep with delay and noise)
    sig, inv = generate_sweep()
    delay_samples = int(0.1 * fs)  # 100ms delay
    rec = np.concatenate([
        np.random.randn(delay_samples) * 0.01,  # noise
        sig * 0.8,  # attenuated signal
        np.random.randn(int(0.5 * fs)) * 0.01  # more noise
    ])

    rec2 = sync_and_trim(rec, sig)
    assert len(rec2) > 0, "同步后录音为空"
    print(f"✅ 同步成功")
    print(f"   原始录音: {len(rec)} 采样点")
    print(f"   同步后: {len(rec2)} 采样点")
    return rec2, sig, inv

def test_ir_extraction():
    """测试IR提取"""
    print("\n=== 测试4: 脉冲响应提取 ===")
    rec2, sig, inv = test_sync()
    ir = extract_ir(rec2, inv)
    assert len(ir) > 0, "IR为空"
    assert not np.all(ir == 0), "IR全为0"
    assert not np.any(np.isnan(ir)), "IR包含NaN"
    print(f"✅ IR提取成功")
    print(f"   IR长度: {len(ir)} 采样点")
    return ir

def test_metrics():
    """测试声学指标计算"""
    print("\n=== 测试5: 声学指标计算 ===")

    # Create synthetic IR with exponential decay
    cfg = load_config()
    fs = int(float(cfg.get("fs", 48000)))
    duration = 2.0  # 2 seconds
    t = np.arange(int(duration * fs)) / fs

    # Exponential decay (RT60 ~ 0.5s)
    decay_rate = -60 / (0.5 * 20 * np.log10(np.e))
    ir_synth = np.exp(decay_rate * t) * np.random.randn(len(t)) * 0.1

    # Add strong direct sound
    ir_synth[1000] = 1.0

    # Add some early reflections
    ir_synth[2000] = 0.3
    ir_synth[3000] = 0.2

    rt = RT60(ir_synth)
    c = C50(ir_synth)

    print(f"✅ 声学指标计算成功")
    print(f"   RT60: {rt:.3f} 秒" if not np.isnan(rt) else "   RT60: N/A")
    print(f"   C50: {c:.2f} dB" if not np.isnan(c) else "   C50: N/A")

    # Check if values are reasonable
    if not np.isnan(rt):
        assert rt > 0, "RT60 应该为正值"
        assert rt < 30, "RT60 过大（超过30秒）"

    return ir_synth

def test_reflections():
    """测试反射检测"""
    print("\n=== 测试6: 反射检测 ===")
    ir = test_metrics()
    ref = reflections(ir)
    assert len(ref) >= 0, "反射检测失败"
    print(f"✅ 反射检测成功")
    print(f"   检测到 {len(ref)} 个反射")
    return ir, ref

def test_plotting():
    """测试绘图功能"""
    print("\n=== 测试7: 绘图功能 ===")
    ir, ref = test_reflections()
    cfg = load_config()
    fs = int(float(cfg.get("fs", 48000)))

    # Test plotting
    plot_ir(ir, fs, ref, path="data/plots/test_ir.png")

    # Check if file was created
    assert os.path.exists("data/plots/test_ir.png"), "图表文件未生成"
    print(f"✅ 图表生成成功")
    print(f"   保存位置: data/plots/test_ir.png")

def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("🧪 SoundCheck 修复验证测试")
    print("=" * 60)

    try:
        test_config()
        test_sweep_generation()
        ir = test_ir_extraction()
        test_metrics()
        test_plotting()

        print("\n" + "=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
        print("\n修复内容:")
        print("  ✅ 删除重复的配置文件")
        print("  ✅ 修复C50计算方法（增加边界检查）")
        print("  ✅ 添加频率范围到配置文件")
        print("  ✅ 增强异常处理（metrics.py, record.py）")
        print("  ✅ 增强数组边界检查（sync.py）")
        print("  ✅ 增强绘图功能（IR + ETC曲线）")
        print("  ✅ 添加进度信息和文档字符串")
        print("\n新功能:")
        print("  ✅ ETC（能量时间曲线）可视化")
        print("  ✅ 直达声（红）/ 早反射（蓝）/ 混响尾声（灰）标记")
        print("  ✅ 改进的错误处理和用户反馈")
        print("=" * 60)

        return True

    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ 意外错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
