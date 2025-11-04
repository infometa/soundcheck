
import os
import numpy as np
import soundfile as sf
from utils.config import load_config


cfg = load_config()
FS = int(float(cfg.get("fs", 48000)))
EARLY_REFL_TIME = float(cfg.get("early_reflection_time", 0.08))


def separate_ir_components(ir, output_dir="data/separated"):
    """
    将脉冲响应分离为三个部分并保存为单独的wav文件：
    1. 直达声 (Direct Sound)
    2. 早反射 (Early Reflections)
    3. 混响尾声 (Late Reverb)

    Args:
        ir: 脉冲响应数组
        output_dir: 输出目录

    Returns:
        dict: 包含三个部分的文件路径
    """
    os.makedirs(output_dir, exist_ok=True)

    # 找到直达声位置（最大峰值）
    direct_idx = np.argmax(np.abs(ir))

    # 计算时间边界
    # 直达声窗口：峰值前后各5ms
    direct_window_samples = int(0.005 * FS)  # 5ms
    direct_start = max(0, direct_idx - direct_window_samples)
    direct_end = min(len(ir), direct_idx + direct_window_samples)

    # 早反射窗口：直达声结束到EARLY_REFL_TIME之后
    early_end_idx = min(len(ir), direct_idx + int(EARLY_REFL_TIME * FS))

    # === 1. 直达声 (Direct Sound) ===
    direct_sound = np.zeros_like(ir)
    direct_sound[direct_start:direct_end] = ir[direct_start:direct_end]

    # === 2. 早反射 (Early Reflections) ===
    early_reflections = np.zeros_like(ir)
    if direct_end < early_end_idx:
        early_reflections[direct_end:early_end_idx] = ir[direct_end:early_end_idx]

    # === 3. 混响尾声 (Late Reverb) ===
    late_reverb = np.zeros_like(ir)
    if early_end_idx < len(ir):
        late_reverb[early_end_idx:] = ir[early_end_idx:]

    # 归一化（保持相对能量比例）
    max_val = np.max(np.abs(ir))
    if max_val > 0:
        direct_sound = direct_sound / max_val
        early_reflections = early_reflections / max_val
        late_reverb = late_reverb / max_val

    # 保存文件
    paths = {
        'direct': os.path.join(output_dir, 'direct_sound.wav'),
        'early': os.path.join(output_dir, 'early_reflections.wav'),
        'late': os.path.join(output_dir, 'late_reverb.wav'),
    }

    sf.write(paths['direct'], direct_sound, FS)
    sf.write(paths['early'], early_reflections, FS)
    sf.write(paths['late'], late_reverb, FS)

    # 计算各部分能量
    direct_energy = np.sum(direct_sound ** 2)
    early_energy = np.sum(early_reflections ** 2)
    late_energy = np.sum(late_reverb ** 2)
    total_energy = direct_energy + early_energy + late_energy

    # 输出信息
    print(f"\n{'='*60}")
    print("📁 IR分离完成 - 已保存为单独的WAV文件")
    print(f"{'='*60}")
    print(f"🔴 直达声:     {paths['direct']}")
    print(f"   时间窗口:   {direct_start/FS*1000:.1f} - {direct_end/FS*1000:.1f} ms")
    print(f"   能量占比:   {direct_energy/total_energy*100:.1f}%")
    print()
    print(f"🔵 早反射:     {paths['early']}")
    print(f"   时间窗口:   {direct_end/FS*1000:.1f} - {early_end_idx/FS*1000:.1f} ms")
    print(f"   能量占比:   {early_energy/total_energy*100:.1f}%")
    print()
    print(f"⚪ 混响尾声:   {paths['late']}")
    print(f"   时间窗口:   {early_end_idx/FS*1000:.1f} ms - 结束")
    print(f"   能量占比:   {late_energy/total_energy*100:.1f}%")
    print(f"{'='*60}\n")

    return paths


def export_ir_comparison(ir, output_path="data/separated/comparison.wav"):
    """
    导出一个包含4个通道的对比文件：
    通道1: 完整IR
    通道2: 直达声
    通道3: 早反射
    通道4: 混响尾声

    这样可以在DAW中直接对比各部分
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 找到直达声位置
    direct_idx = np.argmax(np.abs(ir))

    # 计算时间边界
    direct_window_samples = int(0.005 * FS)
    direct_start = max(0, direct_idx - direct_window_samples)
    direct_end = min(len(ir), direct_idx + direct_window_samples)
    early_end_idx = min(len(ir), direct_idx + int(EARLY_REFL_TIME * FS))

    # 创建分离信号
    direct_sound = np.zeros_like(ir)
    direct_sound[direct_start:direct_end] = ir[direct_start:direct_end]

    early_reflections = np.zeros_like(ir)
    if direct_end < early_end_idx:
        early_reflections[direct_end:early_end_idx] = ir[direct_end:early_end_idx]

    late_reverb = np.zeros_like(ir)
    if early_end_idx < len(ir):
        late_reverb[early_end_idx:] = ir[early_end_idx:]

    # 归一化
    max_val = np.max(np.abs(ir))
    if max_val > 0:
        ir_norm = ir / max_val
        direct_sound = direct_sound / max_val
        early_reflections = early_reflections / max_val
        late_reverb = late_reverb / max_val

    # 合并为多通道
    multichannel = np.column_stack([ir_norm, direct_sound, early_reflections, late_reverb])

    # 保存
    sf.write(output_path, multichannel, FS)

    print(f"💾 对比文件已保存: {output_path}")
    print(f"   通道1: 完整IR")
    print(f"   通道2: 直达声")
    print(f"   通道3: 早反射")
    print(f"   通道4: 混响尾声")
    print()

    return output_path
