#!/usr/bin/env python3
"""
SoundCheck - 室内声学测量系统
自动扫频测量 + IR提取 + RT60/C50计算
"""

from core.device import choose_device
from core.sweep import generate_sweep
from core.record import play_and_record
from core.sync import sync_and_trim
from core.ir import extract_ir
from core.metrics import RT60, C50
from core.reflections import reflections
from core.separate import separate_ir_components, export_ir_comparison
from utils.plot import plot_ir
from utils.report import generate_report
from utils.config import load_config

def main():
    print("=" * 60)
    print("🔊 SoundCheck - 室内声学测量系统")
    print("=" * 60)

    try:
        # Load config
        cfg = load_config()
        fs = int(float(cfg.get("fs", 48000)))

        # Step 1: Choose audio device
        print("\n[1/9] 选择音频设备...")
        choose_device()

        # Step 2: Generate sweep signal
        print("\n[2/9] 生成扫频信号...")
        sig, inv = generate_sweep()
        print(f"✅ 扫频信号生成完成 ({len(sig)/fs:.1f}秒)")

        # Step 3: Play and record
        print("\n[3/9] 播放并录制...")
        rec = play_and_record(sig)

        # Step 4: Synchronize and trim
        print("\n[4/9] 同步和裁剪录音...")
        rec2 = sync_and_trim(rec, sig)

        # Step 5: Extract impulse response
        print("\n[5/9] 提取脉冲响应 (IR)...")
        ir = extract_ir(rec2, inv)

        # Step 6: Calculate acoustic metrics
        print("\n[6/9] 计算声学指标...")
        rt = RT60(ir)
        c = C50(ir)
        print(f"   RT60: {rt:.3f} 秒" if not float('nan') == rt else "   RT60: N/A")
        print(f"   C50: {c:.2f} dB" if not float('nan') == c else "   C50: N/A")

        # Step 7: Detect reflections and plot
        print("\n[7/9] 检测反射并绘制图表...")
        ref = reflections(ir)
        plot_ir(ir, fs, ref)

        # Step 8: Separate IR components
        print("\n[8/9] 分离IR成分并导出WAV文件...")
        separated_paths = separate_ir_components(ir)
        export_ir_comparison(ir)

        # Step 9: Generate report
        print("\n[9/9] 生成PDF报告...")
        generate_report(rt, c)
        print("✅ PDF报告生成完成")

        # Summary
        print("\n" + "=" * 60)
        print("✅ 测量完成！")
        print("=" * 60)
        print(f"📊 RT60 (混响时间):  {rt:.3f} 秒" if not float('nan') == rt else "📊 RT60: N/A")
        print(f"📊 C50 (清晰度):     {c:.2f} dB" if not float('nan') == c else "📊 C50: N/A")
        print(f"🔍 检测到反射:       {len(ref)} 个")
        print(f"\n📁 输出文件:")
        print(f"   波形图:     data/plots/ir.png")
        print(f"   报告:       data/reports/report.pdf")
        print(f"   直达声:     {separated_paths['direct']}")
        print(f"   早反射:     {separated_paths['early']}")
        print(f"   混响尾声:   {separated_paths['late']}")
        print(f"   对比文件:   data/separated/comparison.wav")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断")
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
