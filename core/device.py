
import sounddevice as sd

def choose_device():
    """让用户分别选择麦克风（输入）和扬声器（输出）设备"""
    devices = sd.query_devices()

    # 获取输入设备列表
    input_devices = []
    output_devices = []

    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            input_devices.append((i, d))
        if d['max_output_channels'] > 0:
            output_devices.append((i, d))

    # 显示麦克风列表
    print("\n" + "="*60)
    print("🎤 可用麦克风设备 (输入)")
    print("="*60)
    for idx, (dev_id, dev) in enumerate(input_devices):
        default_marker = " [默认]" if dev_id == sd.default.device[0] else ""
        print(f"  {dev_id:2d}. {dev['name']:<45} ({dev['max_input_channels']}通道){default_marker}")

    # 显示扬声器列表
    print("\n" + "="*60)
    print("🔊 可用扬声器设备 (输出)")
    print("="*60)
    for idx, (dev_id, dev) in enumerate(output_devices):
        default_marker = " [默认]" if dev_id == sd.default.device[1] else ""
        print(f"  {dev_id:2d}. {dev['name']:<45} ({dev['max_output_channels']}通道){default_marker}")

    print("\n" + "="*60)

    try:
        # 选择麦克风
        if len(input_devices) == 0:
            print("❌ 错误：没有找到可用的输入设备")
            raise SystemExit

        inp_str = input(f"选择麦克风ID (默认: {sd.default.device[0]}): ").strip()
        if inp_str == "":
            inp = sd.default.device[0]
        else:
            inp = int(inp_str)

        # 验证输入设备
        if inp >= len(devices) or devices[inp]['max_input_channels'] == 0:
            print(f"❌ 错误：设备 {inp} 不是有效的输入设备")
            raise SystemExit

        # 选择扬声器
        if len(output_devices) == 0:
            print("❌ 错误：没有找到可用的输出设备")
            raise SystemExit

        outp_str = input(f"选择扬声器ID (默认: {sd.default.device[1]}): ").strip()
        if outp_str == "":
            outp = sd.default.device[1]
        else:
            outp = int(outp_str)

        # 验证输出设备
        if outp >= len(devices) or devices[outp]['max_output_channels'] == 0:
            print(f"❌ 错误：设备 {outp} 不是有效的输出设备")
            raise SystemExit

        # 设置设备
        sd.default.device = (inp, outp)
        print(f"\n✅ 已设置音频设备:")
        print(f"   🎤 麦克风: {devices[inp]['name']}")
        print(f"   🔊 扬声器: {devices[outp]['name']}")

    except ValueError as e:
        print(f"❌ 输入无效: {e}")
        raise SystemExit
    except Exception as e:
        print(f"❌ 声卡选择失败: {e}")
        raise SystemExit
