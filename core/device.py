
import sounddevice as sd

def choose_device():
    print("=== 音频设备列表 ===")
    for i, d in enumerate(sd.query_devices()):
        print(i, d['name'])
    try:
        inp = int(input("输入录音设备ID: "))
        outp = int(input("输入播放设备ID: "))
        sd.default.device = (inp, outp)
        print("✅ 已设置音频设备")
    except Exception as e:
        print("❌ 声卡选择失败:", e)
        raise SystemExit
