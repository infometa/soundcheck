
import numpy as np, scipy.signal as sig

def sync_and_trim(rec, sweep):
    """Synchronize recording with sweep signal using cross-correlation."""
    if len(rec) < len(sweep):
        raise ValueError(f"录音长度 ({len(rec)}) 短于扫频信号 ({len(sweep)})")

    xc=sig.correlate(rec, sweep, mode='full')
    peak=np.argmax(xc)
    start = peak - (len(sweep)-1)

    if start<0:
        print(f"⚠️ 警告：同步起始点为负 ({start})，重置为0")
        start=0

    if start >= len(rec):
        raise ValueError(f"同步失败：起始点 {start} 超出录音范围 {len(rec)}")

    result = rec[start:]

    if len(result) < len(sweep) // 2:
        print(f"⚠️ 警告：同步后的录音较短 ({len(result)} 采样点)")

    print(f"✅ 同步完成，起始点: {start}, 相关峰值: {xc[peak]:.2e}")
    return result
