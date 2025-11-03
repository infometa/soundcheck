
import numpy as np, scipy.signal as sig

def sync_and_trim(rec, sweep):
    xc=sig.correlate(rec, sweep)
    peak=np.argmax(xc)
    start = peak - (len(sweep)-1)
    if start<0: start=0
    return rec[start:]
