
import numpy as np,scipy.signal as sig
def modes(ir,fs):
    N=len(ir); win=ir*sig.hann(N)
    H=np.fft.rfft(win); f=np.fft.rfftfreq(N,1/fs)
    Hdb=20*np.log10(np.abs(H)/np.max(np.abs(H))+1e-12)
    return f,Hdb
