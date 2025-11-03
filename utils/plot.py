
import numpy as np, matplotlib.pyplot as plt, os

def plot_ir(ir,fs,ref=None,path="data/plots/ir.png"):
    t=np.arange(len(ir))/fs
    plt.figure(figsize=(12,4))
    plt.plot(t,ir,label="IR")
    d=np.argmax(np.abs(ir))
    plt.axvline(t[d],color='g',ls="--",label="Direct")
    if ref is not None:
        for tt in ref: plt.axvline(tt,color='r',ls=":",alpha=.6)
    plt.grid(); plt.legend(); plt.title("Impulse Response")
    os.makedirs("data/plots",exist_ok=True)
    plt.savefig(path); plt.close()
