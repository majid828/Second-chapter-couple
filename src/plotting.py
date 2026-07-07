
from pathlib import Path
import matplotlib.pyplot as plt

def plot_all(btc,snap,true,H,G,pred):
    out=Path("results/figures")
    out.mkdir(parents=True,exist_ok=True)

    plt.figure()
    plt.plot(true["H"],label="true H")
    plt.plot(H["H"],label="recovered H")
    plt.legend()
    plt.savefig(out/"memory_kernel_recovery.png",dpi=300)
    plt.close()
