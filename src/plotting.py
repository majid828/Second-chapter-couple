import matplotlib.pyplot as plt
def make_plots(b,s,t,r):
 plt.plot(t['h'],label='true'); plt.plot(r['h'],label='recovered'); plt.legend(); plt.savefig('results/figures/memory_kernel.png'); plt.close()
