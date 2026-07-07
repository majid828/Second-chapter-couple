import numpy as np
from scipy.optimize import lsq_linear
def recover_kernels(d):
 t=d['time']; f=d['btc']; g=f.copy(); g[t>0.4*t.max()]=0; g/=np.trapz(g,t)
 n=len(t); dt=t[1]-t[0]; G=np.zeros((n,n))
 for i in range(n):
  for j in range(i+1): G[i,j]=g[i-j]*dt
 h=lsq_linear(np.vstack([G,np.sqrt(1e-3)*np.eye(n)]),np.r_[f,np.zeros(n)],bounds=(0,np.inf)).x
 h/=np.trapz(h,t)
 return {'time':t,'g':g,'h':h}
