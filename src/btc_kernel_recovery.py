
import numpy as np
from scipy.optimize import lsq_linear

def recover_memory(data):
    t=data["time"]
    c=data["btc"]

    g=c.copy()
    g[t>0.4*t.max()]=0
    g=np.maximum(g,0)
    g/=np.trapz(g,t)

    n=len(t)
    dt=t[1]-t[0]

    G=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1):
            G[i,j]=g[i-j]*dt

    A=np.vstack([G,np.sqrt(1e-3)*np.eye(n)])
    b=np.r_[c,np.zeros(n)]

    H=lsq_linear(A,b,bounds=(0,np.inf)).x
    H/=np.trapz(H,t)

    return {"time":t,"g":g,"H":H}
