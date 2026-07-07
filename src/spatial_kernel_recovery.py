
import numpy as np

def recover_spatial(data):
    kernels=[]
    for t,x,c in data["snapshots"]:
        p=c/(np.trapz(c,x)+1e-15)
        kernels.append((t,x,p))
    return {"kernels":kernels}
