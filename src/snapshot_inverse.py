import numpy as np
def snapshot_moments(d):
 r=[]
 for t,x,c in d['snapshots']:
  m=np.trapz(c,x); xc=np.trapz(x*c,x)/m; s=np.sqrt(np.trapz((x-xc)**2*c,x)/m); r.append([t,xc,s,m])
 return np.array(r)
