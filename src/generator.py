import numpy as np
def generate_all():
 t=np.linspace(.1,100,300)
 g=np.exp(-(np.log(t+.5)-1.5)**2/(2*.5**2))/(t+.5); g/=np.trapz(g,t)
 h=np.exp(-(.15*t)**.65); h/=np.trapz(h,t)
 btc=np.convolve(g,h)[:len(t)]*(t[1]-t[0])
 snaps=[]
 x=np.linspace(0,100,200)
 for ti in [20,40,60,80]: snaps.append((ti,x,np.exp(-(x-.6*ti)**2/(2*(2*ti)**2))))
 return {'time':t,'btc':btc},{'snapshots':snaps},{'g':g,'h':h}
