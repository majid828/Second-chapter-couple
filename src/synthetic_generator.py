
import numpy as np

def normalize(y,x):
    return y/(np.trapz(y,x)+1e-15)

def generate_data():
    t=np.linspace(0.1,100,300)

    g=np.exp(-(np.log(t+0.5)-1.5)**2/(2*0.5**2))/(t+0.5)
    g=normalize(g,t)

    H=np.exp(-(0.15*t)**0.65)
    H=normalize(H,t)

    btc=np.convolve(g,H)[:len(t)]*(t[1]-t[0])

    x=np.linspace(0,100,200)
    snapshots=[]
    G=[]
    for ti in [20,40,60,80]:
        xc=0.6*ti
        sig=np.sqrt(0.8*ti)
        c=np.exp(-(x-xc)**2/(2*sig**2))
        snapshots.append((ti,x,c))
        G.append(c/(np.trapz(c,x)+1e-15))

    return {"time":t,"btc":btc},{"snapshots":snapshots},{"g":g,"H":H,"G":G}
