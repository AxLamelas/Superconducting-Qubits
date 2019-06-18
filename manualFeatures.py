
from scipy.signal import find_peaks
import numpy as np
from numpy.fft import fft, fftshift
from math import floor
from statistics import mean
from numba import jit
import h5py



def freqsWeigthAbove(x,t,I):
    f = np.array([i/(len(t)*(t[1]-t[0])) for i in range(round(-len(t)/2),round(len(t)/2),1)])
    It = np.array(fftshift(fft(I)))
    s = abs(It)
    s = s/max(s)
    inds,_ = find_peaks(s)
    inds = [e for e in inds if f[e]>=0 and s[e] >= x]
    return f[inds]

@jit
def peakMean(x):
    inds,_ = find_peaks(x.copy())
    if len(inds)==0:
        return 0
    return mean(x[inds])

@jit
def maxPeak(x):
    inds, _ =find_peaks(x.copy())
    if len(inds) == 0:
        return max(x)
    return max(x[inds])
            

@jit
def averageInIntervals(x,n):
    s = floor(len(x)/n)
    rs = np.zeros(n)
    for i in range(0,n):
        rs[i] = mean(x[i*s:(i+1)*s])
    
    return rs
    

def nFrequencies(n,t,x):
    f = np.array([i/(len(t)*(t[1]-t[0])) for i in range(round(-len(t)/2),round(len(t)/2),1)])
    It = np.array(fftshift(fft(x)))
    s = abs(It)
    if max(s) != 0:
        s = s/max(s)
    inds,_ = find_peaks(s)
    inds = list(filter(lambda i: f[i]>=0,inds))
    if len(inds) == 0:
        return np.zeros(n)
    l = [(ind,x[ind])for ind in inds]
    sorted(l,key= lambda x:x[1])
    if len(inds) < n:
        return np.concatenate((f[inds],f[inds[0]]*np.ones(n-len(inds))))
    else:
        return np.array(f[[l[i][0] for i in range(0,n)]])


def nFrequenciesFromFile(n,file):
        
    Ic  = 10**-6
    phi0= 6.62607004*10**-34/(2*1.60217662*10**-19)
    with h5py.File(file,"r") as f:
        data = f["deltas"][:]
        t = np.transpose(np.tile(phi0/(2*np.pi*Ic*f["ps"][:,0]),(10**6,1)))*np.tile(np.linspace(0,10,10**6),(len(f["ps"][:,0]),1))
    
    return [nFrequencies(n,t[i,:],data[i,:]) for i in range(t.shape[0])]


def buildFeatureSet(t,x):
    np.array([*nFrequencies(3,t,x),maxPeak(x),peakMean(x), *averageInIntervals(x,20)])

def buildScaledFeatureSet(t,p,x):
    Ic=10**-6
    fs = np.array(nFrequencies(3,t,x))/p[5]
    vs = np.array(averageInIntervals(x,20))/Ic
    return np.array([*fs,maxPeak(x)/Ic,peakMean(x)/Ic, *vs])

vBuildScaledFeatureSet = np.vectorize(buildScaledFeatureSet,signature='(i),(j),(i)->(k)')
vBuildFeatureSet = np.vectorize(buildScaledFeatureSet,signature='(i),(j)->(k)')



def main():
    with h5py.File('simData.h5','r') as f:
        t,ps,Is,deltas= f['t'],f['ps'],f['Is'],f['deltas']
        scaled=True
        
        if scaled:
            print("Calculating")
            features = vBuildScaledFeatureSet(t,ps,Is)
            # features = [buildScaledFeatureSet(t,ps[i],Is[i]) for i in range(0,len(Is))]
            print("Saving")
            with h5py.File('scaledManualFeatures.h5', 'w') as hf:
                hf.create_dataset("features",  data=features)

        else:
            print("Calculating")
            features = vBuildFeatureSet(t,Is)
            
            #features = [buildFeatureSet(t,i) for i in Is]
            print("Saving")
            with h5py.File('manualFeatures.h5', 'w') as hf:
                hf.create_dataset("features",  data=features)



if __name__=="__main__":
    main()

