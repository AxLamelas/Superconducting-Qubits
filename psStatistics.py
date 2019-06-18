import matplotlib.pyplot as plt
import numpy as np
from math import floor,ceil,log10
from sklearn.decomposition import PCA
 
def plotIn2D(data):
    pca = PCA(n_components=2)
    data2D = pca.fit_transform(data)
    plt.scatter(data2D[:,0],data2D[:,1],marker="*")
    plt.show()


def logHist(x):
    lower = floor(np.log10(min(filter(lambda y: y!=0,x))))
    higher = ceil(np.log10(max(filter(lambda y: y!=0,x))))
    logbins = np.logspace(lower,higher,50)
    if 0 in x:
        logbins = np.append(logbins,0)
        logbins.sort()
    
    plt.hist(x,bins=logbins)
    plt.xscale("log")
    plt.show()

if __name__=="__main__":
    import h5py
    with h5py.File("simData.h5") as f:
        plotIn2D(f["ps"][:])