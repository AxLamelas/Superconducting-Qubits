import numpy as np
import h5py
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import pandas as pd
from joblib import dump, load
from umap import UMAP
from random import choice
from visualization import plotSimulation, plotClusters
import matplotlib.pyplot as plt
from numpy.random import choice

def kmeansCluster(data,clusterFile="clusterModels"):
    
    nClusters = list(range(2,51))
    model = list()

    for n in nClusters:
        print(n)        
        model.append(KMeans(n_clusters=n,n_init=50,max_iter=50000,tol=1e-20).fit(data))
        
    dump(model, f'{clusterFile}.joblib')

def gmCluster(data,clusterFile="clusterModels",bicfig="bicSeries"):
    nComponents = list(range(1,16))
    models = list()
    bics = list()
    for c in nComponents:
        print(c)
        models.append(GaussianMixture(n_components=c,reg_covar=1e10,max_iter=500).fit(data))
        bics.append(models[-1].bic(data))

    fig,ax = plt.subplots(1,1,figsize=(15,10))
    ax.scatter(nComponents,bics,c="grey")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("BIC")
    plt.tight_layout()
    fig.savefig(f"./figures/{bicfig}")

    dump(models, f'{clusterFile}.joblib')



def sampleFromClass(labels,n,l,dataFile="./simSeriesData.h5"):
    inds = [j for j in range(len(labels)) if labels[j] == l]
    if len(inds) <= n:
        print("The class is constitude by less or the same number of samples you requested.")
        n = len(inds)

    plotSimulation(dataFile,np.sort(choice(inds,n)),prefix=f"sampleFrom{l}")

def saveDataFromClass(labels,n,l,dataFile="./simSeriesData.h5"):
    inds = [j for j in range(len(labels)) if labels[j] == l]
    with h5py.File(dataFile,"r") as f:
        with h5py.File(f"sampleFromClass{l}.h5","w") as sf:
            if len(inds) <= n:
                print("The class is constitude by less or the same number of samples you requested.")
                n = len(inds)
            sf.create_dataset("deltas",data=f["deltas"][np.sort(choice(inds,n,replace=False)),:])
            sf.create_dataset("ps",data=f["ps"][np.sort(choice(inds,n,replace=False)),:])

def representation2D(data):
    if isinstance(data,str):
        data = pd.read_hdf(data,'table').values

    data2D= UMAP().fit_transform(data)

    return data2D

def plotOutAndParamSpace(conf,nModel):
    featureFile = f"tsfreshFeatures{conf}.h5"
    models = load(f'clusterModels{conf}.joblib')

    with h5py.File(f"p{conf}.h5","r") as f:
        ps = f["ps"][:]
   

    data2D = representation2D(featureFile)
    param2D = UMAP().fit_transform(ps)

    plotClusters(models[nModel],featureFile,data2D,f'out{conf}')
    plotClusters(models[nModel],featureFile,param2D,name=f'param{conf}')


def plotElbow(conf):
    models = load(f'clusterModels{conf}.joblib')
    d = list()
    for m in models:
        d.append(m.inertia_)
    fig,ax = plt.subplots(1,1,figsize=(15,10))
    ax.scatter(list(range(2,51)),d,c='grey')
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Inertia")
    plt.tight_layout()
    fig.savefig(f"./figures/elbow{conf}")

