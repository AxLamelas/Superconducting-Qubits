import numpy as np 
import h5py
import os


def writeResultsToFile(t,ps,deltas,file):
    with h5py.File(file,'w') as f:
        f.create_dataset('t',data=t,maxshape=(len(t),))
        for label,data in zip(['ps','deltas'],[ps,deltas]):
            f.create_dataset(label,data=data,maxshape=(None,len(data[0])),compression="gzip", compression_opts=9)


def addResultsToFile(ps,deltas,file="./simData.h5"):
    with h5py.File(file,'a') as f:
        for fdata,data in zip([f['ps'],f['deltas']],[ps,deltas]):
            fdata.resize(fdata.shape[0]+len(data), axis=0)   
            fdata[-len(data):] = data

def joinResultFiles(outputFile,*files):
    files = iter(files)
    chunckLength = 1000
    with h5py.File(outputFile,"w") as outF:
        file = next(files)
        print("Loading first file")
        with h5py.File(file,"r") as f:
            n = f["ps"].shape[0]//chunckLength
            outF.create_dataset("t",data=f['t'][0:],maxshape=(f["t"].shape[0],),compression="gzip", compression_opts=9)
            outF.create_dataset("ps",data=f["ps"][0:chunckLength],maxshape=(None,f["ps"].shape[1]),compression="gzip", compression_opts=9)
            outF.create_dataset("deltas",data=f["deltas"][0:chunckLength],maxshape=(None,f["deltas"].shape[1]),compression="gzip", compression_opts=9)
            for i in range(1,n):
                for fdata,data in zip([outF['ps'],outF['deltas']],[f['ps'][i*chunckLength:(i+1)*chunckLength],f['deltas'][i*chunckLength:(i+1)*chunckLength]]):
                    fdata.resize(fdata.shape[0]+len(data), axis=0)   
                    fdata[-len(data):] = data
            
        for file in files:
            print("Loading file")
            with h5py.File(file,"r") as f:
                n = f["ps"].shape[0]//chunckLength
                for i in range(0,n):
                    for fdata,data in zip([outF['ps'],outF['deltas']],[f['ps'][i*chunckLength:(i+1)*chunckLength],f['deltas'][i*chunckLength:(i+1)*chunckLength]]):
                        fdata.resize(fdata.shape[0]+len(data), axis=0)   
                        fdata[-len(data):] = data


