from keras.models import load_model
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm

plt.rc('text',usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{siunitx}')
plt.rc('font',size=40)
plt.figure(figsize=(20,13))



def plotEnconder(ind=[99,38,3],encoderFile='en_Is1.h5',dataFile="./simSeriesData.h5",prefix="decoder"):
    model = load_model(encoderFile)

    with h5py.File(dataFile,"r") as f:
        for i in ind:
            y = np.sin(f["deltas"][i,:])
            yEn = model.predict(y.reshape(1,y.shape[0],1,1))
            yEn = yEn.reshape(y.shape[0])
            plt.clf()
            plt.plot(f["t"],y,'grey',label='Original')
            plt.plot(f["t"],yEn,'grey',linestyle="--",label='Reconstructed')
            plt.xlabel(r'$\tau$')
            plt.ylabel(r'$I/I_c$')
            plt.tight_layout()
            plt.savefig(f"./figures/{prefix}_{i}")



def plotSimulation(data,ind=None,prefix="series",xrange=(0,10)):
    if isinstance(data,str):
        f = h5py.File(data,"r")
        data = f["deltas"]
        close= True

    if ind == None:
        ind = range(0,len(data))
        
    t = np.linspace(0,10,10**6)
    for i in ind:
        y = np.sin(data[i,:])
        plt.clf()
        plt.plot(t,y,'grey')
        plt.xlabel(r'$\tau$')
        plt.ylabel(r'$I/I_c$')
        plt.ylim(-1.05,1.05)
        plt.xlim(xrange[0],xrange[1])
        R,L,C,forceDc,forceAc,w,delta0,_ = f["ps"][i,:]
        if prefix == "parallel":
            plt.title(f"$R = \\num{{{R:.2e}}}\:\si{{\ohm}},\:L = \\num{{{L:.2e}}}\:\si{{\henry}},\:C = \\num{{{C:.2e}}}\:\si{{\\farad}},$"+ "\n" 
                +f"$\:i_{{dc}} = \\num{{{forceDc:.2e}}}\:\si{{\\ampere}},i_{{ac}} = \\num{{{forceAc:.2e}}}\:\si{{\\ampere}},$"+"\n"
                    +f"$\:\omega = \\num{{{w:.2e}}}\:\si{{\hertz}},\:\delta_0=\\num{{{delta0:.2e}}}$",multialignment='center')     
        else:
            plt.title(f"$R = \\num{{{R:.2e}}}\:\si{{\ohm}},\:L = \\num{{{L:.2e}}}\:\si{{\henry}},\:C = \\num{{{C:.2e}}}\:\si{{\\farad}},$"+ "\n" 
                +f"$\:v_{{dc}} = \\num{{{forceDc:.2e}}}\:\si{{\\volt}},v_{{ac}} = \\num{{{forceAc:.2e}}}\:\si{{\\volt}},$"+"\n"
                    +f"$\:\omega = \\num{{{w:.2e}}}\:\si{{\hertz}},\:\delta_0=\\num{{{delta0:.2e}}}$",multialignment='center')
                       
        plt.tight_layout()
        plt.savefig(f"./figures/{prefix}_{i}")
    
    if close:
        f.close()



def plotClusters(m,data,p,name="fig"):
    if isinstance(data,str):
        data = pd.read_hdf(data,'table').values

    labels = m.predict(data)    
    n = len(set(labels))-1
    fig, ax = plt.subplots(1,1,figsize=(11,11))
    scatter = ax.scatter(p[:,0],p[:,1],alpha=.5,c=labels,s =100)#cmap=cm.get_cmap("winter")
    legend = ax.legend(*scatter.legend_elements(num=n),
                    loc='lower right', title="Classes",ncol=n//10+1,fontsize=19)
    ax.add_artist(legend)
    plt.tight_layout()
    fig.savefig(f"./figures/{name}")

