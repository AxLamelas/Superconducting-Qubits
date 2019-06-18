import h5py
import matplotlib.pyplot as plt

def plot(i):
    plt.plot(f["t"],f["deltas"][i,:])
    plt.show()

    

f = h5py.File("teste.h5","r")