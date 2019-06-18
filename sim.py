import scipy as sp
import numpy as np
import random as rnd
from scipy.integrate import odeint  
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from math import sin,cos,pi, floor, log10
from numba import jit
from multiprocessing import Pool, Value
from dataIO import addResultsToFile, writeResultsToFile
from scipy.fftpack import fft, fftshift,ifft,ifftshift
import os, sys


Ic  = 10**-6
phi0= 6.62607004*10**-34/(2*1.60217662*10**-19)
nPoints = 10**6
maxReducedTime=10

Rs = np.logspace(-3,6,100)

Cs = np.logspace(-9,-3,100)
ws = np.logspace(0,15,200)
forceDCs = np.logspace(-8,-2,50)
forceACs = np.logspace(-8,-2,50)
deltas0= np.linspace(0,pi,15)
qs0 = [0]



def genRandomParamSet():
    return [rnd.choice(param) for param in paramPool]


def roundToDecimalPlace(x,p):
    return 10**-p*round(10**p*x)

def decimalPlaces(x):
    return -floor(log10(x))


@jit(nopython=True)
def simParallel(s,t,a,b,c,d,e):
    x= s[0]
    xl = s[1]   

    return [xl,a+b*cos(c*t)-xl-d*x-e*sin(x)]


@jit(nopython=True)
def simSeries(s,t,a,b,c,d,e):
    x= s[0]
    xl = s[1]   
    return [xl,(a*cos(b*t)-(cos(x)-c*sin(x)*xl)*xl-d*sin(x))/(1+c*cos(x))]


def simulate(R,L,C,forceDC,forceAC,w,delta0,q0,simFunction):
    t = np.linspace(0,maxReducedTime,nPoints)

    if simFunction == "parallel":
        s0 = [delta0,q0/C*2*pi/phi0]

        a = 2*pi*R**2*C*forceDC/phi0
        b = 2*pi*R**2*C*forceAC/phi0
        c = R*C*w
        d = R**2*C/L
        e = 2*pi*R**2*C*Ic/phi0
        f = simParallel

    elif simFunction == "series":
        s0 = [delta0,phi0/(2*pi*R*Ic)*(forceDC-R*Ic*sin(delta0)-q0/C)/(phi0/(2*pi)+L*Ic*cos(delta0))]        
        a = w*forceAC*phi0/(2*pi*Ic**2*R**2)
        b = phi0/(2*pi*Ic*R)*w
        c = 2*pi*L*Ic/phi0
        d = phi0/(2*pi*Ic*R**2*C)
        e = 0
        f = simSeries

    sol,info = odeint(f,s0,t,full_output=1,args=(a,b,c,d,e),mxstep=int(2*10**9))
    
    return (sol[:,0],info["message"] == "Integration successful.")


def getFromCommandLine():
    return sys.argv[1], int(sys.argv[2]), int(sys.argv[3]),sys.argv[4],int(sys.argv[5])

    

if __name__=="__main__":
    simFunction, n, savePeriod,filename, nPools = getFromCommandLine()

    if simFunction == "parallel":
        Ls = np.logspace(-3,1,100)
    else:
        Ls = np.logspace(-15,-10,100)

    paramPool = [Rs,Ls,Cs,forceDCs,forceACs,ws,deltas0,qs0]
    ps = np.array([genRandomParamSet() for i in range(0,n)])
    t = np.linspace(0,maxReducedTime,nPoints)
    deltas = np.zeros((savePeriod,nPoints))
    psUsed = np.zeros((savePeriod,ps.shape[1]))

    def simulate_vector(p):
        return p, simulate(*p,simFunction)
       
    p = Pool(nPools)
    sims = p.imap_unordered(simulate_vector,ps)
    print("Starting calculations")
    successCounter  = 0
    for j,(pUsed,(delta,success)) in enumerate(sims):
        print(f"{j+1}/{n}, success: {success}")
        psUsed[j%savePeriod,:] = pUsed
        deltas[j%savePeriod,:] = delta
        successCounter += success
        if (j+1)%savePeriod == 0:
            k = j//savePeriod
            print("\n\n Saving Results \n")
            if filename in os.listdir():
                addResultsToFile(psUsed,deltas,filename)
            else:
                writeResultsToFile(t,psUsed,deltas,filename)

    print(f"Success rate: {successCounter/n}")    

    p.close()
    p.join()
