import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fmin
from sim import simulate
import matplotlib
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from manualFeatures import nFrequencies



plt.rc('text',usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{siunitx}')
plt.rc('font',size=40)

Ic  = 10**-6
phi0= 6.62607004*10**-34/(2*1.60217662*10**-19)

Rs = [10**-3,1]
Cs = [10**-9,10**-3]


w = 2*np.pi*10**9
forceDC = 10**-6
forceAC = 10**-6
delta0= np.pi/3
q0 = 0
t = np.linspace(0,10,10**6)
counter = 0
freqs = list()
for f in ["series","parallel"]:
    if f == "parallel":
        Ls = [10**-3,10**1]
    else:
        Ls = [10**-15,10**-10]

    for R in Rs:
        for C in Cs:
            for L in Ls:
                delta,successful = simulate(R,L,C,forceDC,forceAC,w,delta0,q0,f)
                i = np.sin(delta)
                if f =="series":
                    freqs.append(nFrequencies(3,phi0/(2*np.pi*Ic*R)*t,i))
                else:
                    freqs.append(nFrequencies(3,R*C*t,i))
                plt.close()
                fig, ax = plt.subplots(figsize=(14,10))
                ax.plot(t,i,'grey')
                ax.set_xlabel(r'$\tau$')
                ax.set_ylabel(r'$I/I_c$')
                ax.set_ylim(-1.05,1.05)
                ax.set_title(f"$R = \\num{{{R:.2e}}}\:\si{{\ohm}},\:L = \\num{{{L:.2e}}}\:\si{{\henry}},$"+"\n"+f"$\:C = \\num{{{C:.2e}}}\:\si{{\\farad}}$",pad=20,multialignment='center')
                plt.tight_layout()
                fig.savefig(f"./figures/manual/{counter}")
                counter +=1

                plt.close()
                fig, ax = plt.subplots(figsize=(14,10))
                ax.plot(t,i,'grey')
                ax.set_xlabel(r'$\tau$')
                ax.set_ylabel(r'$I/I_c$')
                ax.set_ylim(-1.05,1.05)
                ax.set_xlim(6,6.05)
                ax.set_title(f"$R = \\num{{{R:.2e}}}\:\si{{\ohm}},\:L = \\num{{{L:.2e}}}\:\si{{\henry}},$"+"\n"+f"$\:C = \\num{{{C:.2e}}}\:\si{{\\farad}}$",pad=20,multialignment='center')
                plt.tight_layout()
                fig.savefig(f"./figures/manual/{counter}")
                counter +=1
