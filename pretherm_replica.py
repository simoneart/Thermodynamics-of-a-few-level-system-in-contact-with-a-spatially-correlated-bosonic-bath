import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from functions import *
from pretherm_main import *

'''
One at a time.
'''
P = False
SS = True
obs = False

'''
Here I want to replicate the figures of merit of the paper
'''

beta = 2*np.arctanh(0.8)/omega0

'''
PURITY CHECK
'''
if P == True:
    #initial condition for Fig. (1d)
    rho = np.kron(P0,P0)
    Mab = M(I,rho)
    F = Mab[0,0] + Mab[1,1] + Mab[2,2]
    
    initial_rho = numpy.reshape(rho, (levels**2,1))
    
    rhot = []
    for t in times:
        rhot.append(numpy.reshape(time_evolve(L, t, initial_rho), (levels,levels)))
    
    P = np.array([np.trace(r@r) for r in rhot])
    
    #theoretical value of the purity of the asymptotic state for alfa < 1 (and symmetric ID)
    Pth = 0.5* (1+np.cosh(2*beta*omega0))/(1+np.cosh(beta*omega0))**2
    P_GGE = (-2-8*F+(5+8*F*(1+2*F))*np.cosh(beta*omega0))/(4+8*np.cosh(beta*omega0))
    #checked for both alfa = 1 and < 1 with rho0 = Id4
    th = np.array([P_GGE for t in times])
    
    plt.figure(4)
    plt.grid()
    plt.plot(times, P)
    #plt.plot(times, th, '-.')
    plt.ylabel(r'$\mathbb{Tr}[\rho^2(t)]$')
    plt.xlabel('t')
    plt.xscale("log")
    plt.title(r'Purity over time - $\rho_0 = MMS$ - $\alpha = $'+str(alfa))
    #plt.savefig('C:\\Users\\simon\\OneDrive\\Desktop\\PhD - QThermo\\Pretherm\\pretherm spatial correlations\\pretherm1_images\\P.pdf', format='pdf')

'''
STEADY STATE CHECK
'''
if SS == True:
    #initial condition for Fig. (1a-c)
    initial_data = np.reshape(Id4, (levels**2,1)) #Id4, np.kron(P0,P0)
    
    gibbs = linalg.expm(-beta*H0s)
    gibbs /= np.real(np.trace(gibbs))
    
    Mab = M(I,Id4) #This depends on the initial data!!
    
    l1 = lagr_mult(Mab[0,0],Mab[1,1],Mab[2,2], beta, omega0)
    
    #second term for the GGE
    exp2 = 0.25*(np.kron(sigmax,sigmax)+np.kron(sigmay,sigmay)+np.kron(sigmaz,sigmaz))
                
    GGE = linalg.expm(-beta*H0s -l1*exp2)
    GGE /= np.real(np.trace(GGE))
    
    rhot = []
    for t in times:
        rhot.append(np.reshape(time_evolve(L, t, initial_data), (levels,levels)))
        
    rhot = np.array(rhot)
        
    zz = np.array([gibbs[0][0] for t in times])
    zo = np.array([gibbs[2][2] for t in times])
    oz = np.array([gibbs[3][3] for t in times])
    oo = np.array([gibbs[1][1] for t in times])
    
    zz2 = np.array([GGE[0][0] for t in times])
    zo2 = np.array([GGE[2][2] for t in times])
    oz2 = np.array([GGE[3][3] for t in times])
    oo2 = np.array([GGE[1][1] for t in times])
    
    fig, axs = plt.subplots(2, 2,figsize=(15, 15))
    axs[0, 0].plot(times, rhot[:,0,0])
    axs[0, 0].grid()
    axs[0, 0].plot(times, oz, '-.', label = r'$\rho^{G}_{00}$')
    axs[0, 0].plot(times, oz2, '-.', label = r'$\rho^{GGE}_{00}$')
    axs[0, 0].set_title(r'$\rho_{00}(t)$')
    axs[0, 0].set_xscale('log')
    axs[0, 0].legend()
    axs[0, 1].plot(times, rhot[:,2,2])
    axs[0, 1].grid()
    axs[0, 1].plot(times, zo, '-.', label = r'$\rho^{G}_{22}$')
    axs[0, 1].plot(times, zo2, '-.', label = r'$\rho^{GGE}_{22}$')
    axs[0, 1].set_title(r'$\rho_{22}(t)$')
    axs[0, 1].set_xscale('log')
    axs[0, 1].legend()
    axs[1, 0].plot(times, rhot[:,3,3])
    axs[1, 0].grid()
    axs[1, 0].plot(times, zz, '-.', label = r'$\rho^{G}_{33}$')
    axs[1, 0].plot(times, zz2, '-.', label = r'$\rho^{GGE}_{33}$')
    axs[1, 0].set_title(r'$\rho_{33}(t)$')
    axs[1, 0].set_xscale('log')
    axs[1, 0].legend()
    axs[1, 1].plot(times, rhot[:,1,1])
    axs[1, 1].grid()
    axs[1, 1].plot(times, oo, '-.', label = r'$\rho^{G}_{11}$')
    axs[1, 1].plot(times, oo2, '-.', label = r'$\rho^{GGE}_{11}$')
    axs[1, 1].set_title(r'$\rho_{11}(t)$')
    axs[1, 1].set_xscale('log')
    axs[1, 1].legend()
    plt.savefig('C:\\Users\\simon\\OneDrive\\Desktop\\PhD - QThermo\\Pretherm\\pretherm spatial correlations\\pretherm1_images\\SS_test.pdf', format='pdf')

    
    '''
    THE STEADY STATE IS THE CORRECT ONE, BUT THE ENTRIES HAVE A DIFFERENT ORDER
    How could I make the fidelity right then? Not important for now...
    '''

if obs == True:
    alfa_array = np.array([1,0.99999,0.99,0.5])
    #initial condition for Fig. (1a-c)
    initial_data = np.reshape(Id4, (levels**2,1))
    O = np.zeros((len(alfa_array),3,length))
    c = 0
    for a in alfa_array:
        rhot2 = np.zeros((length,levels,levels))
        alfa2 = a
        #perturbed Liouvillian operator
        correction2 = correction_term(alfa2, A, B, sigma_plus, sigma_minus)
        Lrho2 = Lindblad + correction2
        L2 = OBE_matrix(Lrho2) 
        
        #EVOLUTION
        n = 0
        for t in times:
            rhot2[n] = numpy.reshape(time_evolve(L2, t, initial_data), (levels,levels))
            Mag = M(I,rhot2[n])
            mag = m(I,rhot2[n])
            O[c,0,n] = Mag[0,0] + Mag[1,1]
            O[c,1,n] = Mag[2,2]
            O[c,2,n] = mag[2]
            n += 1
            
        c += 1
    
    fig, axs = plt.subplots(3, figsize=(9, 9))
    for i in range(len(alfa_array)):
        axs[0].plot(times, O[i,0,:], label = r'$\alpha$ = '+str(alfa_array[i]))
    axs[0].legend()
    axs[0].grid()
    axs[0].set_title(r'$M_{c}$')
    axs[0].set_xscale('log')
    for i in range(len(alfa_array)):
        axs[1].plot(times, O[i,1,:], label = r'$\alpha$ = '+str(alfa_array[i]))
    axs[1].legend()
    axs[1].grid()
    axs[1].set_title(r'$M_{zz}$')
    axs[1].set_xscale('log')
    for i in range(len(alfa_array)):
        axs[2].plot(times, O[i,2,:], label = r'$\alpha$ = '+str(alfa_array[i]))
    axs[2].legend()
    axs[2].grid()
    axs[2].set_title(r'$M_{z}$')
    axs[2].set_xscale('log')
    plt.savefig('C:\\Users\\simon\\OneDrive\\Desktop\\PhD - QThermo\\Pretherm\\pretherm spatial correlations\\pretherm1_images\\obs_replica.pdf', format='pdf')

