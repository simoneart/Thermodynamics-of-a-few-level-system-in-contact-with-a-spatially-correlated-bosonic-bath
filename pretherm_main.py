import numpy as np
from functions import *
from numpy.linalg import eig
import matplotlib.pyplot as plt

from scipy import linalg


alfa = 0.99999


d = 4 

#hbar = 1

#Hamiltonian of two independet spins 
# def Hsys(omega):
#     return omega*np.diagflat([1,0,0,-1])

#parameters of the bath
#chosing M0 = 0.8 and R1 = 1.0
A,B = 0.1, 0.9 
omega0 = 1. #time is expressed in units of the Larmor frequency
beta = 2*np.arctanh(0.8)/omega0

#IMPLEMENTING THE QME

#ladder operators of the two-components Hilbert space
sigma_minus = [np.kron(np.array([[0,0],[1,0]]),np.eye(2)), np.kron(np.eye(2),np.array([[0,0],[1,0]]))]
sigma_plus = [np.kron(np.array([[0,1],[0,0]]),np.eye(2)), np.kron(np.eye(2),np.array([[0,1],[0,0]]))]

#single systems Pauli matrix
sigmax = np.array([[0,1],[1,0]])
sigmay = np.array([[0,-1j],[1j,0]])
sigmaz = np.array([[1,0],[0,-1]])

def Hsys(omega):
    return 0.5*omega*(np.kron(sigmaz,np.eye(2)) + np.kron(np.eye(2), sigmaz))

#system's Hamiltonian
H0s = Hsys(omega0)
energy_levels, eig_states = eig(H0s)
Peig = [projector(eig_states[:,i]) for i in range(len(energy_levels))]
levels = len(energy_levels)

I = [0.5*sigmax, 0.5*sigmay, 0.5*sigmaz, np.eye(2)]

sigma_vec_1 = [np.kron(I[k],np.eye(2)) for k in range(3)]
sigma_vec_2 = [np.kron(np.eye(2),I[k]) for k in range(3)]

zero = np.array([1,0])
one = np.array([0,1])

P0 = projector(zero)
P1 = projector(one)

#Lindblad operators, correction term
LB1 = np.sqrt(2*B)*sigma_minus[0]
LB2 = np.sqrt(2*B)*sigma_minus[1]
LA1 = np.sqrt(2*A)*sigma_plus[0]
LA2 = np.sqrt(2*A)*sigma_plus[1]

L = [LB1,LB2,LA1,LA2]

Lindblad = Lindblad_term(L)
correction = correction_term(alfa, A, B, sigma_plus, sigma_minus)

Lrho = Lindblad + correction

#LIOUVILLIAN OPERATOR
L = OBE_matrix(Lrho) #I factor out the matrix elements of the Liouvillian, 
#possibile because there are no quadratic terms in the density matrix entries

#Time step of th evolution and number of total steps
dt = 0.01
num_steps = 100000000 #1e8
times = [(n+1)*dt for n in range(0,num_steps//10000)] 
times = times + [(n+1)*dt for n in range(num_steps//10000,num_steps//100,100)]
times = times + [(n+1)*dt for n in range(num_steps//100,num_steps,1000)]
length = len(times)

#--------------------------------ENERGY FLUCTUATIONS---------------------------
if __name__ == "__main__":
    
    #INITIAL STATE

    #MMS + COHERENCES
    Id4 = 1./d * np.eye(d) #np.kron(rho01,rho02)
    chi1 = np.array([[0,0.2],[0.2,0]])
    chi2 = np.array([[0,0.3],[0.3,0]])
    chi = np.kron(chi1,chi2)
    rho0 = Id4 + chi

    #PURE STATE, BLOCH REPRESENTATION (separable)
    # theta1 = np.pi/2
    # rho_s1 = 0.5*np.array([[1+np.cos(theta1),np.sin(theta1)],[np.sin(theta1),1-np.cos(theta1)]])
    # theta2 = np.pi/4
    # rho_s2 = 0.5*np.array([[1+np.cos(theta2),np.sin(theta2)],[np.sin(theta2),1-np.cos(theta2)]])
    # rho0 = np.kron(rho_s1,rho_s2)
    
    #THERMAL + COHERENCES (separable)
    # beta_S = 1.5*beta 

    # rho_th = linalg.expm(-beta_S*omega0*0.5*sigmaz)
    # part_fun = np.trace(rho_th)
    # rho_th = rho_th/part_fun
    
    # #r1 < 1/part_func

    # r1 = 0
    # theta1 = np.pi/3
    # a1 = r1*(np.cos(theta1) + np.sin(theta1)*1j) 
    # chi1 = np.array([[0,a1],[a1.conjugate(),0]])
    # rho1 = rho_th + chi1

    # rho0 = np.kron(rho1, rho1)
    
    e1, e2 = eig(rho0)
    for k in range(4):
        if abs(e1[k])<1e-14:
            e1[k] = 0
        if e1[k] < 0:
            print(e1)
            raise ValueError('The given initial data is not a proper Quantum state!')
    
    Ptpm = np.array([TPM(rho0, t, H0s, L) for t in times])
    Qtpm = np.array([average_heat(Ptpm[k],energy_levels) for k in range(length)])
    
    Pepm = []
    coh_ef = []
   
    for t in times:
        x = EPM(rho0, t, H0s, L)
        Pepm.append(np.array(x[0]))
        coh_ef.append(np.array(x[1]))
    
    Qepm = np.array([average_heat(Pepm[k],energy_levels) for k in range(length)])
    
    plt.figure(1)
    plt.grid()
    plt.plot(times, Qtpm ,label = 'TPM')
    plt.plot(times, Qepm ,label = 'EPM')
    plt.plot(times, coh_ef, '--', label = r'$\sum_{i=1,2,3,4}E_{i}Tr[\Phi(\chi)\Pi_{i}]$')
    plt.ylabel(r'$\Delta E$')
    plt.xlabel('Final times')
    plt.xscale("log")
    plt.title(r"Energy fluctuations varying the time of the final measure - $\alpha = $"+str(alfa))
    plt.legend()
    plt.savefig('C:\\Users\\simon\\Desktop\\PhD - QThermo\\Pretherm\\pretherm spatial correlations\\pretherm1_images\\fluct2.pdf', format='pdf')
    
    


