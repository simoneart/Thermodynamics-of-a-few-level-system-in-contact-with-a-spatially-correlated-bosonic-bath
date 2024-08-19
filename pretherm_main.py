import numpy as np
from functions import *
from numpy.linalg import eig
import matplotlib.pyplot as plt


alfa = 0.99999


d = 4 

#hbar = 1

#Hamiltonian of two independet spins 
def Hsys(omega):
    return omega*np.diagflat([1,0,0,-1])


#parameters of the bath
#chosing M0 = 0.8 and R1 = 1.0
A,B = 0.1, 0.9 #(checked)
omega0 = 1. #time is expressed in units of the Larmor frequency
H0s = Hsys(omega0)
energy_levels, eig_states = eig(H0s)
Peig = [projector(eig_states[:,i]) for i in range(len(energy_levels))]
levels = len(energy_levels)

#ladder operators of the two-components Hilbert space
sigma_plus = [np.kron(np.array([[0,0],[1,0]]),np.eye(2)), np.kron(np.eye(2),np.array([[0,0],[1,0]]))]
sigma_minus = [np.kron(np.array([[0,1],[0,0]]),np.eye(2)), np.kron(np.eye(2),np.array([[0,1],[0,0]]))]


#single systems Pauli matrix
sigmax = np.array([[0,1],[1,0]])
sigmay = np.array([[0,-1j],[1j,0]])
sigmaz = np.array([[1,0],[0,-1]])

I = [0.5*sigmax, 0.5*sigmay, 0.5*sigmaz, np.eye(2)]

sigma_vec_1 = [np.kron(I[k],np.eye(2)) for k in range(3)]
sigma_vec_2 = [np.kron(np.eye(2),I[k]) for k in range(3)]

zero = np.array([1,0])
one = np.array([0,1])

P0 = projector(zero)
P1 = projector(one)

#INITIAL STATE

#MMS + COHERENCES
# Id4 = 1./d * np.eye(d) #np.kron(rho01,rho02)
# chi1 = np.array([[0,0.2],[0.2,0]])
# chi2 = np.array([[0,0.3],[0.3,0]])
# chi = np.kron(chi1,chi2)
# rho0 = Id4 + chi

#PURE STATE, BLOCH REPRESENTATION (separable)
# theta1 = np.pi/2
# rho_s1 = 0.5*np.array([[1+np.cos(theta1),np.sin(theta1)],[np.sin(theta1),1-np.cos(theta1)]])
# theta2 = np.pi/4
# rho_s2 = 0.5*np.array([[1+np.cos(theta2),np.sin(theta2)],[np.sin(theta2),1-np.cos(theta2)]])
# rho0 = np.kron(rho_s1,rho_s2)


#THERMAL STATE + COHERENCES

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
L = OBE_matrix(Lrho) #I factor out the matrix elements of the Liouvillian, possibile because there are no quadratic terms in the density matrix entries
# array1, array2 = eig(L)

# eigenvalues = []
# for e in array1:
#     eigenvalues.append(np.real(e))
    
# eigenvalues.sort(reverse = True)

# print("Eigenvalues of the Liouvillian operator: ", eigenvalues)


#Time step of th evolution and number of total steps
dt = 0.01
num_steps = 10000000
times = [(n+1)*dt for n in range(0,num_steps//1000)] 
times = times + [(n+1)*dt for n in range(num_steps//1000,num_steps,100)]
length = len(times)

#--------------------------------ENERGY FLUCTUATIONS---------------------------
if __name__ == "__main__":

    Ptpm = np.array([TPM(rho0, t, H0s, L) for t in times])
    Qtpm = np.array([average_heat(Ptpm[k],energy_levels) for k in range(length)])
    
    Pepm = []
    Pend = []
    coh_ef = []
    deltasigma = []
    
    for t in times:
        x = EPM(rho0, t, H0s, L)
        Pepm.append(np.array(x[0]))
        coh_ef.append(np.array(x[1]))
        deltasigma.append(np.array(x[2]))
    
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
    #plt.savefig('C:\\Users\\simon\\OneDrive\\Desktop\\PhD - QThermo\\Pretherm\\pretherm spatial correlations\\pretherm1_images\\fluct1.pdf', format='pdf')
    
    
    deltasigma_ave = []
    
    for n in range(length):
        sum_elem = 0
        for i in range(levels):
            for j in range(levels):
                sum_elem += deltasigma[n][i]*Pepm[n][j][i]
        deltasigma_ave.append(sum_elem)
    
    
    plt.figure(2)
    plt.grid()
    plt.plot(times, deltasigma_ave)
    plt.ylabel(r'$<\Delta \Sigma>_{EPM}$')
    plt.xlabel('Final times')
    plt.xscale("log")
    plt.title(r'Coherences-affected entropy production - $\alpha = $'+str(alfa))
    #plt.savefig('C:\\Users\\simon\\OneDrive\\Desktop\\PhD - QThermo\\Pretherm\\pretherm spatial correlations\\pretherm1_images\\Pi1.pdf', format='pdf')


