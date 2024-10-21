import numpy as np
from numpy.linalg import eig

I = np.array([[1, 0], [0, 1]], dtype=complex)
sigma_1 = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_3 = np.array([[1, 0], [0, -1]], dtype=complex)    
pauli_matrices = [I, sigma_1, sigma_2, sigma_3]

#-----------------------NUMBER OF INDEPENDENT QUBITS---------------------------
N = 2
d = 2**N #dimension of the system
#------------------------------------------------------------------------------

#Parameters of the bath and of the qubits
#chosing M0 = 0.8 and R1 = 1.0
A,B = 0.1, 0.9 
omega0 = 1. #time is expressed in units of the Larmor frequency
beta = 2*np.arctanh(0.8)/omega0

#-----------------------CORRELATION LENGTH FUNCTION----------------------------
alfa = 0.99999
#------------------------------------------------------------------------------

#|psi><psi|
def projector(eigenvector):
    P = np.zeros((len(eigenvector),len(eigenvector)), dtype=np.complex128)
    for i in range(len(eigenvector)):
        for j in range(len(eigenvector)):
            P[i,j] = eigenvector[i] * eigenvector[j].conjugate()
    return np.array(P) 

#Function that gives the local Hamiltonian of the jth qubit embedded in the dimension of the full Hilbert space
def Hlocal(N,j):
    local = omega0/2*sigma_3
        
    if j == 1:
        right = np.eye(2)
        for n in range(N-2):
            right = np.kron(np.eye(2),right)
        H = np.kron(local, right)
          
    if j == N:
        left = np.eye(2)
        for n in range(N-2):
            left = np.kron(np.eye(2),left)
        H = np.kron(left, local)
        
    if j > 1 and j < N:
        left = np.eye(2)
        right = np.eye(2)
        for n in range(j-2):
            left = np.kron(np.eye(2),left)
        for n in range(N-j-1):
            right = np.kron(np.eye(2),right)
        H = np.kron(left,np.kron(local,right))
        
    return H

#Creating the Hamilotonian
H0s = np.zeros((d,d), dtype=complex)
for k in range(N):
    H0s += Hlocal(N,k+1)

energy_levels, eig_states = eig(H0s)
Peig = [projector(eig_states[:,i]) for i in range(len(energy_levels))]


#Functions that gives the list of the local ladder operators
#ladders[i] gives the ladder operator on site i+1
def ladders_plus(N):
    sigma_plus = np.array([[0,1],[0,0]]) 
    #sigma_plus = np.array([[0,0],[1,0]])
    ladders = np.empty((N,d,d))
    
    for j in range(1,N+1):
        if j == 1:
            right = np.eye(2)
            for n in range(N-2):
                right = np.kron(np.eye(2),right)
            ladders[j-1] = np.kron(sigma_plus, right)
              
        if j == N:
            left = np.eye(2)
            for n in range(N-2):
                left = np.kron(np.eye(2),left)
            ladders[j-1] = np.kron(left, sigma_plus)
            
        if j > 1 and j < N:
            left = np.eye(2)
            right = np.eye(2)
            for n in range(j-2):
                left = np.kron(np.eye(2),left)
            for n in range(N-j-1):
                right = np.kron(np.eye(2),right)
            ladders[j-1] = np.kron(left,np.kron(sigma_plus,right))
        
    return ladders

sigmas_plus = ladders_plus(N)

def ladders_minus(N):
    sigma_minus = np.array([[0,0],[1,0]]) 
    #sigma_minus = np.array([[0,1],[0,0]])
    ladders = np.empty((N,d,d))
    
    for j in range(1,N+1):
        if j == 1:
            right = np.eye(2)
            for n in range(N-2):
                right = np.kron(np.eye(2),right)
            ladders[j-1] = np.kron(sigma_minus, right)
              
        if j == N:
            left = np.eye(2)
            for n in range(N-2):
                left = np.kron(np.eye(2),left)
            ladders[j-1] = np.kron(left, sigma_minus)
            
        if j > 1 and j < N:
            left = np.eye(2)
            right = np.eye(2)
            for n in range(j-2):
                left = np.kron(np.eye(2),left)
            for n in range(N-j-1):
                right = np.kron(np.eye(2),right)
            ladders[j-1] = np.kron(left,np.kron(sigma_minus,right))
        
    return ladders

sigmas_minus = ladders_minus(N)

#---------------CONSERVED QUANTITIES AND LAGRANGE MULTIPLIER-------------------
#---------------WRITTEN FOR THE N=2 CASE---------------------------------------

def m(rho):
    Iset = [0.5*sigma_1, 0.5*sigma_2, 0.5*sigma_3, np.eye(2)]
    Ma = []
    for i in Iset:
        Ma.append(np.real(np.trace(rho@(np.kron(np.eye(2),i) + np.kron(i,np.eye(2))))))
    return Ma

def M(rho):
    Iset = [0.5*sigma_1, 0.5*sigma_2, 0.5*sigma_3, np.eye(2)]
    Mab = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            if i == j:
                Mab[i,j] = np.real(np.trace(rho@np.kron(Iset[i],Iset[j])))
            else:
                Mab[i,j] = np.real(np.trace(rho@(np.kron(Iset[i],Iset[j]) + np.kron(Iset[j],Iset[i]))))
            
    return Mab

def lagr_mult(Mxx,Myy,Mzz,beta,omega0):
    F = Mxx + Myy + Mzz
    l = np.log((1-4*F)/(3+4*F)*(1+2*np.cosh(beta*omega0)))
    return l
