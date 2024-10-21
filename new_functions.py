import sympy as sp
import math
from scipy import linalg
from scipy.optimize import root_scalar, minimize_scalar
#IMPORTING THE BASIC INFORMATIONS ABOUT THE SYSTEM
from system_setup import *


#----------------------------------DYNAMICS------------------------------------

def create_rho_list():
    rho_list = []
    for i in range(d):
        for j in range(d):
            globals()['rho_'+str(i+1)+str(j+1)] = sp.Symbol(\
            'rho_'+str(i+1)+str(j+1))
            rho_list.append(globals()['rho_'+str(i+1)+str(j+1)])
    return rho_list

def create_rho_matrix():
    rho_matrix = np.empty((d, d), dtype = 'object')
    for i in range(d):
        for j in range(d):
            globals()['rho_'+str(i+1)+str(j+1)] = \
            sp.Symbol('rho_'+str(i+1)+str(j+1))
            rho_matrix[i,j] = globals()['rho_'+str(i+1)+str(j+1)]
    return np.matrix(rho_matrix)

#Function to get the Lindblad term
def Lind():
    L = []
    for k in range(N):
        L.append(np.sqrt(2*B)*sigmas_minus[k]) #LB1, LB2, LB3, ...
    for k in range(N):
        L.append(np.sqrt(2*A)*sigmas_plus[k]) #LA1, LA2, LA3, ...
    return L

def Lindblad_term(Ls):
    rhos = create_rho_matrix()
    Lind_rho = np.zeros((d,d))
    for L in Ls:
        Lind_rho = Lind_rho + L@rhos@L.conjugate().T - 0.5*(L.conjugate().T@L@rhos + rhos@L.conjugate().T@L)
    return Lind_rho

Lindblad = Lindblad_term(Lind())

#GENERALIZED CORRECTION TERM
def correction_term():
    rhos = create_rho_matrix()
    def c(i,j,rho):
        corr_rhoij = np.zeros((d,d))
        corr_rhoij = alfa*B*(2*sigmas_minus[i]@rhos@sigmas_plus[j] - (sigmas_plus[j]@sigmas_minus[i]@rhos + rhos@sigmas_plus[j]@sigmas_minus[i]))\
                   + alfa*A*(2*sigmas_plus[i]@rhos@sigmas_minus[j] - (sigmas_minus[j]@sigmas_plus[i]@rhos + rhos@sigmas_minus[j]@sigmas_plus[i]))
        return corr_rhoij
    corr_rho = np.zeros((d,d))
    for m in range(N):
        for n in range(N):
            if m != n:
                if m == 0 and n == 1: #first contributing term of the summation
                    corr_rho = c(m,n,rhos) #in this way I have a symbolic matrix
                else:
                    corr_rho += c(m,n,rhos)
    return corr_rho


corr = correction_term()

def OBE_matrix(Master_matrix): #the argument is Lindblad + correction
    levels = Master_matrix.shape[0]
    rho_vector = create_rho_list()
    coeff_matrix = np.zeros((d**2, d**2), dtype = 'complex')
    count = 0
    for i in range(levels):
        for j in range(levels):
            entry = Master_matrix[i,j]
            expanded = sp.expand(entry)
            for n,r in enumerate(rho_vector):
                coeff_matrix[count, n] = complex(expanded.coeff(r))
            count += 1
    return coeff_matrix

Lrho = Lindblad + corr

#LIOUVILLIAN OPERATOR
L = OBE_matrix(Lrho) #I factor out the matrix elements of the Liouvillian, 
#possibile because there are no quadratic terms in the density matrix entries

def time_evolve(operator, t, p_0): #evolve vec(rho)
    exponent = operator*t
    term = linalg.expm(exponent) 
    return np.matmul(term, p_0)

#------------------------------------------------------------------------------


#--------------------------THERMODYANMICS STUFF--------------------------------

'''Function that gives a thermal state of N qubits with (real) coherences in 
their local energy basis.'''
def thermal_state(beta_S, rs):
    
    local = linalg.expm(-beta_S*omega0/2*sigma_3)
    part_func = np.trace(local)
    local /= np.trace(local)
    
    if np.any(rs > part_func):
        raise ValueError('Not a proper quantum state: change the value of the coherences.') 
        
    if len(rs) == 1: #identical qubits
        chi = np.array([[0,rs[0]],[rs[0],0]])
        local += chi
        rho = local
        for n in range(1,N):
            rho = np.kron(local,rho)
    
    elif len(rs) == N:
        chi = np.array([[0,rs[0]],[rs[0],0]])
        rho = local + chi
        
        for k in range(1,len(rs)):
            chi_k = np.array([[0,rs[k]],[rs[k],0]])
            local_k = local + chi_K
            rho = np.kron(rho,local_k)
    else: 
        raise ValueError('Number of coherences must be equal to N or to 1.')
    return rho

''' Two-point measurement protocol, returns the probability distribution
given the initial datum and the time of the final measurement.'''
def TPM(rho, final_time):
    initial_rho = np.copy(rho)
    rhot = []
    
    '''Time propagations.'''
    for P in Peig:
        rho0 = np.reshape(P@initial_rho@P, (d**2,1))
        rhot.append(np.reshape(time_evolve(L, final_time, rho0), (d,d)))
    
    #Computing the TPM distribution 
    ptpm = [[np.real(np.trace(rho@P)) for P in Peig] for rho in rhot]
        
    return np.array(ptpm)


''' End-point measurement protocol, returns the probability distribution
given the initial datum and the time of the final measurement. '''
def EPM(rho, final_time): 
   initial_rho = np.copy(rho)
   
   '''Traceless operator that encodes initial coherences in the energy basis.'''
   chi = initial_rho
   for P in Peig:
       chi = chi - P@initial_rho@P
       
   '''Time propagation of the initial datum.'''
   rho0 = np.reshape(initial_rho, (d**2,1))
   rhot = np.reshape(time_evolve(L, final_time, rho0), (d,d))
   
   pepm = [[np.real(np.trace(rhot@P))*np.real(np.trace(initial_rho@Q)) for P in Peig] for Q in Peig]
   
   '''Computing the coherences' contribution to the energy fluctuations '''
   
   '''Time propagation of the traceless part of the initial datum.'''
   rho0 = np.reshape(chi, (d**2,1))
   chit = np.reshape(time_evolve(L, final_time, rho0), (d,d))
   
   coh = 0
   
   for i in range(d):
       coh = coh + energy_levels[i]*np.real(np.trace(Peig[i]@chit))
   
   return [pepm, coh] 

'''Evaluating energy fluctuations starting from distributions'''
def average_heat(pdf):
    Q = 0
    for i in range(d):
        for j in range(d):
                Q = Q + pdf[i,j]*(energy_levels[j]-energy_levels[i]) 
    return Q

'''Variance of the heat fluctuations'''
def var_heat(pdf):
    aveQ2 = average_heat(pdf, energy_levels)**2
    x = 0
    for i in range(d):
        for j in range(d):
                x = x + pdf[i,j]*(energy_levels[j]-energy_levels[i])**2
    return (x - aveQ2)

'''Average of two-times observables'''
def average(pdf, obs):
    ave = 0
    for i in range(d):
        for j in range(d):
                ave = ave + pdf[i,j]*obs[i,j] 
    return ave     

'''Entropy production via log ratio of the transition probabilities'''
def entropy_production(pdf):
    log_ratio = np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            if pdf[i,j] != 0 and pdf[j,i] != 0:
                log_ratio[i,j] = np.log(pdf[i,j]/pdf[j,i])
            else: log_ratio[i,j] = 0
    return average(pdf, log_ratio)

#------------------------------------------------------------------------------

#------------------------------ADVANCED ANALYSIS-------------------------------
    
'''Average entropy production, SNR of the current, standard lower bound of the 
TUR, Landi's lower bound of the TUR.'''
def TUR(pdf, ave, var):
    
    def landi(x):
        
        def h(y):
            return y*np.tanh(y)
        
        '''inverse of x*tanh(x)'''
        def g(z, x0=1.):
            #Find t such that h(t) = z
            root_result = root_scalar(lambda t: h(t)-z, bracket=[0,1000],method='brentq')
            return root_result.root
        
        return np.sinh(g(x/2))**2
    
    
    sigma = entropy_production(pdf)
    SNR = ave**2/var
    std_upper_bound = sigma/2
 
    return sigma, SNR, std_upper_bound, landi(sigma)

'''Information content of the system variation measured with the VON NEUMAN 
ENTROPY. Notice that we are interested in S_IN - S_FIN so the notation for the 
Delta is different here.
'''
def info_var(initial_rho, final_time): 
    S_in = -np.trace(initial_rho@linalg.logm(initial_rho))
    '''Time propagations.'''
    
    rho0 = np.reshape(initial_rho, (d**2,1))
    final_rho = np.reshape(time_evolve(L, final_time, rho0), (d,d))
    
    S_fin = -np.trace(final_rho@linalg.logm(final_rho))
    
    return S_in-S_fin

'''Improved Landauer's principle for bath_dim -> inf (cfr. Reeb and Wolf)'''
def imp_land_principle(dS, bath_dim):
    def f(r):
        return r*(1-r)*np.log((bath_dim-1)*(1-r)/r)**2
    
    def neg_f(r):
        return -f(r)
    
    min_problem = minimize_scalar(neg_f,bounds=(0,0.5), method='bounded')
    
    rmax = min_problem.x
    
    n = f(rmax)
    
    if dS > 0:
        return dS + dS**2/n
    if dS < 0:
        return n - np.sqrt(n**2-2*n*dS)
    if dS == 0:
        return 0
#------------------------------------------------------------------------------

#------------------------STUFF THAT ONLY WORKS IN N=2--------------------------

if N==2:

    '''EPM protocol on the common basis of H and the GGE.'''
    def EPM2(initial_rho, final_time): 
       
       energy_levels = np.array([1.,0,0,-1])
       common = np.array([[1,0,0,0],[0,1/np.sqrt(2),1/np.sqrt(2),0],[0,-1/np.sqrt(2),1/np.sqrt(2),0],[0,0,0,1]])
       Pcommon = np.array([projector(v) for v in common])
       
       '''Traceless operator that encodes initial coherences in the energy basis.'''
       chi = initial_rho
       for P in Peig:
           chi = chi - P@initial_rho@P
           
       '''Time propagation of the initial datum.'''
       rho0 = np.reshape(initial_rho, (d**2,1))
       rhot = np.reshape(time_evolve(L, final_time, rho0), (d,d))
       
       pepm = [[np.real(np.trace(rhot@P))*np.real(np.trace(initial_rho@Q)) for P in Pcommon] for Q in Pcommon]
       
       '''Computing the coherences' contribution to the energy fluctuations '''
       
       '''Time propagation of the traceless part of the initial datum.'''
       rho0 = np.reshape(chi, (d**2,1))
       chit = np.reshape(time_evolve(L, final_time, rho0), (d,d))
       
       coh = 0
       deltasigma = [] 
       
       for i in range(d):
           coh = coh + energy_levels[i]*np.real(np.trace(Peig[i]@chit))
       
       return [pepm, coh] 
    
    '''TPM protocol on the common basis of H and the GGE.'''
    def TPM2(initial_rho, final_time):
    
        energy_levels = np.array([1.,0,0,-1])
        common = np.array([[1,0,0,0],[0,1/np.sqrt(2),1/np.sqrt(2),0],[0,-1/np.sqrt(2),1/np.sqrt(2),0],[0,0,0,1]])
        Pcommon = np.array([projector(v) for v in common])
        
        rhot = []
        
        '''Time propagations.'''
        for P in Pcommon:
            rho0 = np.reshape(P@initial_rho@P, (d**2,1))
            rhot.append(np.reshape(time_evolve(L, final_time, rho0), (d,d)))
        
        #Computing the TPM distribution 
        ptpm = [[np.real(np.trace(rho@P)) for P in Pcommon] for rho in rhot]
            
        return np.array(ptpm)
    
    
    
    '''Theoretical deviations of the EPM-2 from the standard XFT'''
    def gamma(initial_rho, final_time, betaS, r):
        
        def epsilon(initial_rho, final_time):
            rho0 = np.reshape(initial_rho, (d**2,1))
            rho_GGE = np.reshape(time_evolve(L, final_time, rho0), (d,d))
    
            common = np.array([[1,0,0,0],[0,1/np.sqrt(2),1/np.sqrt(2),0],[0,-1/np.sqrt(2),1/np.sqrt(2),0],[0,0,0,1]])
            commonProj = np.array([projector(v) for v in common])
            lambdas = np.array([np.trace(P@rho_GGE@P) for P in commonProj])
    
            rho_th_bath = np.diag([np.exp(-0.5*beta*omega0), np.exp(0.5*beta*omega0)])
            part_func = np.trace(rho_th_bath)
    
            es = np.array([-1./beta*np.log(l*part_func**2) for l in lambdas])
            
            return es
        
        es = epsilon(initial_rho, final_time, beta)
        es = np.real(es)
    
        ZbetaS = np.trace(np.diag([np.exp(-0.5*betaS*omega0), np.exp(0.5*betaS*omega0)]))
        
        gammas = np.empty((d,d))
        
        def delta(n):
            if n == 0 or n == 3:
                return 0
            if n == 1:
                return r**2
            if n == 2:
                return -r**2
        
        for i in range(d):
            for f in range(d):
                gammas[i,f] = beta*((energy_levels[f]-energy_levels[i])-(es[f]-es[i])) + \
                    np.log((1+np.exp(betaS*energy_levels[i])*delta(i)*ZbetaS**2)/ \
                              (1+np.exp(betaS*energy_levels[f])*delta(f)*ZbetaS**2))
        return gammas
    
    
    '''Functions that gives a discordant state with maximally mixed marginals and 
    relative discord value'''
    def disc_state(c1,c2,c3):
        
        cvalues = [c1,c2,c3]
        
        lambdas = 0.25*np.array([1-c1-c2-c3, 1-c1+c2+c3, 1+c1-c2+c3, 1+c1+c2-c3])
        if lambdas.any() > 1 or lambdas.any() < 0:
            raise ValueError('The chose parameters do not give a proper quantum state!') 
        
        c = np.max([abs(c1),abs(c2),abs(c3)])
        
        sigma_1 = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_3 = np.array([[1, 0], [0, -1]], dtype=complex)
        
        pauli_matrices = [sigma_1, sigma_2, sigma_3]
        
        rho = np.eye(4, dtype=complex) 
        for i in range(3):
               rho +=  cvalues[i] * np.kron(pauli_matrices[i], pauli_matrices[i])
               
        rho /= 4
        
        lambdas *= 4
        
        D = 0.25*sum([lambdas[i]*math.log2(lambdas[i]) for i in range(4)]) - \
            (1-c)/2*math.log2(1-c) - (1+c)/2*math.log2(1+c)
        
        return rho, D
    
    def thermal_density_matrix(H, beta):
        rho_thermal = linalg.expm(-beta * H)
        return np.array(rho_thermal / np.trace(rho_thermal), dtype=complex)
    
    '''Function that gives a state with thermal populations and general off-diagonal
    entries'''
    def random_thermal(H,beta,scale):
       
        def random_hermitian_off_diagonal(size, scale):
            """Generate a random Hermitian matrix with zero diagonal elements."""
            random_matrix = (np.random.rand(size, size) + 1j * np.random.rand(size, size)) * scale
            hermitian_matrix = (random_matrix + np.conjugate(random_matrix.T)) / 2
            np.fill_diagonal(hermitian_matrix, 0)  # Keep diagonal entries zero
            return hermitian_matrix
    
        def is_positive_semidefinite(matrix):
            """Check if a matrix is positive semidefinite."""
            return np.all(np.linalg.eigvals(matrix) >= 0)
        
        # Compute the thermal density matrix (thermal diagonal elements)
        rho_thermal = thermal_density_matrix(H, beta)
        
        check = False
        counter = 0
        
        while check == False:
            
            if counter > 5:
                scale /= 2
        
            # Generate random Hermitian matrix for the off-diagonal elements
            random_off_diagonal = random_hermitian_off_diagonal(4, scale)  # Keep scale small to ensure stability
            
            # Combine: Keep thermal diagonal, add random off-diagonal elements
            rho_modified = np.copy(rho_thermal)
            rho_modified += random_off_diagonal  # Modify only the off-diagonal elements
            
            # Check if the matrix is still positive semidefinite
            if is_positive_semidefinite(rho_modified):
                rho_psd = rho_modified  # If already PSD, no need for further corrections
                check = True
            else:
                counter += 1
                
        return rho_psd

