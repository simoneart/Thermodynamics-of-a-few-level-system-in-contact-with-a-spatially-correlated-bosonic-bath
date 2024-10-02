import numpy 
import sympy
from scipy import linalg
from scipy.optimize import root_scalar, minimize_scalar
from numpy.linalg import eig

d = 4

#----------------------------------DYNAMICS------------------------------------

def create_rho_list(levels = d):
    rho_list = []
    for i in range(levels):
        for j in range(levels):
            globals()['rho_'+str(i+1)+str(j+1)] = sympy.Symbol(\
            'rho_'+str(i+1)+str(j+1))
            rho_list.append(globals()['rho_'+str(i+1)+str(j+1)])
    return rho_list

def create_rho_matrix(levels = d):
    rho_matrix = numpy.empty((levels, levels), dtype = 'object')
    for i in range(levels):
        for j in range(levels):
            globals()['rho_'+str(i+1)+str(j+1)] = \
            sympy.Symbol('rho_'+str(i+1)+str(j+1))
            rho_matrix[i,j] = globals()['rho_'+str(i+1)+str(j+1)]
    return numpy.matrix(rho_matrix)

def Lindblad_term(Ls):
    rhos = create_rho_matrix(levels = d)
    Lind_rho = numpy.zeros((d,d))
    for L in Ls:
        Lind_rho = Lind_rho + L@rhos@L.conjugate().T - 0.5*(L.conjugate().T@L@rhos + rhos@L.conjugate().T@L)
    return Lind_rho

#the second term containing the cross terms (non-Markovian)
def correction_term(alfa,A,B,sigma_plus,sigma_minus):
    rhos = create_rho_matrix(levels = d)
    corr_rho = numpy.zeros((d,d))
    corr_rho = 2* (alfa*B*(sigma_minus[0]@rhos@sigma_plus[1] - 0.5*(sigma_plus[1]@sigma_minus[0]@rhos + rhos@sigma_plus[1]@sigma_minus[0])\
                   + sigma_minus[1]@rhos@sigma_plus[0] - 0.5*(sigma_plus[0]@sigma_minus[1]@rhos + rhos@sigma_plus[0]@sigma_minus[1]))\
                   + alfa*A*(sigma_plus[0]@rhos@sigma_minus[1] - 0.5*(sigma_minus[1]@sigma_plus[0]@rhos + rhos@sigma_minus[1]@sigma_plus[0])\
                   + sigma_plus[1]@rhos@sigma_minus[0] - 0.5*(sigma_minus[0]@sigma_plus[1]@rhos + rhos@sigma_minus[0]@sigma_plus[1])) )
    return corr_rho

def OBE_matrix(Master_matrix): #the argument is Lindblad + correction
    levels = Master_matrix.shape[0]
    rho_vector = create_rho_list(levels = levels)
    coeff_matrix = numpy.zeros((levels**2, levels**2), dtype = 'complex')
    count = 0
    for i in range(levels):
        for j in range(levels):
            entry = Master_matrix[i,j]
            expanded = sympy.expand(entry)
            for n,r in enumerate(rho_vector):
                coeff_matrix[count, n] = complex(expanded.coeff(r))
            count += 1
    return coeff_matrix

def time_evolve(operator, t, p_0): #evolve vec(rho)
    exponent = operator*t
    term = linalg.expm(exponent) 
    return numpy.matmul(term, p_0)

#------------------------------------------------------------------------------


#---------------CONSERVED QUANTITIES AND LAGRANGE MULTIPLIER-------------------

def m(I,rho):
    Ma = []
    for i in I:
        Ma.append(numpy.real(numpy.trace(rho@(numpy.kron(numpy.eye(2),i) + numpy.kron(i,numpy.eye(2))))))
    return Ma

def M(I, rho):
    Mab = numpy.zeros((4,4))
    for i in range(4):
        for j in range(4):
            if i == j:
                Mab[i,j] = numpy.real(numpy.trace(rho@numpy.kron(I[i],I[j])))
            else:
                Mab[i,j] = numpy.real(numpy.trace(rho@(numpy.kron(I[i],I[j]) + numpy.kron(I[j],I[i]))))
            
    return Mab

def lagr_mult(Mxx,Myy,Mzz,beta,omega0):
    F = Mxx + Myy + Mzz
    l = numpy.log((1-4*F)/(3+4*F)*(1+2*numpy.cosh(beta*omega0)))
    return l

#|psi><psi|
def projector(eigenvector):
    P = numpy.zeros((len(eigenvector),len(eigenvector)), dtype=numpy.complex128)
    for i in range(len(eigenvector)):
        for j in range(len(eigenvector)):
            P[i,j] = eigenvector[i] * eigenvector[j].conjugate()
    return numpy.array(P) 

#------------------------------------------------------------------------------

#--------------------------THERMODYANMICS STUFF--------------------------------

''' Two-point measurement protocol, returns the probability distribution
given: the initial datum, the time of the final measurement
the Hamiltonian and the Liouvillan. '''

def TPM(initial_rho, final_time, H, Liouvillian):

    energy_levels, eig_states = eig(H)
    Peig = [projector(eig_states[:,i]) for i in range(len(energy_levels))]
    
    levels = len(energy_levels)
    
    rhot = []
    
    '''Time propagations.'''
    for P in Peig:
        rho0 = numpy.reshape(P@initial_rho@P, (levels**2,1))
        rhot.append(numpy.reshape(time_evolve(Liouvillian, final_time, rho0), (levels,levels)))
    
    #Computing the TPM distribution 
    ptpm = [[numpy.real(numpy.trace(rho@P)) for P in Peig] for rho in rhot]
        
    return numpy.array(ptpm)

'''Margenau-Hill protocol, returns the quasi distribution
given: the initial datum, the time of the final measurement
the Hamiltonian and the Liouvillan. '''

def MH(initial_rho, final_time, H, Liouvillian): 
    
   energy_levels, eig_states = eig(H)
   Peig = [projector(eig_states[:,i]) for i in range(len(energy_levels))]
   
   levels = len(energy_levels)
   
   '''Time propagations'''
   rhot = []
   for P in Peig:
       rho0 = numpy.reshape(P@initial_rho, (levels**2,1))
       rhot.append(numpy.reshape(time_evolve(Liouvillian, final_time, rho0), (levels,levels)))
   
   q = [[numpy.real(numpy.trace(rho@P)) for P in Peig] for rho in rhot]
   
   return numpy.array(q)

''' End-point measurement protocol, returns the probability distribution
given: the initial datum, the time of the final measurement
the Hamiltonian and the Liouvillan. '''

def EPM(initial_rho, final_time, H, Liouvillian): 
   energy_levels, eig_states = eig(H)
   Peig = [projector(eig_states[:,i]) for i in range(len(energy_levels))]
   
   levels = len(energy_levels)
   
   '''Traceless operator that encodes initial coherences in the energy basis.'''
   chi = initial_rho
   for P in Peig:
       chi = chi - P@initial_rho@P
       
   '''Time propagation of the initial datum.'''
   rho0 = numpy.reshape(initial_rho, (levels**2,1))
   rhot = numpy.reshape(time_evolve(Liouvillian, final_time, rho0), (levels,levels))
   
   pepm = [[numpy.real(numpy.trace(rhot@P))*numpy.real(numpy.trace(initial_rho@Q)) for P in Peig] for Q in Peig]
   
   '''Computing the coherences' contribution to the energy fluctuations '''
   
   '''Time propagation of the traceless part of the initial datum.'''
   rho0 = numpy.reshape(chi, (levels**2,1))
   chit = numpy.reshape(time_evolve(Liouvillian, final_time, rho0), (levels,levels))
   
   coh = 0
   deltasigma = [] 
   
   for i in range(levels):
       coh = coh + energy_levels[i]*numpy.real(numpy.trace(Peig[i]@chit))
   
   return [pepm, coh] 

'''Evaluating energy fluctuations starting from distributions'''
def average_heat(pdf, eigen_energies):
    Q = 0
    for i in range(len(eigen_energies)):
        for j in range(len(eigen_energies)):
                Q = Q + pdf[i,j]*(eigen_energies[j]-eigen_energies[i]) 
    return Q

'''Variance of the heat fluctuations'''
def var_heat(pdf, eigen_energies):
    aveQ2 = average_heat(pdf, eigen_energies)**2
    x = 0
    for i in range(len(eigen_energies)):
        for j in range(len(eigen_energies)):
                x = x + pdf[i,j]*(eigen_energies[j]-eigen_energies[i])**2
    return (x - aveQ2)

'''Average of two-times observables'''
def average(pdf, obs):
    ave = 0
    for i in range(4):
        for j in range(4):
                ave = ave + pdf[i,j]*obs[i,j] 
    return ave     

def index_elements(Q):
    element_indices = {}
    for i, row in enumerate(Q):
        for j, value in enumerate(row):
            if value not in element_indices:
                element_indices[value] = []
            element_indices[value].append([i, j])
    return element_indices


'''Building the histogram'''
def hist_dist(pdf, eigen_energies):
    Q = numpy.zeros((len(eigen_energies),len(eigen_energies)))
    
    for i in range(len(eigen_energies)):
        for j in range(len(eigen_energies)):
                Q[i,j] = (eigen_energies[j]-eigen_energies[i])
                
    ind = index_elements(Q)
    
    newQ = list(ind.keys())
    repInd = list(ind.values())
    
    newp = numpy.zeros(len(newQ))
    
    # for i in range(len(newQ)):
    #     newp[i] = sum([pdf[repInd[i][j][0], repInd[i][j][1]] for j range(len(repInd[i]))])
    for i in range(len(newQ)):
        for j in range(len(repInd[i])):
            newp[i] += pdf[repInd[i][j][0], repInd[i][j][1]]
    
    return numpy.array(newQ), newp

def entropy_production(pdf, eigen_energies):
    
    Q = numpy.zeros((len(eigen_energies),len(eigen_energies)))
    negQ = numpy.zeros((len(eigen_energies),len(eigen_energies)))

    for i in range(len(eigen_energies)):
        for j in range(len(eigen_energies)):
                Q[i,j] = (eigen_energies[j]-eigen_energies[i])
                negQ[i,j] = (eigen_energies[i]-eigen_energies[j])
                
    ind = index_elements(Q)
    newQ = numpy.array(list(ind.keys()))
    repInd = list(ind.values())
    newp = numpy.zeros(len(newQ))

    for i in range(len(newQ)):
        for j in range(len(repInd[i])):
            newp[i] += pdf[repInd[i][j][0], repInd[i][j][1]]
    
    neg_ind = index_elements(negQ)
    newNegQ = numpy.array(list(neg_ind.keys()))
    repNeg_Ind = list(neg_ind.values())
    newNegp = numpy.zeros(len(newNegQ))

    for i in range(len(newNegQ)):
        for j in range(len(repNeg_Ind[i])):
            newNegp[i] += pdf[repNeg_Ind[i][j][1], repNeg_Ind[i][j][0]]
    
    log_ratio = numpy.empty(len(newQ))
    for i in range(len(newNegQ)):
        if newp[i] != 0 and newNegp[i] != 0:
            log_ratio[i] = numpy.log(newp[i]/newNegp[i])
        else:
            log_ratio[i] = 0
    
    sigma = 0
    for i in range(len(newNegQ)):
        sigma += newp[i]*log_ratio[i]
        
    return sigma
        
    
'''Average entropy production, SNR of the current, standard lower bound of the 
TUR, Landi's lower bound of the TUR.'''

def TUR(pdf, ave, var):
    
    def landi(x):
        
        #def csch(z):
        #    return 1./numpy.sinh(z)
        
        def h(y):
            return y*numpy.tanh(y)
        
        '''inverse of x*tanh(x)'''
        def g(z, x0=1.):
            #Find t such that h(t) = z
            root_result = root_scalar(lambda t: h(t)-z, bracket=[0,1000],method='brentq')
            return root_result.root
        
        return numpy.sinh(g(x/2))**2
    
    log_ratio = numpy.empty((4,4))
    for i in range(4):
        for j in range(4):
            if pdf[i,j] != 0 and pdf[j,i] != 0:
                log_ratio[i,j] = numpy.log(pdf[i,j]/pdf[j,i])
            else: log_ratio[i,j] = 0
    sigma = average(pdf, log_ratio)
    SNR = ave**2/var
    std_upper_bound = sigma/2
 
    return sigma, SNR, std_upper_bound, landi(sigma)

'''Information content of the system variation measured with the VON NEUMAN 
ENTROPY. Notice that we are interested in S_IN - S_FIN so the notation for the 
Delta is different here.
'''
def info_var(initial_rho, final_time, Liouvillian): 
    S_in = -numpy.trace(initial_rho@linalg.logm(initial_rho))
    '''Time propagations.'''
    
    rho0 = numpy.reshape(initial_rho, (d**2,1))
    final_rho = numpy.reshape(time_evolve(Liouvillian, final_time, rho0), (d,d))
    
    S_fin = -numpy.trace(final_rho@linalg.logm(final_rho))
    
    return S_in-S_fin

'''Improved Landauer's principle for bath_dim -> inf (cfr. Reeb and Wolf)'''
def imp_land_principle(dS, bath_dim):
    def f(r):
        return r*(1-r)*numpy.log((bath_dim-1)*(1-r)/r)**2
    
    def neg_f(r):
        return -f(r)
    
    min_problem = minimize_scalar(neg_f,bounds=(0,0.5), method='bounded')
    
    rmax = min_problem.x
    
    N = f(rmax)
    
    if dS > 0:
        return dS + dS**2/N
    if dS < 0:
        return N - numpy.sqrt(N**2-2*N*dS)
    if dS == 0:
        return 0
    
'''Fidelity/Loschimdt Echo'''
def F(ro1,ro2):
    a = linalg.sqrtm(ro1)
    b = linalg.sqrtm(a@ro2@a)
    f = numpy.trace(b)**2
    return numpy.real(f)

#----------------THEORETICAL DEVIATION OF THE EPM FROM THE XFT-----------------

def epsilon(initial_rho, final_time, Liouvillian, beta):
    omega0 = 1. #it must be the same as in main.py, but I don't think I will ever change it
    rho0 = numpy.reshape(initial_rho, (d**2,1))
    rho_GGE = numpy.reshape(time_evolve(Liouvillian, final_time, rho0), (d,d))

    common = numpy.array([[1,0,0,0],[0,1/numpy.sqrt(2),1/numpy.sqrt(2),0],[0,-1/numpy.sqrt(2),1/numpy.sqrt(2),0],[0,0,0,1]])
    commonProj = numpy.array([projector(v) for v in common])
    lambdas = numpy.array([numpy.trace(P@rho_GGE@P) for P in commonProj])

    rho_th_bath = numpy.diag([numpy.exp(-0.5*beta*omega0), numpy.exp(0.5*beta*omega0)])
    part_func = numpy.trace(rho_th_bath)

    es = numpy.array([-1./beta*numpy.log(l*part_func**2) for l in lambdas])
    
    return es

def gamma(initial_rho, final_time, Liouvillian, beta, betaS, r, energy_levels):
    
    es = epsilon(initial_rho, final_time, Liouvillian, beta)
    
    omega0 = 1. #it must be the same as in main.py, but I don't think I will ever change it
    ZbetaS = numpy.trace(numpy.diag([numpy.exp(-0.5*betaS*omega0), numpy.exp(0.5*betaS*omega0)]))
    
    gammas = numpy.empty((d,d))
    
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
                numpy.log((1+numpy.exp(betaS*energy_levels[i])*delta(i)*ZbetaS**2)/ \
                          (1+numpy.exp(betaS*energy_levels[f])*delta(f)*ZbetaS**2))
    return gammas

    
