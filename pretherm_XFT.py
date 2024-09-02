from pretherm_main import *
from mpl_toolkits.mplot3d import Axes3D


#INITIAL DATUM: THERMAL + COHERENCES (separable)
beta_S = 1.5*beta 
#the system is colder than the bath (larger \beta)
Delta_beta =  beta_S - beta

rho_th = np.diag([np.exp(-0.5*beta_S*omega0), np.exp(0.5*beta_S*omega0)])
part_func = np.trace(rho_th)
rho_th /= part_func

final_t = 50

#I will use two identical quibits
r_array = np.linspace(0, 1./part_func, 50)
theta_array =  np.linspace(0, 2*np.pi, 50)

XFT_std = np.zeros((len(r_array),len(theta_array)))
XFT_std2 = np.zeros((len(r_array),len(theta_array)))
XFT_epm = np.zeros((len(r_array),len(theta_array)))
XFT_tpm = np.zeros((len(r_array),len(theta_array)))

for n in range(len(r_array)):
    for m in range(len(theta_array)):

        r = r_array[n]
        theta = theta_array[m]
        
        a = r*np.exp(1j*theta)
        chi = np.array([[0,a],[a.conjugate(),0]])
        rho1 = rho_th + chi
        
        rho0 = np.kron(rho1, rho1)
        
        e1, e2 = eig(rho0)
        for k in range(4):
            if abs(e1[k])<1e-14:
                e1[k] = 0
            if e1[k] < 0:
                print(e1)
                raise ValueError('The given initial data is not a proper Quantum state!')
        
        #----------------------------------------------------------------------
        '''EPM'''
        epm = EPM(rho0, final_t, H0s, L)
        Pepm = np.array(epm[0])
        log_ratio = np.empty((4,4))
        for i in range(4):
            for j in range(4):
                if Pepm[i,j] != 0 and Pepm[j,i] != 0:
                    log_ratio[i,j] = np.log(Pepm[i,j]/Pepm[j,i])
                else: log_ratio[i,j] = 0
                
        XFT_epm[n,m] = average(Pepm, log_ratio)
        
        aveQepm = average_heat(Pepm, energy_levels)
        XFT_std[n,m] = Delta_beta*aveQepm
        #----------------------------------------------------------------------
        '''TPM'''
        Ptpm = TPM(rho0, final_t, H0s, L)
        log_ratio2 = np.empty((4,4))
        for i in range(4):
            for j in range(4):
                if Ptpm[i,j] != 0 and Ptpm[j,i] != 0:
                    log_ratio2[i,j] = np.log(Ptpm[i,j]/Ptpm[j,i])
                else: log_ratio2[i,j] = 0
                
        XFT_tpm[n,m] = average(Ptpm, log_ratio2)
        
        aveQtpm = average_heat(Ptpm, energy_levels)
        XFT_std2[n,m] = Delta_beta*aveQtpm
        #----------------------------------------------------------------------
        
        
        
                        
dev = XFT_epm - XFT_std
dev2 = XFT_tpm - XFT_std2

# Create a meshgrid
x, y = np.meshgrid(theta_array,r_array)

# Create a figure and a 3D axis
fig1 = plt.figure(figsize=(5, 5))
ax = fig1.add_subplot(111, projection='3d')
ax.set_title(r'Deviation from the standard XFT - EPM')
# Plot the surface
ax.plot_surface(x, y, dev, cmap='viridis')
# Add labels
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$r$')
ax.set_zlabel(r'$\Gamma$')
plt.savefig('C:\\Users\\simon\\Desktop\\PhD - QThermo\\Pretherm\\pretherm spatial correlations\\pretherm1_images\\dev_epm.pdf', format='pdf')

# Create a figure and a 3D axis
fig2 = plt.figure(figsize=(5, 5))
ax = fig2.add_subplot(111, projection='3d')
ax.set_title(r'Deviation from the standard XFT - TPM')
# Plot the surface
ax.plot_surface(x, y, dev2, cmap='viridis')
# Add labels
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$r$')
ax.set_zlabel(r'$\Gamma$')
plt.savefig('C:\\Users\\simon\\Desktop\\PhD - QThermo\\Pretherm\\pretherm spatial correlations\\pretherm1_images\\dev_tpm.pdf', format='pdf')




plt.figure(3)
plt.grid()
plt.plot(r_array, XFT_std2[:,0] ,label = 'TPM')
plt.plot(r_array, XFT_std[:,0] ,label = 'EPM')
plt.ylabel(r'$\Sigma$')
plt.xlabel('r')
plt.title(r'Standard XFT - Fixed $\theta = $'+str(theta_array[0]))
plt.legend()
plt.savefig('C:\\Users\\simon\\Desktop\\PhD - QThermo\\Pretherm\\pretherm spatial correlations\\pretherm1_images\\std_XFTs.pdf', format='pdf')

# plt.figure(2)
# plt.grid()
# plt.plot(r_array, dev2[:,0] ,label = 'TPM')
# plt.plot(r_array, dev[:,0] ,label = 'EPM')
# plt.ylabel(r'$\Gamma$')
# plt.xlabel('r')
# plt.title(r'Deviation from the standard XFT - Fixed $\theta = $'+str(theta_array[0]))
# plt.legend()
# plt.savefig('C:\\Users\\simon\\Desktop\\PhD - QThermo\\Pretherm\\pretherm spatial correlations\\pretherm1_images\\devs.pdf', format='pdf')


#------------------------------------------------------------------------------
# Z = np.kron(sigmaz,sigmaz)
# z_levels, z_states = eig(Z)

# theoretical prediction
        
# aveZepm = average_heat(Pepm,z_levels)
# RK: the function is called average_heat but it computes the average of any fluctuations
# of a quantity over the given two-points distribution...
# RK: z_levels and energy_levels are ordered in the same way

# F = 0.25*np.tanh(beta_S*omega0/2)**2 + \
#     (a1.real*a2.real + a1.imag*a2.imag)/ \
#     (np.exp(0.5*beta_S*omega0)+np.exp(-0.5*beta_S*omega0))**2
# l1 = np.log((1-4*F)/(3+4*F)*(1+2*np.cosh(beta*omega0))) 

# ent_th = Delta_beta*aveQepm - 0.25*l1*aveZepm

