from pretherm_main import *
from mpl_toolkits.mplot3d import Axes3D

#look at the program MHQ.py for the blueprint of other functions
#(it's in few_levels_system)


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
L = OBE_matrix(Lrho) 

angles1 = np.linspace(0, np.pi, 50)
angles2 = np.linspace(0, np.pi, 50)
CAEP = np.zeros((len(angles1),len(angles2)))

'''
CHARACTERIZATION OF THE COHERENCES-AFFECTED ENTROPY PRODUCTION
FOR SEPARABLE INITIAL STATES.
'''

for n in range(len(angles1)):
    for m in range(len(angles2)):
        theta1 = angles1[n]
        rho_s1 = 0.5*np.array([[1+np.cos(theta1),np.sin(theta1)],[np.sin(theta1),1-np.cos(theta1)]])
        
        theta2 = angles2[m]
        rho_s2 = 0.5*np.array([[1+np.cos(theta2),np.sin(theta2)],[np.sin(theta2),1-np.cos(theta2)]])
        
        rho0 = np.kron(rho_s1,rho_s2)
        
        #I now choose a fixed final time, s.t. the system is in the prethermal phase
        final_t = 50
        
        '''EPM'''
        epm = EPM(rho0, final_t, H0s, L)
        Pepm = np.array(epm[0])
        coh_ef = np.array(epm[1])
        deltasigma = np.array(epm[2])
        
        sum_elem = 0
        for i in range(levels):
            for j in range(levels):
                sum_elem += deltasigma[i]*Pepm[j][i]
        
        CAEP[n,m] = sum_elem

# Create a meshgrid
x, y = np.meshgrid(angles1, angles2)

# Create a figure and a 3D axis
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')
ax.set_title(r'Coherences-affected entropy production - $\alpha = $'+str(alfa))
# Plot the surface
ax.plot_surface(x, y, CAEP, cmap='viridis')
# Add labels
ax.set_xlabel(r'$\theta_1$')
ax.set_ylabel(r'$\theta_2$')
ax.set_zlabel(r'$<\Delta \Sigma>_{EPM}$')
#Contours
ax.contour(x, y, CAEP, zdir='x', offset=0, cmap='viridis')
ax.contour(x, y, CAEP, zdir='y', offset=np.pi, cmap='viridis')
#plt.savefig('C:\\Users\\simon\\OneDrive\\Desktop\\PhD - QThermo\\Pretherm\\pretherm spatial correlations\\pretherm1_images\\CAEP.pdf', format='pdf')


'''
(x) Can I get a negative MHQ distribution using entangled initial states? 
    --> No, but look at their properties in terms of the observables, the show
    some interesting thermodynamic behaviour (TPM>EPM or TPM=EPM, cfr main)
    --> psi^{-} shows the most dramatic difference between EPM and TPM
'''

#HISTOGRAMS AT FIXED INITIAL DATUM
rho0 = 0.5*np.array([[0,0,0,0],[0,1,-1,0],[0,-1,1,0],[0,0,0,0]])
#rho0 = 0.5*np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])


'''TPM'''
Ptpm = TPM(rho0, final_t, H0s, L)
Qtpm = average_heat(Ptpm,energy_levels)

'''EPM'''
epm = EPM(rho0, final_t, H0s, L)
Pepm = np.array(epm[0])
coh_ef = np.array(epm[1])
deltasigma = np.array(epm[2])

Qepm = average_heat(Pepm,energy_levels)

deltasigma_ave = []

sum_elem = 0
for i in range(levels):
    for j in range(levels):
        sum_elem += deltasigma[i]*Pepm[j][i]
deltasigma_ave.append(sum_elem)

'''MHQ'''
mhqp = MH(rho0, final_t, H0s, L)
Qmhq = average_heat(mhqp, energy_levels)

'''Histograms'''
x1, y1 = hist_dist(mhqp, energy_levels)
x2, y2 = hist_dist(Ptpm, energy_levels)
x3, y3 = hist_dist(Pepm, energy_levels)

fig, axs = plt.subplots(1, 3,figsize=(9, 5))
axs[0].bar(x2, y2)
axs[0].grid()
axs[0].set_title('TPM')
axs[1].bar(x1, y1)
axs[1].grid()
axs[1].set_title('Margenau-Hill')
axs[2].bar(x3, y3)
axs[2].grid()
axs[2].set_title('EPM')
fig.suptitle(r'Comparison of the distributions')
for ax in axs:
    ax.set(xlabel='Q')
#plt.savefig('C:\\Users\\simon\\OneDrive\\Desktop\\PhD - QThermo\\Pretherm\\pretherm spatial correlations\\pretherm1_images\\histo.pdf', format='pdf')


