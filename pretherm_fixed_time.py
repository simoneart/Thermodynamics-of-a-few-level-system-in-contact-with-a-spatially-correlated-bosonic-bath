from pretherm_main import *
from mpl_toolkits.mplot3d import Axes3D

#final time of measurement: ensures the prthermal regime
final_t = 50

'''
Select the initial data:
    Th: thermal initial state with coherences, separable
    Sep: separable, pure states
    Ent: maximally entangled states
'''

state = 'Sep'

init_rhos = []

if state == 'Th':
    beta_S = 1.5*beta 
    #the system is colder than the bath (larger \beta)
    Delta_beta =  beta_S - beta

    rho_th = np.diag([np.exp(-0.5*beta_S*omega0), np.exp(0.5*beta_S*omega0)])
    part_func = np.trace(rho_th)
    rho_th /= part_func

    r_array = np.linspace(0, 1./part_func, 50)
    #we saw that the phase is not relevant
    
    aveQepm = np.empty(len(r_array))
    varQepm = np.empty(len(r_array))
    
    aveQtpm = np.empty(len(r_array))
    varQtpm = np.empty(len(r_array))
    
    for n in range(len(r_array)):
        r = r_array[n]

        chi = np.array([[0,r],[r,0]])
        rho1 = rho_th + chi
        
        rho0 = np.kron(rho1, rho1)
        
        Ptpm = TPM(rho0, final_t, H0s, L)
        epm = EPM(rho0, final_t, H0s, L)
        Pepm = np.array(epm[0])
        
        aveQepm[n] = average_heat(Pepm, energy_levels)
        varQepm[n] = var_heat(Pepm, energy_levels)
        
        aveQtpm[n] = average_heat(Ptpm, energy_levels)
        varQtpm[n] = var_heat(Ptpm, energy_levels)
    
    varQepm = np.sqrt(varQepm)
    varQtpm = np.sqrt(varQtpm)
        
    plt.figure(1)
    plt.grid()
    plt.errorbar(r_array, aveQtpm, yerr=varQtpm, label = 'TPM')
    plt.errorbar(r_array, aveQepm, yerr=varQepm, label = 'EPM')
    plt.ylabel(r'$\Delta E$')
    plt.xlabel('r')
    plt.title(r'Average heat varying initial coherences')
    plt.legend()
    #plt.savefig('C:\\Users\\simon\\Desktop\\PhD - QThermo\\Pretherm\\pretherm spatial correlations\\pretherm1_images\\std_XFTs.pdf', format='pdf')
    
    
    plt.figure(2)
    plt.grid()
    plt.plot(r_array, varQtpm, label = 'TPM')
    plt.plot(r_array, varQepm, label = 'EPM')
    plt.ylabel(r'$\mathcal{Var}(\Delta E$)')
    plt.xlabel('r')
    plt.title(r'Heat fluctuations varying initial coherences')
    plt.legend()
    #plt.savefig('C:\\Users\\simon\\Desktop\\PhD - QThermo\\Pretherm\\pretherm spatial correlations\\pretherm1_images\\std_XFTs.pdf', format='pdf')
    
    

if state == 'Sep':
    angles1 = np.linspace(0, np.pi, 50)
    angles2 = np.linspace(0, np.pi, 50)
    
    aveQepm = np.empty((len(angles1),len(angles2)))
    varQepm = np.empty((len(angles1),len(angles2)))
    
    aveQtpm = np.empty((len(angles1),len(angles2)))
    varQtpm = np.empty((len(angles1),len(angles2)))
    
    for n in range(len(angles1)):
        for m in range(len(angles2)):
            theta1 = angles1[n]
            rho_s1 = 0.5*np.array([[1+np.cos(theta1),np.sin(theta1)],[np.sin(theta1),1-np.cos(theta1)]])
            
            theta2 = angles2[m]
            rho_s2 = 0.5*np.array([[1+np.cos(theta2),np.sin(theta2)],[np.sin(theta2),1-np.cos(theta2)]])
            
            rho0 = np.kron(rho_s1,rho_s2)
            
            Ptpm = TPM(rho0, final_t, H0s, L)
            epm = EPM(rho0, final_t, H0s, L)
            Pepm = np.array(epm[0])
            
            aveQepm[n,m] = average_heat(Pepm, energy_levels)
            varQepm[n,m] = var_heat(Pepm, energy_levels)
            
            aveQtpm[n,m] = average_heat(Ptpm, energy_levels)
            varQtpm[n,m] = var_heat(Ptpm, energy_levels)
            
    # Create a meshgrid
    x, y = np.meshgrid(angles1, angles2)
    
        
    # Create a figure and a 3D axis
    fig1 = plt.figure(figsize=(12, 6))
    fig1.suptitle('Average heat varying initial coherences')
    ax1 = fig1.add_subplot(121, projection='3d')
    ax1.set_title(r'TPM')
    # Plot the surface
    ax1.plot_surface(x, y, aveQtpm, cmap='viridis')
    # Add labels
    ax1.set_xlabel(r'$\theta_1$')
    ax1.set_ylabel(r'$\theta_2$')
    ax1.set_zlabel(r'$\Delta E$')
    
    ax2 = fig1.add_subplot(122, projection='3d')
    ax2.set_title(r'EPM')
    # Plot the surface
    ax2.plot_surface(x, y, aveQepm, cmap='viridis')
    # Add labels
    ax2.set_xlabel(r'$\theta_1$')
    ax2.set_ylabel(r'$\theta_2$')
    ax2.set_zlabel(r'$\Delta E$')
    
    # Create a figure and a 3D axis
    fig2 = plt.figure(figsize=(12, 6))
    fig2.suptitle('Heat fluctuations varying initial coherences')
    ax1 = fig2.add_subplot(121, projection='3d')
    ax1.set_title(r'TPM')
    # Plot the surface
    ax1.plot_surface(x, y, varQtpm, cmap='viridis')
    # Add labels
    ax1.set_xlabel(r'$\theta_1$')
    ax1.set_ylabel(r'$\theta_2$')
    ax1.set_zlabel(r'$\mathcal{Var}(\Delta E$)')
    
    ax2 = fig2.add_subplot(122, projection='3d')
    ax2.set_title(r'EPM')
    # Plot the surface
    ax2.plot_surface(x, y, varQepm, cmap='viridis')
    # Add labels
    ax2.set_xlabel(r'$\theta_1$')
    ax2.set_ylabel(r'$\theta_2$')
    ax2.set_zlabel(r'$\mathcal{Var}(\Delta E$)')
    

if state == 'Ent':
    rho4 = 0.5*np.array([[0,0,0,0],[0,1,-1,0],[0,-1,1,0],[0,0,0,0]])
    rho3 = 0.5*np.array([[0,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,0]])
    rho1 = 0.5*np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])
    rho2 = 0.5*np.array([[1,0,0,-1],[0,0,0,0],[0,0,0,0],[-1,0,0,1]])
    
    init_rhos = [rho1,rho2,rho3,rho4]






'''Histograms
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
plt.savefig('C:\\Users\\simon\\OneDrive\\Desktop\\PhD - QThermo\\Pretherm\\pretherm spatial correlations\\pretherm1_images\\histo.pdf', format='pdf')
'''
