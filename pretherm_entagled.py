from pretherm_main import *

#MAXIMALLY ENTANGLED INITIAL STATEs
rho1 = 0.5*np.array([[0,0,0,0],[0,1,-1,0],[0,-1,1,0],[0,0,0,0]])
rho2 = 0.5*np.array([[0,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,0]])
rho3 = 0.5*np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])
rho4 = 0.5*np.array([[1,0,0,-1],[0,0,0,0],[0,0,0,0],[-1,0,0,1]])

rho_set = [rho1,rho2,rho3,rho4]

mz = []
Mxx = []
Myy = []
Mzz = []

for rho in rho_set:
    mz.append(m(I, rho))
    Mxx.append(M(I, rho)[0,0])
    Myy.append(M(I, rho)[1,1])
    Mzz.append(M(I, rho)[2,2])


#Time step of th evolution and number of total steps
dt = 0.01
num_steps = 10000000
times = [(n+1)*dt for n in range(0,num_steps//1000)] 
times = times + [(n+1)*dt for n in range(num_steps//1000,num_steps,100)]
length = len(times)

Qtpm = []
Qepm = []

#--------------------------------ENERGY FLUCTUATIONS---------------------------
for rho0 in rho_set:

    Ptpm = np.array([TPM(rho0, t, H0s, L) for t in times])
    Qtpm.append(np.array([average_heat(Ptpm[k],energy_levels) for k in range(length)]))
    
    Pepm = []
    
    for t in times:
        x = EPM(rho0, t, H0s, L)
        Pepm.append(np.array(x[0]))
    
    Qepm.append(np.array([average_heat(Pepm[k],energy_levels) for k in range(length)]))
    
fig, axs = plt.subplots(2, 2,figsize=(9, 9))
axs[0,0].plot(times, Qtpm[2] ,label = 'TPM')
axs[0,0].plot(times, Qepm[2] ,label = 'EPM')
axs[0,0].legend()
axs[0,0].grid()
axs[0,0].set_title(r'$\Phi^+$')
axs[0,0].set_xscale('log')

axs[0,1].plot(times, Qtpm[3] ,label = 'TPM')
axs[0,1].plot(times, Qepm[3] ,label = 'EPM')
axs[0,1].legend()
axs[0,1].grid()
axs[0,1].set_title(r'$\Phi^-$')
axs[0,1].set_xscale('log')

axs[1,0].plot(times, Qtpm[1] ,label = 'TPM')
axs[1,0].plot(times, Qepm[1] ,label = 'EPM')
axs[1,0].legend()
axs[1,0].grid()
axs[1,0].set_title(r'$\Psi^+$')
axs[1,0].set_xscale('log')

axs[1,1].plot(times, Qtpm[0] ,label = 'TPM')
axs[1,1].plot(times, Qepm[0] ,label = 'EPM')
axs[1,1].legend()
axs[1,1].grid()
axs[1,1].set_title(r'$\Psi^-$')
axs[1,1].set_xscale('log')
fig.suptitle(r'Maximally Entangled initial states- $\alpha = $'+str(alfa))
for ax in axs.flat:
    ax.set(xlabel='final time')
plt.savefig('C:\\Users\\simon\\OneDrive\\Desktop\\PhD - QThermo\\Pretherm\\pretherm spatial correlations\\pretherm1_images\\ent_fluct.pdf', format='pdf')
    
