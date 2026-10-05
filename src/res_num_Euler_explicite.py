import numpy as np
import matplotlib.pyplot as plt
from constant import gamma_c, K, gamma_min, gamma_max, s, Q_0, T_ad


'''
Résolution de la ditribution d'électrons avec le terme de cooling 
selon l'équation suivante (Ghisellini 2013, Eq. 3.2) :
dN/dt =  d/dt(gamma * N(gamma)/t_cool(gamma)) + 
            Q(gamma) - N(gamma)/t_ad

            avec t_cool(gamma) = 1/(K * gamma) et t_ad = R/c
'''
gamma = np.logspace(0, 7, 400)

Q = Q_0 * (gamma**(-s))
Q[(gamma < gamma_min) | (gamma > gamma_max)] = 0.0

#dt = 0.05 * T_ad          # Pas de temps (s)
dt = 1.67       # Pas de temps (s)      
print('le pas de temps est :', dt, 's')
t_max = 5.0 * T_ad        # Temps de simulation total (s)
n_steps = int(t_max / dt)

'''
verification de la condition CFL pour le pas de temps
On doit avoir dt < min(dgamma / |gamma_dot|) 
    pour la stabilité de l'intégration

'''
dgamma = np.diff(gamma)               # Écart Δγ entre chaque point
gamma_loss = K * gamma[:-1]**2        # |γ_dot| sur la grille (sans le dernier point)

dt_cfl_array = dgamma / gamma_loss
dt_cfl_max = np.min(dt_cfl_array)
print('le pas de temps CFL est :', dt_cfl_max, 's')

if dt >= dt_cfl_max:
    print("WARNING : le pas de temps dt dépasse la limite CFL pour la stabilité.")
    print("Il est recommandé de réduire dt à une valeur inférieure à", dt_cfl_max, "s.")
    print("on pourrait prendre dt = 0.5 * dt_cfl_max =  ",0.5 * dt_cfl_max,"par exemple.")
else:
    print(" Condition CFL respectée : la simulation sera stable.")
N = np.zeros_like(gamma) #intialisation 
dF_dgamma= np.zeros_like(gamma) #intialisation du terme de cooling
#gamma_point_before = gamma_max

#Euler explicite
for step in range(n_steps):
    gamma_loss = - K * gamma**2 
    F = gamma_loss * N
    for i in range(len(gamma)-1):
        F_i = gamma_loss[i] * N[i]
        F_i_plus_1 = gamma_loss[i+1] * N[i+1]
        #gamma_point = gamma[i]
        dF_dgamma[i] = (F[i+1] - F[i]) / (gamma[i+1] - gamma[i])  # Approximation du gradient
    
    dF_dgamma[len(gamma)-1] = (0 - F[len(gamma)-1]) / (gamma[len(gamma)-1] - gamma[len(gamma)-2]) #condition au bord
    dN_dt = - dF_dgamma + Q - (N / T_ad)  # Équation différentielle
    N += dN_dt * dt        # Mise à jour de N




plt.figure(figsize=(8, 5))
plt.loglog(gamma, N, label='Euler explicite (numérique)', color='navy', lw=2)
plt.xlabel(r'$\gamma$')
plt.ylabel(r'$N(\gamma)$')
plt.title('Résolution avec cooling (Euler explicite)')
plt.legend()
plt.savefig('Résolution avec cooling (Euler explicite).png', dpi=300)
#plt.grid(True, standard_style=False, ls='--', alpha=0.5)
plt.show()