import numpy as np
import matplotlib.pyplot as plt
from constant import gamma_c, K, gamma_min, gamma_max, s, Q_0, T_ad

gamma = np.logspace(0, 7, 400)

Q = Q_0 * (gamma**(-s))
Q[(gamma < gamma_min) | (gamma > gamma_max)] = 0.0


dt = 0.05 * T_ad          # Pas de temps (s)
t_max = 5.0 * T_ad        # Temps de simulation total (s)
n_steps = int(t_max / dt)

N = np.zeros_like(gamma) #intialisation 

#Euler explicite
for step in range(n_steps):
    dN_dt = Q - (N / T_ad)  # Équation différentielle
    N += dN_dt * dt        # Mise à jour de N

N_cooling_exacte = Q * T_ad * (1.0 - np.exp(-t_max / T_ad))  # Solution exacte pour comparaison

plt.figure(figsize=(8, 5))
plt.loglog(gamma, N, label='Euler explicite (numérique)', color='navy', lw=2)
plt.loglog(gamma, N_cooling_exacte, label=r'Asymptote $Q(\gamma) \cdot t_{\text{esc}}$', color='crimson', ls='--')
plt.xlabel(r'$\gamma$')
plt.ylabel(r'$N(\gamma)$')
plt.title('Résolution sans cooling (Euler explicite)')
plt.legend()
plt.savefig('Résolution sans cooling (Euler explicite).png', dpi=300)
#plt.grid(True, standard_style=False, ls='--', alpha=0.5)
plt.show()