import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from N_analytical import N_analytical 

c = 2.99792458e10       # Vitesse de la lumière (cm/s)
m_e = 9.10938356e-28    # Masse de l'électron (g)
sigma_T = 6.6524587e-25 #

B = 1.0                 # Champ magnétique (Gauss)
R = 1e15                # Taille de la zone d'émission (cm)
s = 2.2                 # Indice d'injection
gamma_min = 10.0        # Lorentz factor min
gamma_max = 1e6         # Lorentz factor max
Q_0 = 1.0               # Normalisation de l'injection

# 3. Grandeurs physiques dérivées
u_B = (B**2) / (8 * np.pi)                          # Densité d'énergie magnétique
K = (4/3) * (sigma_T * c / (m_e * c**2)) * u_B    # Coefficient de perte synchrotron
t_esc = R / c                                       # Temps d'échappement
gamma_c = 1.0 / (K * t_esc)


gamma = np.logspace(0, 7, 400)
N_values  = N_analytical(gamma, gamma_c, gamma_min, gamma_max, s, Q_0, K)

plt.figure(figsize=(9, 6))
plt.loglog(gamma, N_values, label=r'$N_{\text{stat}}(\gamma)$ (intégrale exacte)', color='navy', lw=2)

# Repères visuels
plt.axvline(gamma_min, color='pink', linestyle=':', label=rf'$\gamma_{{\min}} = {gamma_min}$')
plt.axvline(gamma_max, color='pink', linestyle=':', label=rf'$\gamma_{{\max}} = {gamma_max:.0e}$')
plt.axvline(gamma_c, color='red', linestyle='--', label=rf'$\gamma_c = {gamma_c:.2e}$')

plt.xlabel(r'Facteur de Lorentz $\gamma$', fontsize=12)
plt.ylabel(r'$N(\gamma)$ [u.a.]', fontsize=12)
plt.title(r'Solution stationnaire analytique exacte', fontsize=13)
plt.legend(fontsize=10)
#plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()

# Sauvegarde dans le dossier docs pour le rapport
plt.savefig('solution_analytique_exacte.png', dpi=300)
plt.show()