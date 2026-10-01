import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from N_analytical import N_analytical 
from constant import gamma_c, K, gamma_min, gamma_max, s, Q_0



gamma = np.logspace(0, 7, 400)
N_values  = N_analytical(gamma, gamma_c, gamma_min, gamma_max, s, Q_0, K)

plt.figure(figsize=(9, 6))
plt.loglog(gamma, N_values, label=r'$N_{\text{stat}}(\gamma)$ (intégrale exacte)', color='navy', lw=2)

# Repères visuels
plt.axvline(gamma_min, color='pink', linestyle=':', label=rf'$\gamma_{{\min}} = {gamma_min}$')
plt.axvline(gamma_max, color='pink', linestyle=':', label=rf'$\gamma_{{\max}} = {gamma_max:.0e}$')
plt.axvline(gamma_c, color='red', linestyle='--', label=rf'$\gamma_c = {gamma_c:.2e}$')

#plt.ylim(10e-5,10e20)
plt.xlabel(r'Facteur de Lorentz $\gamma$', fontsize=12)
plt.ylabel(r'$N(\gamma)$ [u.a.]', fontsize=12)
plt.title(r'Solution stationnaire analytique exacte', fontsize=13)
plt.legend(fontsize=10)
#plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()

# Sauvegarde dans le dossier docs pour le rapport
plt.savefig('solution_analytique_exacte.png', dpi=300)
plt.show()