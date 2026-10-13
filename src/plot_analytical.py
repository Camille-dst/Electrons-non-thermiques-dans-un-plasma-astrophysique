import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from N_analytical import N_analytical , N_analytical_exp 
from constant import gamma_c, K, gamma_min, gamma_max, s, Q_0, T_ad
from N_approx import N_approx


gamma = np.logspace(0, 7, 400)
N_values = N_analytical(gamma, gamma_c, gamma_min, gamma_max, s, Q_0, K)
N_values_exp = N_analytical_exp(gamma, gamma_c, gamma_min, gamma_max, s, Q_0, K)
K_norm = Q_0 * T_ad
N_approx_values = N_approx(gamma, s=s, gamma_c=gamma_c, K=K_norm)

plt.figure(figsize=(9, 6))
plt.loglog(gamma, N_values, label=r'$N_{\text{stat}}$ (Loi de puissance pure $Q \propto \gamma^{-s}$)', color='navy', lw=2)
plt.loglog(gamma, N_values_exp, label=r'$N_{\text{stat}}(\gamma)$ (intégrale exacte)', color='pink', lw=2)
plt.loglog(gamma, N_approx_values, label=r'Approximation Éq. 2.28 ($N \propto \gamma^{-s} (1 + \gamma/\gamma_b)^{-1}$)', color='darkorange', linestyle='--')


plt.axvline(gamma_min, color='pink', linestyle=':', label=rf'$\gamma_{{\min}} = {gamma_min}$')
plt.axvline(gamma_max, color='pink', linestyle=':', label=rf'$\gamma_{{\max}} = {gamma_max:.0e}$')
plt.axvline(gamma_c, color='red', linestyle='--', label=rf'$\gamma_c = {gamma_c:.2e}$')

plt.ylim(10e-6,10e7)
plt.xlabel(r'Facteur de Lorentz $\gamma$', fontsize=12)
plt.ylabel(r'$N(\gamma)$ [u.a.]', fontsize=12)
plt.title(r'Solution stationnaire analytique exacte', fontsize=13)
plt.legend(fontsize=10)
plt.tight_layout()

plt.savefig('comparaison_toutes_analytiques.png', dpi=300)
plt.show()
