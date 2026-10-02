import numpy as np
import matplotlib.pyplot as plt
from constant import gamma_c, K, gamma_min, gamma_max, s, Q_0




def N_approx(gamma, s, gamma_c, K):
    """
    Approximation analytique de la distribution d'électrons (Éq. 2.28 de ).
    
    Paramètres :
    gamma : array_like
        Facteur de Lorentz des électrons.
    s : float
        Indice d'injection.
    gamma_c : float
        Facteur de Lorentz critique.
    K : float
        Constante de normalisation.
    """
    gamma = np.asarray(gamma)
    return K * (gamma**(-s)) * (1.0 + gamma / gamma_c)**(-1.0)

'''

# Grille de calcul
gamma_grid = np.logspace(0, 7, 500)

# Calcul avec la formule simplifiée (en posant gamma_b = gamma_c)
#N_approx = N_approx(gamma_grid, s=s, gamma_c=gamma_c, Q_0=Q_0)

plt.figure(figsize=(8, 5))
plt.loglog(gamma_grid, N_approx, label=r'Approximation Éq. 2.28 ($N \propto \gamma^{-s} (1 + \gamma/\gamma_b)^{-1}$)', color='crimson', linestyle='--')

plt.axvline(gamma_c, color='gray', linestyle=':', label=rf'$\gamma_b = \gamma_c = {gamma_c:.2e}$')
plt.xlabel(r'Facteur de Lorentz $\gamma$')
plt.ylabel(r'$N(\gamma)$')
plt.title('Approximation analytique simplifiée')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.show()
'''