import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from N_approx import N_approx
from N_analytical import N_analytical_exp, integrer_Q_y_exp
#from constant import gamma_c, K, gamma_min, gamma_max, s, Q_0
'''
# Constantes physiques (CGS)
sigma_T = 6.6524e-25  # cm^2
m_e = 9.1094e-28      # g
c = 2.9979e10         # cm/s

# Paramètres du problème
B = 1                 # G
R = 10e15             # cm
s = 2                 # Indice spectral
gamma_min = 10
gamma_max = 10e6
Q_0 = 1               # Normalisation

# Grandeurs dérivées
u_B = B**2 / (8 * np.pi)
K = 4 * u_B * sigma_T / (3 * m_e * c)
t_ad = R / c
gamma_c = 1 / (K * t_ad)

'''
from constant import gamma_c, K, gamma_min, gamma_max, s, Q_0, T_ad

K_norm = Q_0 * T_ad

#on commence par resoudre la formule en stationnaire
# Formule du temps de refroidissement selon Ghisellini (2013), Eq. 3.2).   

def integrer (gamma, gamma_c, gamma_max,s , Q_0):
    """
    Integrate de la distribution de puissance.

    Parametre:
    gamma : float
        Facteur de Lorentz des electrons.
    gamma_min : float
        Min du facteur de Lorentz.
    gamma_max : float
        Maximum du facteur de Lorentz.
    s : float
        indice spectral.

    Returns:
    float
        Valeurs intégrés de la distribution.
    """
    Q_gp = Q_0 * (gamma**(-s)) #* np.exp(-gamma / gamma_max)
    exposant = gamma_c / gamma
    return (Q_gp * np.exp(exposant))


def N_analytical(gamma,gamma_c,gamma_min,gamma_max,s,Q_0,K):
    """
    Distributtion analytique de la densité d'electrons en stationnaire.
    
    Parametre:
    gamma : array
        Facteur de Lorentz des electrons.
    gamma_c : float
        Facteur de Lorentz critique.
    gamma_max : float
        Maximum du facteur de Lorentz.
    gamma_min : float
        Min du facteur de Lorentz.
    s : float
        indice spectral.
    Q_0 : float
        Paramètre de normalisation de l'injection.
    K : float
        Constante de la formule.

    Returns:
    float
        .
    """
    N = np.zeros_like(gamma)

    for i , g in enumerate(gamma):
        if gamma_min < g < gamma_max:
            I,err = quad(integrer,g,gamma_max,args=(gamma_c,gamma_max,s,Q_0))
            N[i] = np.exp(-gamma_c/g) *I/(K*g**2)
    return N


gamma = np.logspace(0, 7, 400)
N_values  = N_analytical(gamma, gamma_c, gamma_min, gamma_max, s, Q_0, K)
N_approx_values = N_approx(gamma, s=s, gamma_c=gamma_c, K=K_norm)

N_values_exp  = N_analytical_exp(gamma, gamma_c, gamma_min, gamma_max, s, Q_0, K)
N_approx_values = N_approx(gamma, s=s, gamma_c=gamma_c, K=K_norm)


plt.figure(figsize=(9, 6))

plt.loglog(gamma, N_values, label=r'$N_{\text{stat}}(\gamma)$ (intégrale exacte)', color='navy', lw=2)
plt.loglog(gamma, N_approx_values, label=r'Approximation Éq. 2.28 ($N \propto \gamma^{-s} (1 + \gamma/\gamma_b)^{-1}$)', color='violet', linestyle='--')


# Repères visuels
plt.axvline(gamma_min, color='pink', linestyle=':', label=rf'$\gamma_{{\min}} = {gamma_min}$')
plt.axvline(gamma_max, color='pink', linestyle=':', label=rf'$\gamma_{{\max}} = {gamma_max:.0e}$')
plt.axvline(gamma_c, color='red', linestyle='--', label=rf'$\gamma_c = {gamma_c:.2e}$')

plt.ylim(10e-15,10e6)
plt.xlabel(r'Facteur de Lorentz $\gamma$', fontsize=12)
plt.ylabel(r'$N(\gamma)$ [u.a.]', fontsize=12)
plt.title(r'Comparaison solution exacte / approximation', fontsize=13)
plt.legend(fontsize=10)
#plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()

# Sauvegarde dans le dossier docs pour le rapport
plt.savefig('Comparaison_solution_exacte-approximation.png', dpi=300)
plt.show()

plt.figure(figsize=(9, 6))

plt.loglog(gamma, N_values_exp, label=r'$N_{\text{stat}}(\gamma)$ (approximation exponentielle)', color='black', lw=2, ls='-')
plt.loglog(gamma, N_approx_values, label=r'Approximation Éq. 2.28 ($N \propto \gamma^{-s} (1 + \gamma/\gamma_b)^{-1}$)', color='violet', linestyle='--')


# Repères visuels
plt.axvline(gamma_min, color='pink', linestyle=':', label=rf'$\gamma_{{\min}} = {gamma_min}$')
plt.axvline(gamma_max, color='pink', linestyle=':', label=rf'$\gamma_{{\max}} = {gamma_max:.0e}$')
plt.axvline(gamma_c, color='red', linestyle='--', label=rf'$\gamma_c = {gamma_c:.2e}$')

plt.ylim(10e-15,10e6)
plt.xlabel(r'Facteur de Lorentz $\gamma$', fontsize=12)
plt.ylabel(r'$N(\gamma)$ [u.a.]', fontsize=12)
plt.title(r'Solution stationnaire analytique exacte', fontsize=13)
plt.legend(fontsize=10)
#plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig('comparaison_solution_exacte_exponentielle.png', dpi=300)
plt.show()