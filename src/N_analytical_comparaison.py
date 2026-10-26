import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

from constant import gamma_c, K, gamma_min, gamma_max, s, Q_0, T_ad
from N_approx import N_approx
from N_analytical import N_analytical_exp  

# Normalisation physique pour l'approximation Éq. 2.28
K_norm = Q_0 * T_ad


def integrer_pure(gamma_p,g, gamma_c,gamma_min, gamma_max, s, Q_0):
    if gamma_min <= gamma_p <= gamma_max:
        Q_gp = Q_0 * (gamma_p**(-s))
    else :
        Q_gp = 0.0
    exposant = -gamma_c * (1.0 / g - 1.0 / gamma_p)  # Formule stabilisée
    return Q_gp * np.exp(exposant)

'''
def N_analytical_pure(gamma, gamma_c, gamma_min, gamma_max, s, Q_0, K):
    N = np.zeros_like(gamma)
    for i, g in enumerate(gamma):
        if g <= gamma_max:
            I, _ = quad(integrer_pure, g, gamma_max, args=(g,gamma_c,gamma_min, gamma_max, s, Q_0))
            N[i] = I / (K * g**2)
    return N

'''
def N_analytical_pure(gamma, gamma_c, gamma_min, gamma_max, s, Q_0, K):
    N = np.zeros_like(gamma)
    for i, g in enumerate(gamma):
        gamma_p_sub = gamma[gamma >= g]
        
        if len(gamma_p_sub) > 1:
            # On évalue la fonction pour chaque point de la sous-grille
            y_vals = [integrer_pure(gp, g, gamma_c, gamma_min, gamma_max, s, Q_0) for gp in gamma_p_sub]
            
            #  méthode des trapèzes
            I = np.trapezoid(y_vals, gamma_p_sub)
            N[i] = I / (K * g**2)
            
    return N

def integrer_Q_y_exp(gamma_p, g, gamma_c,gamma_min, gamma_max, s, Q_0):
    if gamma_min <= gamma_p <= gamma_max:
        Q_gp = Q_0 * (gamma_p**(-s)) * np.exp(-gamma_p / gamma_max)
    else :
        Q_gp = 0.0
    exposant = -gamma_c * (1.0 / g - 1.0 / gamma_p)
    
    return Q_gp * np.exp(exposant)

'''
def N_analytical_exp(gamma, gamma_c, gamma_min, gamma_max, s, Q_0, K):
    N = np.zeros_like(gamma)

    for i, g in enumerate(gamma):
        if gamma_min <= g <= gamma_max:
            I, err = quad(integrer_Q_y_exp, g, gamma_max, args=(g, gamma_c, gamma_max, s, Q_0))
        
            N[i] = I / (K * g**2)
            
    return N
'''

def N_analytical_exp(gamma, gamma_c, gamma_min, gamma_max, s, Q_0, K):
    N = np.zeros_like(gamma)
    for i, g in enumerate(gamma):
        gamma_p_sub = gamma[gamma >= g]
        
        if len(gamma_p_sub) > 1:
            # On évalue la fonction pour chaque point de la sous-grille
            y_vals = [integrer_Q_y_exp(gp, g, gamma_c, gamma_min, gamma_max, s, Q_0) for gp in gamma_p_sub]
            
            #  méthode des trapèzes
            I = np.trapezoid(y_vals, gamma_p_sub)
            N[i] = I / (K * g**2)
            
    return N

gamma = np.logspace(0, 7, 400)
if __name__ == "__main__":
    

    N_pure = N_analytical_pure(gamma, gamma_c, gamma_min, gamma_max, s, Q_0, K)
    N_exp = N_analytical_exp(gamma, gamma_c, gamma_min, gamma_max, s, Q_0, K)
    N_approx_values = N_approx(gamma, s=s, gamma_c=gamma_c, K=K_norm)

    plt.figure(figsize=(9, 6))
    plt.loglog(gamma, N_pure, label=r'$N_{\text{stat}}$ (loi de puissance pure $Q \propto \gamma^{-s}$)', color='navy', lw=2)
    plt.loglog(gamma, N_exp, label=r'$N_{\text{stat}}$ (coupure exp $Q \propto \gamma^{-s} e^{-\gamma/\gamma_{\max}}$)', color='black', lw=2, ls='--')
    plt.loglog(gamma, N_approx_values, label=r'Approximation Éq. 2.28', color='violet', linestyle=':')

    # Repères visuels
    plt.axvline(gamma_min, color='gray', linestyle=':', label=rf'$\gamma_{{\min}} = {gamma_min}$')
    plt.axvline(gamma_max, color='gray', linestyle=':', label=rf'$\gamma_{{\max}} = {gamma_max:.0e}$')
    plt.axvline(gamma_c, color='red', linestyle='--', label=rf'$\gamma_c = {gamma_c:.2e}$')

    plt.ylim(10e-15, 10e6)
    plt.xlabel(r'Facteur de Lorentz $\gamma$', fontsize=12)
    plt.ylabel(r'$N(\gamma)$ [u.a.]', fontsize=12)
    plt.title(r'Comparaison des formes d\'injection $Q(\gamma)$', fontsize=13)
    plt.legend(fontsize=9)
    plt.tight_layout()

    plt.savefig('comparaison_injections.png', dpi=300)
    plt.show()