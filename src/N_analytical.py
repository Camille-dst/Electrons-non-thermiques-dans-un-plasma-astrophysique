import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from constant import gamma_c, K, gamma_min, gamma_max, s, Q_0

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
    Q_gp = Q_0 * (gamma**(-s)) * np.exp(-gamma / gamma_max)
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

