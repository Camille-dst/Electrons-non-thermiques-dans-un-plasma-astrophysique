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

