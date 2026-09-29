import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

## Define physical constants
sigma_T = 6.6524e-25  # cm^2
m_e = 9.1094e-28  # g
c = 2.9979e10  # cm/s

#constants for the calculation

B = 1 #G
R = 10e15 #cm
s = 2 #indice spectral
gamma_min = 10
gamma_max = 10e6
Q_0 = 1 #normalisation de l'injection

#calcul des grandeurs caractéristiques
u_B = B**2/(8*np.pi) #erg/cm^3
K = 4*u_B*sigma_T/(3*m_e*c) #s^-1
t_ad = R/c #s
gamma_c = 1/K*t_ad #gamma critique

#on commence par resoudre la formule en stationnaire
# Formule du temps de refroidissement selon Ghisellini (2013), Eq. 3.2).   

def integrer (gamma_c, gamma, gamma_max,s , Q_0):
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
    return (Q_0 * gamma**(- s) * np.exp(-gamma/gamma_max) * np.exp(gamma_c/gamma))


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

