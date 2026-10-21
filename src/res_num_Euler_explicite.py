import numpy as np
import matplotlib.pyplot as plt
from constant import gamma_c, K, gamma_min, gamma_max, s, Q_0, T_ad
from N_analytical_comparaison import N_analytical_pure

'''
Résolution de la ditribution d'électrons avec le terme de cooling 
selon l'équation suivante (Ghisellini 2013, Eq. 3.2) :
dN/dt =  d/dt(gamma * N(gamma)/t_cool(gamma)) + 
            Q(gamma) - N(gamma)/t_ad

            avec t_cool(gamma) = 1/(K * gamma) et t_ad = R/c
'''

def solve_euler_explicit(gamma, s_val, Q_0, gamma_min, gamma_max, K, T_ad, safety_factor=0.5, t_max_factor=5.0):
    """
    Résout l'équation d'évolution de N(gamma) par le schéma d'Euler explicite et Upwind.
    """

    Q = Q_0 * (gamma**(-s_val))
    Q[(gamma < gamma_min) | (gamma > gamma_max)] = 0.0

    
    t_max = 5.0 * T_ad        # Temps de simulation total (s)


    '''
    verification de la condition CFL pour le pas de temps
    On doit avoir dt < min(dgamma / |gamma_dot|) 
    pour la stabilité de l'intégration
    '''
    dgamma = np.diff(gamma)             
    gamma_loss = K * gamma[:-1]**2       
    dt_cfl_array = dgamma / gamma_loss
    dt_cfl_max = np.min(dt_cfl_array)
    print('le pas de temps CFL est :', dt_cfl_max, 's')

    safety_factor = 0.5
    dt = safety_factor * dt_cfl_max
    print('le pas de temps choisie est :', dt, 's')
    n_steps = int(t_max / dt)

    if dt >= dt_cfl_max:
        print("WARNING : le pas de temps dt dépasse la limite CFL pour la stabilité.")
        print("Il est recommandé de réduire dt à une valeur inférieure à", dt_cfl_max, "s.")
        print("on pourrait prendre dt = 0.5 * dt_cfl_max =  ",0.5 * dt_cfl_max,"par exemple.")
    else:
        print(" Condition CFL respectée : la simulation sera stable.")


    N = np.zeros_like(gamma) #intialisation 
    dF_dgamma= np.zeros_like(gamma) #intialisation du terme de cooling


    #Euler explicite
    for step in range(n_steps):
        gamma_loss = - K * gamma**2 
        F = gamma_loss * N

        for i in range(len(gamma)-1):
            dF_dgamma[i] = (F[i+1] - F[i]) / (gamma[i+1] - gamma[i])  # Approximation du gradient
    
        dF_dgamma[len(gamma)-1] = (0 - F[len(gamma)-1]) / (gamma[len(gamma)-1] - gamma[len(gamma)-2]) #condition au bord
        dN_dt = - dF_dgamma + Q - (N / T_ad)  # Équation différentielle
        N += dN_dt * dt        # Mise à jour de N
    return N

#s_val =2
#gamma = np.logspace(0, 7, 400) #pour le test
#N = solve_euler_explicit(gamma, s_val, Q_0, gamma_min, gamma_max, K, T_ad, safety_factor=0.5, t_max_factor=5.0)

'''if __name__ == "__main__":

    N_theo = N_analytical_pure(gamma, gamma_c, gamma_min, gamma_max, s, Q_0, K)

    # Tracé comparatif
    plt.figure(figsize=(8, 5))
    plt.ylim(10e-9,10e9)
    plt.loglog(gamma, N, label='Numérique (Euler Explicite)', color='navy', lw=2)
    plt.loglog(gamma, N_theo, label='Analytique (Théorie)', color='crimson', ls='--', lw=2)

    plt.xlabel(r'$\gamma$')
    plt.ylabel(r'$N(\gamma)$')
    plt.title('Validation du modèle avec Cooling')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Résolution avec cooling (Euler explicite).png', dpi=300)
    plt.show()
'''