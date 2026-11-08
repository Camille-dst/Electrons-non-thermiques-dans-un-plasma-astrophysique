import numpy as np
import matplotlib.pyplot as plt
from constant import gamma_c, K, gamma_min, gamma_max, s, Q_0, T_ad
from N_analytical_comparaison import N_analytical_exp, N_analytical_pure


def solve_euler_implicit(gamma, s, Q_0, gamma_min, gamma_max, K, T_ad, safety_factor=0.5, t_max_factor=5.0):

    N = np.zeros(len(gamma))  # État initial vide

    Q = Q_0 * (gamma**(-s))
    Q[(gamma < gamma_min) | (gamma > gamma_max)] = 0.0

    dt = 100*safety_factor
    t_max = 5*T_ad

    # Boucle temporelle vers l'état stationnaire
    t = 0.0
    while t < t_max:
        # Initialisation de la matrice A et B
        A = np.zeros((len(gamma), len(gamma)))
        b = N + dt * Q  # Termes connus au temps n

        for i in range(len(gamma) - 1):
            dgamma = gamma[i+1] - gamma[i]

            A[i,i] = 1 +dt*(K*gamma[i]**2/(dgamma) + 1/T_ad)
            A[i,i+1] = dt*(-K*gamma[i+1]**2/(dgamma))

        #condition aux bords

        dgamma_l = gamma[-1] - gamma[-2]
        A[len(gamma)-1, len(gamma)-1] = 1.0 + dt * ((K * gamma[-1]**2) / dgamma_l + 1.0 / T_ad)

        N_result = np.linalg.solve(A,b)

        # le test s'arrete avant si jamais on arrive al'état stationnaire
        if np.max(np.abs(N_result - N)) < 1e-8:
            N = N_result
            break

        N = N_result
        t+= dt

    return N


s_val =2
gamma = np.logspace(0, 7, 400) #pour le test
N = solve_euler_implicit(gamma, s_val, Q_0, gamma_min, gamma_max, K, T_ad, safety_factor=0.5, t_max_factor=5.0)


if __name__ == "__main__":

    N_theo = N_analytical_pure(gamma, gamma_c, gamma_min, gamma_max, s, Q_0, K)

    # Tracé comparatif
    plt.figure(figsize=(8, 5))
    plt.ylim(10,10e9)
    plt.loglog(gamma, N*gamma**2, label='Numérique (Euler Implicite)', color='navy', lw=2)
    plt.loglog(gamma, N_theo*gamma**2, label='Analytique (Théorie)', color='crimson', ls='--', lw=2)

    plt.xlabel(r'$\gamma$')
    plt.ylabel(r'$N(\gamma)*\gamma**2$')
    plt.title('Validation du modèle avec Cooling en euler implicite')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Résolution avec cooling (Euler implicite) en gamma*2.png', dpi=300)
    plt.show()
