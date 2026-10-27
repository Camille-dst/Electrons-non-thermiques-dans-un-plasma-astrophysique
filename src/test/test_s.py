import matplotlib.pyplot as plt
import numpy as np
from constant import gamma_c, K, gamma_min, gamma_max, Q_0, T_ad, s
from res_num_Euler_explicite import solve_euler_explicit

s_values = [1,1.5, 2.0, 2.5]


gamma = np.logspace(0, 7, 400)



if __name__ == "__main__":
    for s in s_values: 
        N = solve_euler_explicit(gamma,s,Q_0,gamma_min,gamma_max,K,T_ad,safety_factor=0.5,t_max_factor=5.0)

        plt.loglog(gamma, N*gamma**2, label=f'Numérique (s = {s})', lw=2)

    plt.ylim(10e-5,10e9)
    plt.xlabel(r'$\gamma$')
    plt.ylabel(r'$N(\gamma)*\gamma^2$')
    plt.title('Test du modèle pour différentes valeurs')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Test de modèle pour diff s_values_gamma^2.png', dpi=300)
    plt.show()