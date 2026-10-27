import matplotlib.pyplot as plt
import numpy as np
from constant import sigma_t, m_e, c, gamma_min, gamma_max, Q_0, T_ad, s
from res_num_Euler_explicite import solve_euler_explicit

B_values = np.array([0.1,0.3,1,3.0])

'''on redifinie les constantes qui ont un rapport avec B'''
u_b_values = B_values**2 / (8 * np.pi)                        # Densité d'énergie magnétique (erg/cm³)
K_values = (4 * u_b_values * sigma_t) / (3 * m_e * c)          # Coefficient de perte synchrotron (s⁻¹)
gamma_c_values = 1.0 / (K_values * T_ad)

gamma = np.logspace(0, 7, 400)



if __name__ == "__main__":

    for K, gamma_c, B in zip(K_values, gamma_c_values, B_values): 
        N = solve_euler_explicit(gamma,s,Q_0,gamma_min,gamma_max,K,T_ad,safety_factor=0.5,t_max_factor=5.0)

        plt.loglog(gamma, N*gamma**2, label=f'Numérique (B = {B})', lw=2)
        #plt.axvline(gamma_c, linestyle='--', label=rf'$\gamma_c = {gamma_c:.2e}$')


    plt.ylim(10e-5,10e9)
    plt.xlabel(r'$\gamma$')
    plt.ylabel(r'$N(\gamma)*\gamma^2$')
    plt.title('Test du modèle pour différentes valeurs de B')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Test de modèle pour diff B_values_gamma^2.png', dpi=300)
    plt.show()