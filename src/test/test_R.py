import matplotlib.pyplot as plt
import numpy as np
from constant import sigma_t, m_e, c, gamma_min, gamma_max, Q_0, s, B , K
from res_num_Euler_explicite import solve_euler_explicit

R_values = np.array([10e13 ,10e14 ,10e15 ,10e16])

'''on redifinie les constantes qui ont un rapport avec R'''
T_ad_values = R_values / c  
gamma_c_values = 1.0 / (K * T_ad_values)

gamma = np.logspace(0, 7, 400)



if __name__ == "__main__":

    for R, gamma_c, T_ad in zip(R_values, gamma_c_values, T_ad_values): 
        N = solve_euler_explicit(gamma,s,Q_0,gamma_min,gamma_max,K,T_ad,safety_factor=0.5,t_max_factor=5.0)
        print('le temps d\'échappement est :', T_ad, 's')   
        plt.loglog(gamma, N*gamma**2, label=f'Numérique (R = {R} cm)', lw=2)
        #plt.axvline(gamma_c, linestyle='--', label=rf'$\gamma_c = {gamma_c:.2e}$')


    plt.ylim(10e-5,10e9)
    plt.xlabel(r'$\gamma$')
    plt.ylabel(r'$N(\gamma)*\gamma^2$')
    plt.title('Test du modèle pour différentes valeurs de R')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Test de modèle pour diff R_values_gamma^2.png', dpi=300)
    plt.show()