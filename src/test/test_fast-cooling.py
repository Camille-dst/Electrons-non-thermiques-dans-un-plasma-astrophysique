import numpy as np
import matplotlib.pyplot as plt
from constant import gamma_c, K, gamma_max, s, Q_0, T_ad
from res_num_Euler_explicite import solve_euler_explicit
from N_analytical_comparaison import N_analytical_exp

'''
test dans le régime de fast cooling, c a d pour 
gamma_min > gamma_max et observation 
'''

gamma_min = 10e4  #on prend gamma_min > gamma_c
gamma = np.logspace(0, 7, 400)

# Pour avoir gamma_c < gamma_min (ex: gamma_c ~ 1 ou 2, alors que gamma_min = 10),
# on choisit une grande valeur de R (ou un grand B)
#R_grand = 5.0e18  # Augmente R pour faire baisser gamma_c
#T_ad_test = R_grand / c
#gamma_c_test = 1.0 / (K * T_ad_test)

print(f"Valeur de test -> gamma_c = {gamma_c:.2f} (inférieure à gamma_min = {gamma_min})")

if __name__ == "__main__":
    N = solve_euler_explicit(gamma, s, Q_0, gamma_min, gamma_max, K, T_ad, safety_factor=0.5, t_max_factor=5.0)
    N_theo = N_analytical_exp(gamma, gamma_c, gamma_min, gamma_max, s, Q_0, K)
    plt.figure(figsize=(9, 6))
    plt.ylim(10e-5,10e9)
    plt.loglog(gamma, N * gamma**2, label=rf'Numérique ($\gamma_c = {gamma_c:.1f} < \gamma_{{\min}}$)', color='purple', lw=2)
    plt.loglog(gamma, N_theo*gamma**2, label='Analytique (Théorie)', color='pink', ls=':', lw=2)

    plt.axvline(gamma_min, color='gray', linestyle='--', label=rf'$\gamma_{{\min}} = {gamma_min}$')
    plt.axvline(gamma_c, color='red', linestyle='--', label=rf'$\gamma_c = {gamma_c:.2f}$')

    plt.xlabel(r'Facteur de Lorentz $\gamma$', fontsize=12)
    plt.ylabel(r'$N(\gamma) \cdot \gamma^2$ [u.a.]', fontsize=12)
    plt.title(r'Test du régime de refroidissement rapide ($\gamma_c < \gamma_{\min}$)', fontsize=13)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('test_refroidissement_rapide.png', dpi=300)
    plt.show()