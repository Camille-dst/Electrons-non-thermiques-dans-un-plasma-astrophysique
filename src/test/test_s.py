import matplotlib.pyplot as plt
import numpy as np
from constant import gamma_c, K, gamma_min, gamma_max, Q_0, T_ad, s
from res_num_Euler_explicite import solve_euler_explicit


"""
Script de test de l'indice spectral d'injection s avec le solveur Euler explicite :
trace d'abord la vue globale pour plusieurs valeurs de s, puis un zoom ciblé
sur la zone de pile-up (avec s = 1.0 mis en avant) afin d'analyser
l'accumulation de particules.
"""


s_values = [1,1.5, 2.0, 2.5]


gamma = np.logspace(0, 7, 400)


if __name__ == "__main__":
    
    results = {}
    for s in s_values: 
        N = solve_euler_explicit(gamma, s, Q_0, gamma_min, gamma_max, K, T_ad, safety_factor=0.5, t_max_factor=5.0)
        results[s] = N

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    #  1er GRAPHIQUE :(toutes les valeurs de s) ---
    for s in s_values:
        ax1.loglog(gamma, results[s] * gamma**2, label=f'Numérique ($s = {s}$)', lw=2)

    ax1.axvline(gamma_min, color='gray', linestyle=':', label=rf'$\gamma_{{\min}} = {gamma_min}$')
    ax1.axvline(gamma_c, color='red', linestyle='--', label=rf'$\gamma_c = {gamma_c:.1f}$')
    ax1.set_ylim(1e-5, 1e10)
    ax1.set_xlabel(r'Facteur de Lorentz $\gamma$', fontsize=12)
    ax1.set_ylabel(r'$N(\gamma) \cdot \gamma^2$ [u.a.]', fontsize=12)
    ax1.set_title('Vue globale (toutes valeurs de $s$)', fontsize=13)
    ax1.legend(fontsize=9, loc='upper right')

    # 2nd GRAPHIQUE : Zoom ciblé sur la zone du pile-up 
    for s in s_values:

        alpha_val = 1.0 if s == 1.0 else 0.4
        lw_val = 2.5 if s == 1.0 else 1.5
        ax2.loglog(gamma, results[s] * gamma**2, label=f'Numérique ($s = {s}$)', lw=lw_val, alpha=alpha_val)

    ax2.axvline(gamma_min, color='gray', linestyle=':', label=rf'$\gamma_{{\min}} = {gamma_min}$')
    ax2.axvline(gamma_c, color='red', linestyle='--', label=rf'$\gamma_c = {gamma_c:.1f}$')
    

    ax2.set_xlim(10, 1e6)
    ax2.set_ylim(1e6, 1e10)
    
    ax2.set_xlabel(r'Facteur de Lorentz $\gamma$', fontsize=12)
    ax2.set_ylabel(r'$N(\gamma) \cdot \gamma^2$ [u.a.]', fontsize=12)
    ax2.set_title('Zoom sur la zone de Pile-up ($s = 1.0$ en vedette)', fontsize=13)
    ax2.legend(fontsize=9, loc='upper right')

    plt.tight_layout()
    plt.savefig('Test_modele_global_diff_s_et_zoom_pileup.png', dpi=300)
    plt.show()