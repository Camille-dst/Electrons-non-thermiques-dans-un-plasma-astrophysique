"""
Script de test de la sensibilité du phénomène de pile-up (à s = 1.0) aux
variations du champ magnétique B (via le coefficient de refroidissement K)
et de la taille de la source R (via le temps d'échappement/advection T_ad).
Permet d'observer si le pile-up réagit de la même façon ou différemment
selon B et R.
"""

import matplotlib.pyplot as plt
import numpy as np
from constant import gamma_c, K, gamma_min, gamma_max, Q_0, T_ad
from res_num_Euler_explicite import solve_euler_explicit

gamma = np.logspace(0, 7, 500)
s_pileup = 1.0

# Facteurs multiplicatifs pour faire varier B (donc K) et R (donc T_ad)
# Note : selon le modèle physique de ton fichier constant.py, adapte la relation
# exacte entre B -> K et R -> T_ad si nécessaire.
facteurs_B = [0.5, 1.0, 2.0]  # si K ~ B^2, K_val = K * (facteur_B**2) ou directement K * facteur_B
facteurs_R = [0.5, 1.0, 2.0]  # si T_ad ~ R / v, T_ad_val = T_ad * facteur_R


def test_B_et_R_pileup(gamma, s=1.0, facteurs_B=None, facteurs_R=None):
    """
    Teste l'effet d'une variation de B (via K) et de R (via T_ad) à s = 1.0
    sur la distribution N(gamma) * gamma^2 et la zone de pile-up.
    """
    if facteurs_B is None:
        facteurs_B = [0.5, 1.0, 2.0]
    if facteurs_R is None:
        facteurs_R = [0.5, 1.0, 2.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- Panneau gauche : Variation de B (impact sur le refroidissement K) ---
    for fB in facteurs_B:
        # Adapte fB**2 si ton K est explicitement proportionnel à B^2 dans constant.py
        K_val = K * (fB ** 2)
        N_B = solve_euler_explicit(
            gamma, s, Q_0, gamma_min, gamma_max, K_val, T_ad,
            safety_factor=0.5, t_max_factor=5.0
        )
        ax1.loglog(gamma, N_B * gamma**2, label=rf'$B / B_0 = {fB}$ ($K$ modifié)', lw=2)

    ax1.axvline(gamma_min, color='gray', linestyle=':', label=rf'$\gamma_{{\min}} = {gamma_min}$')
    ax1.axvline(gamma_c, color='red', linestyle='--', label=rf'$\gamma_{{c,0}} = {gamma_c:.1f}$')
    ax1.set_ylim(1e5, 1e12)
    ax1.set_xlabel(r'Facteur de Lorentz $\gamma$', fontsize=12)
    ax1.set_ylabel(r'$N(\gamma) \cdot \gamma^2$ [u.a.]', fontsize=12)
    ax1.set_title(rf'Influence de $B$ sur le pile-up ($s = {s}$)', fontsize=13)
    ax1.legend(fontsize=9, loc='upper right')
    #ax1.grid(True, which="both", ls="--", alpha=0.5)

    # --- Panneau droit : Variation de R (impact sur l'échappement T_ad) ---
    for fR in facteurs_R:
        T_ad_val = T_ad * fR
        N_R = solve_euler_explicit(
            gamma, s, Q_0, gamma_min, gamma_max, K, T_ad_val,
            safety_factor=0.5, t_max_factor=5.0
        )
        ax2.loglog(gamma, N_R * gamma**2, label=rf'$R / R_0 = {fR}$ ($T_{{\mathrm{{ad}}}}$ modifié)', lw=2)

    ax2.axvline(gamma_min, color='gray', linestyle=':', label=rf'$\gamma_{{\min}} = {gamma_min}$')
    ax2.axvline(gamma_c, color='red', linestyle='--', label=rf'$\gamma_{{c,0}} = {gamma_c:.1f}$')
    ax2.set_ylim(1e5, 1e12)
    ax2.set_xlabel(r'Facteur de Lorentz $\gamma$', fontsize=12)
    ax2.set_ylabel(r'$N(\gamma) \cdot \gamma^2$ [u.a.]', fontsize=12)
    ax2.set_title(rf'Influence de $R$ sur le pile-up ($s = {s}$)', fontsize=13)
    ax2.legend(fontsize=9, loc='upper right')
    #ax2.grid(True, which="both", ls="--", alpha=0.5)

    plt.savefig('test_pileup_B_et_R.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    test_B_et_R_pileup(gamma, s=s_pileup, facteurs_B=facteurs_B, facteurs_R=facteurs_R)