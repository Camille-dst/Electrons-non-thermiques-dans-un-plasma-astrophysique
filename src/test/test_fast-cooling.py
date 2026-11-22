import numpy as np
import matplotlib.pyplot as plt
from constant import gamma_c, K, gamma_max, s, Q_0, T_ad
from res_num_Euler_explicite import solve_euler_explicit
from N_analytical_comparaison import N_analytical_exp

"""
Étude du régime de fast cooling (gamma_c < gamma_min) avec ajustements linéaires
en échelle log-log sur les segments séparés par gamma_c, gamma_min et gamma_max.
Affiche côte à côte :
  - À gauche : le tracé complet actuel (numérique, analytique et repères).
  - À droite : uniquement la courbe numérique et ses droites d'ajustement.
"""

gamma_min = 10e4  #on prend gamma_min > gamma_c
gamma = np.logspace(0, 7, 400)

# Pour avoir gamma_c < gamma_min (ex: gamma_c ~ 1 ou 2, alors que gamma_min = 10),
# on choisit une grande valeur de R (ou un grand B)
#R_grand = 5.0e18  # Augmente R pour faire baisser gamma_c
#T_ad_test = R_grand / c
#gamma_c_test = 1.0 / (K * T_ad_test)

print(f"Valeur de test -> gamma_c = {gamma_c:.2f} (inférieure à gamma_min = {gamma_min})")


def ajuster_segments_loglog(gamma, y_vals, gamma_c, gamma_min, gamma_max):
    """
    ajustements linéaires dans l'espace (log10(gamma), log10(y_vals))
    sur les intervalles entre gamma_c, gamma_min et gamma_max .
    """
    intervalles = [

        ("gamma_c -> gamma_min", gamma_c * 1.2, gamma_min * 0.8),
        ("gamma_min -> gamma_max", gamma_min * 1.2, gamma_max * 0.3),
    ]

    ajustements = []
    log_x = np.log10(gamma)
    log_y = np.log10(np.maximum(y_vals, 1e-300))

    for nom, g_inf, g_sup in intervalles:
        if g_sup > g_inf:
            mask = (
                (gamma >= g_inf)
                & (gamma <= g_sup)
                & np.isfinite(y_vals)
                & (y_vals > 0)
            )
            if np.sum(mask) >= 3:
                m, b = np.polyfit(log_x[mask], log_y[mask], 1)
                x_fit = np.geomspace(g_inf, g_sup, 100)
                y_fit = 10 ** (m * np.log10(x_fit) + b)
                ajustements.append((nom, x_fit, y_fit, m, b))

    return ajustements


if __name__ == "__main__":
    N = solve_euler_explicit(
        gamma, s, Q_0, gamma_min, gamma_max, K, T_ad,
        safety_factor=0.5, t_max_factor=5.0
    )
    N_theo = N_analytical_exp(gamma, gamma_c, gamma_min, gamma_max, s, Q_0, K)

    y_num = N * gamma**2
    y_theo = N_theo * gamma**2

    ajustements = ajuster_segments_loglog(gamma, y_num, gamma_c, gamma_min, gamma_max)
    for nom, _, _, pente, _ in ajustements:
        print(f"Ajustement [{nom}] : pente = {pente:.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- FIGURE 1 (gauche) : Ta figure actuelle ---
    ax1.set_ylim(10e-5, 10e9)
    ax1.loglog(
        gamma, y_num,
        label=rf'Numérique ($\gamma_c = {gamma_c:.1f} < \gamma_{{\min}}$)',
        color='purple', lw=2
    )
    ax1.loglog(gamma, y_theo, label='Analytique (Théorie)', color='pink', ls=':', lw=2)

    ax1.axvline(gamma_min, color='gray', linestyle='--', label=rf'$\gamma_{{\min}} = {gamma_min}$')
    ax1.axvline(gamma_c, color='red', linestyle='--', label=rf'$\gamma_c = {gamma_c:.2f}$')

    ax1.set_xlabel(r'Facteur de Lorentz $\gamma$', fontsize=12)
    ax1.set_ylabel(r'$N(\gamma) \cdot \gamma^2$ [u.a.]', fontsize=12)
    ax1.set_title(r'Test du régime de refroidissement rapide ($\gamma_c < \gamma_{\min}$)', fontsize=13)
    ax1.legend(fontsize=10)

    # --- FIGURE 2 (droite) : Courbe + 2 pentes type Figure 1 + gamma_c au-dessus ---
    ax2.set_ylim(10e-5, 10e9)
    ax2.loglog(gamma, y_num, color='purple', lw=2)

    couleurs_fit = ['tab:orange', 'tab:cyan']
    for idx, (nom, x_fit, y_fit, pente, _) in enumerate(ajustements):
        c_fit = couleurs_fit[idx % len(couleurs_fit)]
        ax2.loglog(x_fit, y_fit, color=c_fit, lw=2.2, ls='--')

        # Milieu géométrique du segment pour placer l'annotation de pente comme dans le papier
        mid_idx = len(x_fit) // 2
        x_mid = x_fit[mid_idx]
        y_mid = y_fit[mid_idx] * 1.8

        ax2.text(
            x_mid, y_mid,
            rf'$\propto \gamma^{{{pente:.2f}}}$',
            color=c_fit, fontsize=10, ha='center', va='bottom', fontweight='bold'
        )


    ax2.set_xlabel(r'Facteur de Lorentz $\gamma$', fontsize=12)
    ax2.set_ylabel(r'$N(\gamma) \cdot \gamma^2$ [u.a.]', fontsize=12)
    ax2.set_title('Ajustements linéaires des deux pentes (fast cooling)', fontsize=13)

    plt.tight_layout()
    plt.savefig('test_fast_cooling_ajustements.png', dpi=300)
    plt.show()