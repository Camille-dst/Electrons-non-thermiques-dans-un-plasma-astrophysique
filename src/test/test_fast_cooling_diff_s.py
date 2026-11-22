import numpy as np
import matplotlib.pyplot as plt
from constant import gamma_c, K, gamma_max, Q_0, T_ad
from res_num_Euler_explicite import solve_euler_explicit
from test_fast_cooling import ajuster_segments_loglog


gamma_min = 10e4  # Régime fast cooling : gamma_c < gamma_min
gamma = np.logspace(0, 7, 400)


def test_pentes_selon_s(gamma, s_values=(1.5, 2.0, 2.5), gamma_c=gamma_c, gamma_min=gamma_min, gamma_max=gamma_max):
    """
    Compare la distribution N(gamma) * gamma^2 et les pentes des deux segments
    en régime de fast cooling pour 3 valeurs différentes de s, en réutilisant
    la fonction d'ajustement linéaire de la fonction test_fast_cooling.
    """
    plt.figure(figsize=(10, 6))
    plt.ylim(1e-4, 1e10)

    couleurs_courbes = ['navy', 'darkgreen', 'crimson']
    couleurs_pentes = ['tab:cyan', 'limegreen', 'salmon']

    for idx_s, s_val in enumerate(s_values):
        c_courbe = couleurs_courbes[idx_s % len(couleurs_courbes)]
        c_pente = couleurs_pentes[idx_s % len(couleurs_pentes)]

        N = solve_euler_explicit(
            gamma, s_val, Q_0, gamma_min, gamma_max, K, T_ad,
            safety_factor=0.5, t_max_factor=5.0
        )
        y_num = N * gamma**2

        # Tracé de la courbe avec légende de la valeur de s
        plt.loglog(gamma, y_num, label=rf'$s = {s_val}$', color=c_courbe, lw=2)

        # Appel direct de la fonction d'ajustement
        ajustements = ajuster_segments_loglog(gamma, y_num, gamma_c, gamma_min, gamma_max)

        print(f"--- Valeur de s = {s_val} ---")
        for seg_idx, (nom, x_fit, y_fit, pente, _) in enumerate(ajustements):
            print(f"  [{nom}] : pente = {pente:.3f}")
            plt.loglog(x_fit, y_fit, color=c_pente, lw=1.8, ls='--')

            # Placement de la valeur de la pente au milieu du segment (décalé légèrement selon s pour éviter le chevauchement)
            mid_idx = len(x_fit) // 2
            x_mid = x_fit[mid_idx]
            facteur_y = 1.6 * (1.8 ** idx_s) if seg_idx == 0 else 1.5 * (1.7 ** idx_s)
            y_mid = y_fit[mid_idx] * facteur_y

            plt.text(
                x_mid, y_mid,
                rf'$s={s_val}: \gamma^{{{pente:.2f}}}$',
                color=c_courbe, fontsize=9, ha='center', va='bottom', fontweight='bold'
            )

    plt.xlabel(r'Facteur de Lorentz $\gamma$', fontsize=12)
    plt.ylabel(r'$N(\gamma) \cdot \gamma^2$ [u.a.]', fontsize=12)
    plt.title(r'Évolution des pentes selon $s$ en régime de fast cooling ($\gamma_c < \gamma_{\min}$)', fontsize=13)
    plt.legend(fontsize=10, loc='upper left')
    plt.tight_layout()
    plt.savefig('comparaison_pentes_selon_s_fast_cooling.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    test_pentes_selon_s(gamma, s_values=[1.5, 2.0, 2.5])