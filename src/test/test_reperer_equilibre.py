import numpy as np
import matplotlib.pyplot as plt
from constant import gamma_c, K, s, Q_0, T_ad, gamma_max, gamma_min
from res_num_implicite import solve_euler_implicit 
from N_analytical_comparaison import N_analytical_pure

def test_repere_equilibre_avec_fonction_fixe(gamma_min, gamma_max, gamma):

    #snapshot_times = [0.0, 50.0, 200.0, 1000.0, 5000.0]
    snapshot_times = [0.0] + list(np.logspace(np.log10(0.05 * T_ad), np.log10(5.0 * T_ad), 5))
    print("Temps de snapshots choisis automatiquement :", [f"{t:.1e}" for t in snapshot_times])

    snapshots = {}
    N_current = np.zeros(len(gamma)) 
    
    current_t = 0.0
    snapshots[current_t] = N_current.copy()

    print("Génération des snapshots par appels successifs...")
    

    for t in snapshot_times[1:]:
        # On appelle ta fonction d'origine en lui fixant t_max = target_t
        # Note : Si ta fonction d'origine commence toujours par N = np.zeros, 
        # il faudrait idéalement lui passer N_init=N_current. 
        # Si elle utilise l'état global ou repart de zéro, dis-le-moi.
        # En lui passant t_max=target_t, elle simule du temps 0 jusqu'à target_t d'un coup.
        
        N_current = solve_euler_implicit(
            gamma, s, Q_0, gamma_min, gamma_max, K, T_ad, 
            safety_factor=0.5, t_max=t
        )
        
        snapshots[t] = N_current.copy()
        current_t = t
        print(f"-> Snapshot capturé pour t = {t}")
        
    return snapshots, snapshot_times

#gamma_min = 2.0e3
#gamma_max = 2.2e3
gamma = np.logspace(0, 7, 500)

if __name__ == "__main__":

    #snapshot_times = [0.0, 50.0, 200.0, 1000.0, 5000.0]

    snapshots, snapshot_times = test_repere_equilibre_avec_fonction_fixe(gamma_min, gamma_max, gamma)
    
    N_theo = N_analytical_pure(gamma, gamma_c, gamma_min, gamma_max, s, Q_0, K)


    plt.figure(figsize=(10, 6))
    plt.ylim(1e-4, 1e9)
    
    # Dégradé de couleurs du clair (début) au foncé (équilibre)
    colors = plt.cm.viridis(np.linspace(0.2, 1.0, len(snapshot_times)))

    for idx, (t_snap, N_t) in enumerate(snapshots.items()):
        plt.loglog(gamma, N_t * gamma**2, label=f'Numérique ($t = {t_snap: .2f}$)', color=colors[idx], lw=2)


    plt.loglog(gamma, N_theo * gamma**2, label='Analytique (Stationnaire)', color='red', ls='--', lw=2)

    plt.axvline(gamma_min, color='gray', linestyle=':', label=rf'$\gamma_{{\min}} = {gamma_min}$')
    plt.axvline(gamma_c, color='orange', linestyle=':', label=rf'$\gamma_c = {gamma_c:.2f}$')
    plt.ylim(10,10e9)
    plt.xlabel(r'Facteur de Lorentz $\gamma$', fontsize=12)
    plt.ylabel(r'$N(\gamma, t) \cdot \gamma^2$ [u.a.]', fontsize=12)
    plt.title(r'Approche de l\'équilibre et convergence vers l\'état stationnaire', fontsize=13)
    plt.legend(fontsize=9, loc='upper left')
    plt.tight_layout()
    plt.savefig('test_repere_equilibre_implicite.png', dpi=300)
    plt.show()